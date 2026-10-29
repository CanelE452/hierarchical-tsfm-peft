"""CPU-only single owner for durable controller restarts; never dispatches GPU work itself."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from . import runtime, run_one, controller


def stamp():
    return datetime.now(timezone.utc).isoformat()


def is_controller(record):
    return bool(record and controller.module_name(record) == f'research.{runtime.NAME}.controller')


def matching_controller(pid, created):
    record=controller.process_record(pid)
    return record if controller.same_identity(record,pid,created) and is_controller(record) else None


def controller_candidates(not_before=None):
    records=[]
    for process in psutil.process_iter(['pid']):
        record=controller.process_record(process.pid)
        if (is_controller(record) and '--preflight-only' not in record['command']
                and Path(record['cwd']).resolve()==runtime.REPO.resolve()
                and (not_before is None or record['created']>=not_before-2)):
            records.append(record)
    return sorted(records,key=lambda r:(r['created'],r['pid']))


def restart_decision(progress, return_code):
    status=progress.get('status')
    if status in controller.TERMINAL_STATES:
        return 'TERMINAL'
    if return_code == 20:
        return 'UNKNOWN_SUSPEND'
    if status == 'CONTROLLER_CRASHED' or return_code in (70,-9,-15,137,143,3221225786,3221226505,None):
        return 'RESTART_TECHNICAL'
    return 'UNKNOWN_SUSPEND'


class Supervisor:
    def __init__(self):
        self.path=runtime.REPORT/'SUPERVISOR_STATUS.json'
        self.progress_path=runtime.REPORT/'PROGRESS.json'
        protocol=runtime.read(runtime.REPORT/'PREREGISTRATION.json')
        self.maximum=int(protocol.get('execution_policy',{}).get('max_controller_restarts',3))
        if self.maximum != 3:
            raise ValueError('Registered supervisor restart limit changed')
        self.poll_seconds=float(protocol.get('execution_policy',{}).get('heartbeat_seconds',10))
        self.state=runtime.read(self.path) if self.path.exists() else dict(status='READY',created_utc=stamp(),
            restart_count=0,launch_count=0,launches=[])
        self.state.setdefault('launches',[])
        self.state.setdefault('restart_count',0)
        self.state.setdefault('launch_count',len(self.state['launches']))
        self.seal=runtime.sha(runtime.REPORT/'REGISTRATION_SEAL.json')
        previous=self.state.get('registration_seal_sha256')
        if previous and previous!=self.seal:
            raise ValueError('Supervisor cannot resume a different registration seal')
        self.state['registration_seal_sha256']=self.seal

    def save(self):
        self.state.update(supervisor_pid=os.getpid(),supervisor_creation_time=psutil.Process().create_time(),updated_utc=stamp())
        runtime.atomic(self.path,self.state)

    def progress(self):
        return runtime.read(self.progress_path) if self.progress_path.exists() else {}

    def stop_requested(self):
        return (runtime.CACHE/'USER_STOP_REQUEST.json').exists() or (runtime.REPORT/'USER_STOP_REQUEST.json').exists()

    def existing_controller(self):
        progress=self.progress()
        pid,created=progress.get('controller_pid'),progress.get('controller_creation_time')
        record=matching_controller(pid,created) if pid and created else None
        candidates=controller_candidates()
        return candidates[-1] if candidates else record

    def wait_existing(self,record,process=None):
        cancelled_at=None
        not_before=record['created']
        while True:
            candidates=controller_candidates(not_before)
            current=matching_controller(record['pid'],record['created'])
            if not current and not candidates:
                break
            if candidates:
                record=candidates[-1]
            self.state['active_controller']=dict(pid=record['pid'],created=record['created'])
            self.state.update(status='OBSERVING_CONTROLLER',controller_pid=record['pid'],
                              controller_creation_time=record['created'])
            self.save()
            if self.stop_requested():
                cancelled_at=cancelled_at or time.time()
                if time.time()-cancelled_at>30:
                    active=self.progress().get('active')
                    if active and active.get('descriptor'):
                        controller.terminate_owned(active['descriptor'],active['attempt'],active['started_unix'])
                    current=matching_controller(record['pid'],record['created'])
                    if current:
                        psutil.Process(current['pid']).terminate()
                    self.state.update(status='USER_CANCELLED',finished_utc=stamp())
                    self.save()
                    return 2
            time.sleep(self.poll_seconds)
        if process is not None:
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return None
        return None

    def run(self):
        if self.state.get('status') in ('RESTART_LIMIT_REACHED','SCIENTIFIC_SUSPEND','UNKNOWN_SUSPEND','USER_CANCELLED'):
            return 2 if self.state['status']=='USER_CANCELLED' else 70 if self.state['status']=='RESTART_LIMIT_REACHED' else 20
        plan=runtime.read(runtime.REPORT/'RUN_PLAN.json')
        controller.checked_registration(plan['jobs'][0])
        while True:
            progress=self.progress()
            if self.stop_requested() or progress.get('status')=='USER_CANCELLED':
                self.state.update(status='USER_CANCELLED',finished_utc=stamp())
                self.save()
                return 2
            if progress.get('status') in controller.TERMINAL_STATES:
                self.state.update(status='TERMINAL',training_status=progress['status'],finished_utc=stamp())
                self.save()
                return 0 if progress['status']=='TRAINING_COMPLETE' else 1
            record=self.existing_controller()
            process=None
            if record is None:
                # Restore durable ownership after a supervisor died between Popen and progress creation.
                saved=self.state.get('active_controller')
                if saved:
                    record=matching_controller(saved.get('pid'),saved.get('created'))
            if record is None:
                controller.checked_registration(plan['jobs'][0])
                ordinal=self.state['launch_count']+1
                directory=runtime.CACHE/'controller_runs'
                directory.mkdir(parents=True,exist_ok=True)
                stdout=directory/f'controller_{ordinal:03d}_stdout.log'
                stderr=directory/f'controller_{ordinal:03d}_stderr.log'
                with stdout.open('x',encoding='utf8') as out, stderr.open('x',encoding='utf8') as err:
                    process=subprocess.Popen(controller.command('controller'),cwd=runtime.REPO,
                        env=controller.environment(),stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                record=controller.process_record(process.pid)
                if record is None:
                    candidates=controller_candidates()
                    record=candidates[-1] if candidates else None
                launch=dict(pid=process.pid,created=record['created'] if record else None,started_utc=stamp(),
                            ordinal=ordinal,stdout=str(stdout),stderr=str(stderr))
                self.state['launch_count']=ordinal
                self.state['launches'].append(launch)
                self.state['active_controller']=dict(pid=record['pid'],created=record['created']) if record else None
                self.save()
                if record is None:
                    return_code=process.wait(timeout=10)
                else:
                    return_code=self.wait_existing(record,process)
            else:
                return_code=self.wait_existing(record)
            progress=self.progress()
            decision=restart_decision(progress,return_code)
            self.state['active_controller']=None
            if decision=='TERMINAL':
                self.state.update(status='TERMINAL',training_status=progress.get('status'),finished_utc=stamp())
                self.save()
                return 0 if progress.get('status')=='TRAINING_COMPLETE' else 1
            if self.stop_requested() or self.state.get('status')=='USER_CANCELLED':
                self.state.update(status='USER_CANCELLED',finished_utc=stamp())
                self.save()
                return 2
            if decision!='RESTART_TECHNICAL':
                self.state.update(status='UNKNOWN_SUSPEND',controller_exit_code=return_code,finished_utc=stamp())
                self.save()
                return 20
            if self.state['restart_count']>=self.maximum:
                self.state.update(status='RESTART_LIMIT_REACHED',controller_exit_code=return_code,finished_utc=stamp())
                self.save()
                return 70
            controller.checked_registration(plan['jobs'][0])
            self.state['restart_count']+=1
            self.state.update(status='RESTARTING_TECHNICAL',last_controller_exit=return_code)
            self.save()
            time.sleep(min(30,self.poll_seconds))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    runtime.install_write_guard()
    supervisor=None
    try:
        with runtime.CampaignLock(runtime.CACHE/'supervisor.lock'):
            supervisor=Supervisor()
            controller.prevent_sleep(True)
            return supervisor.run()
    except controller.ScientificSuspend as exc:
        if supervisor:
            supervisor.state.update(status='SCIENTIFIC_SUSPEND',error=str(exc),finished_utc=stamp())
            supervisor.save()
        raise
    except KeyboardInterrupt:
        marker=runtime.CACHE/'USER_STOP_REQUEST.json'
        if not marker.exists():
            runtime.write_new(marker,dict(requested_utc=stamp(),reason='Supervisor received KeyboardInterrupt'))
        if supervisor:
            supervisor.state.update(status='USER_CANCELLED',finished_utc=stamp())
            supervisor.save()
        return 2
    finally:
        controller.prevent_sleep(False)


if __name__=='__main__':
    raise SystemExit(main())
