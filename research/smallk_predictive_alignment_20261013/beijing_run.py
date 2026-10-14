"""Retry only the Beijing loader-failed path under the unchanged method lock."""
from pathlib import Path
import argparse
import new_run as original
import newdata_repair as repair
import engine as en


class RepairAPI:
    @staticmethod
    def load_training(dataset,method_lock_path,**kwargs):
        source=en.CACHE/'raw_dataset'/'beijing_aotizhongxin'/'download'/'uci_501.zip'
        return repair.load_training(dataset,method_lock_path,source_path=source,allow_download=False)
    required_model_selection_fields=staticmethod(repair.required_model_selection_fields)
    open_evaluation=staticmethod(repair.open_evaluation)


def run(tag,devtag):
    previous_snapshots=original.snapshots
    previous_controls=original.add_controls
    def snapshots(output):
        records=previous_snapshots(output)
        paths=[Path(__file__),en.HERE/'newdata_repair.py',en.HERE/'BEIJING_REPAIR_EXECUTION_NOTE_KO.md']
        for path in paths:
            digest=en.ex.sha(path)
            with (output/'sources'/path.name).open('xb') as f: f.write(path.read_bytes())
            assert en.ex.sha(output/'sources'/path.name)==digest
            records[str(path.relative_to(en.ROOT))]=digest
        return records
    def controls(data,*args,**kwargs):
        if data['dataset'] in ('jena','robin'):
            return
        return previous_controls(data,*args,**kwargs)
    original.NEW_DATASETS=('beijing_aotizhongxin',)
    original.newdata=RepairAPI
    original.snapshots=snapshots
    original.add_controls=controls
    original.run(tag,devtag)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); parser.add_argument('--devtag',default='dev_v1')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('SAFE_TAG_REQUIRED')
    run(args.tag,args.devtag)
