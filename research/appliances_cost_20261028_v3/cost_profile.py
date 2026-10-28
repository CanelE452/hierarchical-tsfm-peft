"""Isolated v3 routing around the frozen v2 inference measurement core."""
from research.appliances_cost_20261028_v2 import cost_profile as core

NAME = 'appliances_cost_20261028_v3'
REPORT = core.runtime.REPO / 'reports' / NAME
CACHE = core.runtime.REPO / '.cache' / NAME
core.runtime.NAME = NAME
core.runtime.OUTPUT_REPORT = REPORT
core.runtime.OUTPUT_CACHE = CACHE
core.COST_REPORT = REPORT / 'cost_profile'
core.COST_CACHE = CACHE / 'cost_profile'
core.host_gpu_gate.MODULES = core.host_gpu_gate.MODULES | {f'research.{NAME}.cost_profile'}

if __name__ == '__main__':
    core.main()
