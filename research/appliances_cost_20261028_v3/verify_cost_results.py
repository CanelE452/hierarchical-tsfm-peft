"""CPU verification routed to v3 receipts using the frozen v2 validator."""
import argparse
import json
from research.appliances_cost_20261028_v2 import verify_cost_results as core

NAME = 'appliances_cost_20261028_v3'
REPORT = core.REPO / 'reports' / NAME / 'cost_profile'
core.NAME = NAME
core.REPORT = REPORT
core.host_gpu_gate.MODULES = core.host_gpu_gate.MODULES | {f'research.{NAME}.cost_profile'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate without writing COST_VALIDATION.json')
    args = parser.parse_args()
    value = core.validate(REPORT)
    destination = REPORT / 'COST_VALIDATION.json'
    if not args.check_only:
        with destination.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=value['status'], facets=len(value['facets']), written=not args.check_only,
                          path=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
