"""Create or check a private dev asset inventory; no fitting, raw-video scans or downloads."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from training.btc_joint.assets import create_manifest, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for action in ('create', 'check'):
        cli = sub.add_parser(action)
        cli.add_argument('--reports', type=Path, required=True)
        cli.add_argument('--manifest', type=Path, required=True, help='Private JSON inventory; keep outside tracked source')
        cli.add_argument('--purpose', choices=['fit', 'replay', 'both'], default='both')
    args = p.parse_args()
    purposes = ['fit', 'replay'] if args.purpose == 'both' else [args.purpose]
    try:
        if args.command == 'create':
            create_manifest(args.reports, args.manifest, purposes)
        for purpose in purposes:
            verify(args.reports, args.manifest, purpose)
    except (ValueError, KeyError, OSError) as exc:
        p.exit(2, f'BLOCKED: {exc}\nSee docs/BTC_JOINT_REPRODUCING.md.\n')
    print('PASS: declared dev inventory, cohort and all input hashes; no model was fitted.')


if __name__ == '__main__':
    main()
