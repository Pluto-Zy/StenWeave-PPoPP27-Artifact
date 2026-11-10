#!/usr/bin/env python3
"""Run frozen FP64 AN5D variants and archive commands, logs and measurements."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'adapters'))
from an5d import device_info, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['benchmark', 'size-sweep'], default='benchmark')
    parser.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--stencil', action='append')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--width', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
    args = parser.parse_args()
    if any(v is not None and v < 1 for v in [args.width, args.steps, args.repeats]):
        parser.error('width, steps and repeats must be positive')
    root = Path(__file__).resolve().parents[3]
    build_root = (args.build_root or root / 'build').resolve()
    config_path = root / f'configs/baselines/AN5D/{args.arch}.toml'
    with config_path.open('rb') as stream:
        config = tomllib.load(stream)
    if config['arch'] != args.arch:
        raise RuntimeError('configuration architecture mismatch')
    chosen = args.stencil or list(config['stencils'])
    if len(chosen) != len(set(chosen)) or set(chosen) - config['stencils'].keys():
        parser.error('unknown or duplicate stencil')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = (args.output_dir or root / 'runs' / f'an5d-{args.arch}-{stamp}').resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        parser.error('output directory must be empty')
    manifest_path = build_root / 'bin/AN5D' / args.arch / 'build.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['arch'] != args.arch:
        raise RuntimeError('build architecture mismatch')
    environment = dict(utc=stamp, arch=args.arch, device=args.device, check=args.check,
                       gpu=device_info(args.device, args.arch), build=manifest,
                       config=str(config_path.relative_to(root)),
                       config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    (folder / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    for stencil in chosen:
        c = config['stencils'][stencil]
        cap = c.get('maxrregcount', 0)
        name = c['variant'] + (f'-rreg{cap}' if cap else '')
        binary = build_root / 'bin/AN5D' / args.arch / name
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binaries'][name]['sha256']:
            raise RuntimeError(f'binary differs from build manifest: {binary}')
        width = args.width if args.width is not None else (512 if c['dim'] == 2 else 128) if args.check else c['width']
        steps = args.steps if args.steps is not None else 2*c['bt']+1 if args.check else config['steps']
        repeats = args.repeats if args.repeats is not None else 1 if args.check else config['repeats']
        row = execute(root, args.arch, args.device, name, width, steps, repeats, folder, args.check, build_root=build_root)
        row.update(experiment='validation' if args.check else args.suite,
                   config=str(config_path.relative_to(root)), config_id=name,
                   maxrregcount=cap, bs1=c['bs1'], bs2=c.get('bs2'), bt=c['bt'], sl=c['sl'])
        destination = 'checks.jsonl' if args.check else 'results.jsonl'
        with (folder / destination).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        message = str(row['validation']) if args.check else f'{row["gstencils_per_s"]:.4f} GStencil/s'
        print(f'{stencil} ({name}): {message}', flush=True)
    (folder / 'end.json').write_text(json.dumps(dict(completed=len(chosen), check=args.check)) + '\n')
    print(f'Run saved: {folder}')


if __name__ == '__main__':
    main()
