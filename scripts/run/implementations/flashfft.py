#!/usr/bin/env python3
"""Run frozen FlashFFT configurations and retain logs and successful measurements."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
try:
    import tomllib
except ImportError:
    import tomli as tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--suite', choices=['benchmark'], default='benchmark')
parser.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
parser.add_argument('--device', required=True)
parser.add_argument('--stencil', action='append', help='repeat to select several; default: all')
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--check', action='store_true', help='small random-input CPU comparison')
parser.add_argument('--width', type=int)
parser.add_argument('--steps', type=int)
parser.add_argument('--repeats', type=int)
parser.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
build_root = (args.build_root or root / 'build').resolve()
config_path = root / f'configs/baselines/FlashFFTStencil/{args.arch}.toml'
with config_path.open('rb') as stream:
    config = tomllib.load(stream)
chosen = args.stencil or list(config['stencils'])
if set(chosen) - config['stencils'].keys():
    raise SystemExit(f'unknown stencils: {sorted(set(chosen) - config["stencils"].keys())}')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
folder = args.output_dir or root / 'runs' / f'flashfft-{args.arch}-{stamp}'
folder = folder.resolve()
folder.mkdir(parents=True, exist_ok=True)
if (folder / 'results.jsonl').exists():
    raise SystemExit('output already contains results.jsonl; choose a fresh output directory')
binary_dir = build_root / 'bin/FlashFFTStencil' / args.arch
manifest = binary_dir / 'build.json'
environment = {
    'utc': stamp, 'device': args.device, 'arch': args.arch,
    'config': str(config_path.relative_to(root)), 'check': args.check,
    'build': json.loads(manifest.read_text()),
    'gpu': subprocess.check_output(['nvidia-smi', '-i', args.device,
            '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip(),
}
(folder / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
for name in chosen:
    c = config['stencils'][name]
    dim = c['dim']
    width = args.width if args.width is not None else 65 if args.check else c['width']
    steps = args.steps if args.steps is not None else 2*c['fusion']+1 if args.check else config['steps']
    repeats = args.repeats if args.repeats is not None else 1 if args.check else config[f'repeats_{dim}d']
    input_kind = 'random' if args.check else config['input']
    command = [str(binary_dir / f'flashfft{dim}d'), '--shape', c['shape'],
               '--radius', str(c['radius']), '--fusion', str(c['fusion']),
               '--width', str(width), '--steps', str(steps), '--input', input_kind,
               '--warmups', str(0 if args.check else config['warmups']), '--repeats', str(repeats)]
    if args.check:
        command += ['--check']
    log_path = folder / f'{name}.log'
    (folder / f'{name}-command.json').write_text(json.dumps(command, indent=2) + '\n')
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.device),
                            cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=3600)
    log_path.write_text(result.stdout)
    if result.returncode:
        raise SystemExit(f'{name} failed ({result.returncode}); see {log_path}')
    if (f'arch={args.arch} ' not in result.stdout or f'input={input_kind} ' not in result.stdout
            or f'coefficients={config["coefficients"]} ' not in result.stdout):
        raise SystemExit(f'unexpected device, input, or coefficients in {log_path}')
    if args.check:
        if 'check_max_abs=' not in result.stdout:
            raise SystemExit(f'missing correctness result: {log_path}')
        print(f'{name}: {result.stdout.splitlines()[0]}', flush=True)
        continue
    time = re.search(r'^Time = ([0-9.eE+-]+)\[ms\]$', result.stdout, re.M)
    rate = re.search(r'^GStencil/s = ([0-9.eE+-]+)$', result.stdout, re.M)
    trials = re.search(r'^trials_ms=(.*)$', result.stdout, re.M)
    if not time or not rate or not trials:
        raise SystemExit(f'missing timing results: {log_path}')
    ms, gstencils = float(time[1]), float(rate[1])
    times = [float(v) for v in trials[1].split(',')]
    if (len(times) != repeats or any(not math.isfinite(v) or v <= 0 for v in [ms, gstencils, *times])
            or not math.isclose(gstencils, width**dim*steps/ms/1e6, rel_tol=1e-7)):
        raise SystemExit(f'invalid timing results: {log_path}')
    row = dict(experiment='benchmark', implementation='FlashFFTStencil', arch=args.arch,
               stencil=name, config=str(config_path.relative_to(root)), dim=dim,
               radius=c['radius'], fusion=c['fusion'], width=width, steps=steps,
               input=input_kind, coefficients=config['coefficients'], median_ms=ms,
               gstencils_per_s=gstencils, trials_ms=times, command=command, log_path=str(log_path))
    with (folder / 'results.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(f'{name}: {gstencils:.3f} GStencil/s', flush=True)
print(f'Run saved: {folder}')
