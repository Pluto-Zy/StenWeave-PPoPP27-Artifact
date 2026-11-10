"""Run frozen FP64 tensor stencil selections with raw logs and explicit normalization."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'adapters'))
from tensor_output import device_info, parse_output


def main(implementation):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['benchmark', 'size-sweep'], default='benchmark')
    ap.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
    ap.add_argument('--device', required=True)
    ap.add_argument('--stencil', action='append')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--memcheck', action='store_true')
    ap.add_argument('--dimensions', nargs='+', type=int, help='native order: M N or H M N')
    ap.add_argument('--iterations', type=int)
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
    args = ap.parse_args()
    if any(v is not None and v < 1 for v in [args.iterations, args.repeats]):
        ap.error('iterations/repeats must be positive')
    if args.memcheck and not args.check:
        ap.error('--memcheck requires --check')
    root = Path(__file__).resolve().parents[3]
    build_root = (args.build_root or root / 'build').resolve()
    cp = root / f'configs/baselines/{implementation}/{args.arch}.toml'
    config = tomllib.loads(cp.read_text())
    chosen = args.stencil or list(config['stencils'])
    if set(chosen) - config['stencils'].keys() or len(chosen) != len(set(chosen)):
        ap.error('unknown or duplicate stencil')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((build_root / f'bin/{implementation}/{args.arch}/build.json').read_text())
    if manifest['arch'] != args.arch or config['arch'] != args.arch or manifest['config_sha256'] != sha(cp):
        raise RuntimeError('build/config mismatch; rebuild')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = (args.output_dir or root / 'runs' / f'{implementation.lower()}-{args.arch}-{stamp}').resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        ap.error('output directory must be empty')
    environment = dict(utc=stamp, arch=args.arch, device=args.device, check=args.check,
        gpu=device_info(args.device, args.arch), build=manifest, config_sha256=sha(cp))
    (folder / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    for stencil in chosen:
        c = config['stencils'][stencil]
        b = manifest['binaries'][stencil]
        mode = 'check' if args.check else 'performance'
        binary = root / b[mode]['path']
        if sha(binary) != b[mode]['sha256'] or b['config'] != c:
            raise RuntimeError('binary/config mismatch')
        defaults = ([256, 256] if c['dim'] == 2 else [64, 64, 64]) if args.check else c['dimensions']
        dims = args.dimensions or defaults
        if len(dims) != c['dim'] or any(v <= 0 for v in dims) or dims[-1] % c['col'] or dims[-2] % c['row']:
            ap.error('dimensions must be positive, match rank, and M/N must divide tile row/col')
        iterations = args.iterations or (1 if args.check else c['iterations'])
        repeats = args.repeats or (1 if args.check else config['repeats'])
        cmd = [str(binary), stencil, *map(str,dims), str(iterations)]
        if args.memcheck:
            sanitizer = shutil.which('compute-sanitizer')
            if not sanitizer:
                raise RuntimeError('compute-sanitizer is required')
            cmd = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', *cmd]
        trials, logs = [], []
        for i in range(repeats):
            log = folder / f'{stencil}-{i}.log'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS='8')
            (folder / f'{stencil}-{i}-command.json').write_text(json.dumps(dict(argv=cmd,
                CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS='8'), indent=2)+'\n')
            with log.open('w') as stream:
                proc = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            if proc.returncode:
                raise RuntimeError(f'{implementation} exit {proc.returncode}: {log}')
            output = log.read_text()
            if args.memcheck and 'ERROR SUMMARY: 0 errors' not in output:
                raise RuntimeError(f'missing clean memcheck summary: {log}')
            trials.append(parse_output(output, c, dims, iterations, args.check))
            logs.append(str(log))
        row = dict(implementation=implementation, arch=args.arch, device=args.device, stencil=stencil,
            variant='native_fp64', experiment='validation' if args.check else args.suite,
            config=str(cp.relative_to(root)), config_id=stencil, parameters=c, dimensions=dims,
            dimension_order='MN' if c['dim']==2 else 'HMN', iterations=iterations, repeats=repeats,
            trials=trials, command=cmd, log_paths=logs, binary_sha256=b[mode]['sha256'],
            input='native deterministic rand()%100', boundary='native allocated halo',
            native_work_factor=c['native_work_factor'], normalization_divisor=c['normalization_divisor'],
            work_definition='product(dimensions)*iterations*native_work_factor/normalization_divisor',
            timing='native host steady_clock around dependent launches and synchronization; allocation/copies excluded')
        if not args.check:
            rates = [t['gstencils_per_s'] for t in trials]
            row.update(gstencils_per_s=statistics.mean(rates), statistic='mean_sample_throughput',
                median_gstencils_per_s=statistics.median(rates), native_gstencils_per_s=statistics.mean([t['native_gstencils_per_s'] for t in trials]))
        with (folder / ('checks.jsonl' if args.check else 'results.jsonl')).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(stencil, trials[0]['validation'] if args.check else row['gstencils_per_s'], flush=True)
    (folder / 'end.json').write_text(json.dumps(dict(completed=len(chosen), check=args.check))+'\n')
    print(f'Run saved: {folder}')
