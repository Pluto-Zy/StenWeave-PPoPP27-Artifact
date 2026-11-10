#!/usr/bin/env python3
"""Run frozen FP64 EBISU cases, retaining native logs and independent trial rates."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'adapters'))
from ebisu import check_dimensions, command, device_info, parse_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['benchmark', 'size-sweep'], default='benchmark')
    ap.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
    ap.add_argument('--device', required=True)
    ap.add_argument('--stencil', action='append')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--dimensions', nargs='+', type=int, help='native order: Y X in 2D; X Y Z in 3D')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--repeats', type=int)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
    args = ap.parse_args()
    if any(v is not None and v < 1 for v in [args.steps, args.repeats]):
        ap.error('steps/repeats must be positive')
    root = Path(__file__).resolve().parents[3]
    build_root = (args.build_root or root / 'build').resolve()
    config_path = root / f'configs/baselines/EBISU/{args.arch}.toml'
    config = tomllib.loads(config_path.read_text())
    if config['statistic'] not in ['mean_sample_throughput', 'median_sample_throughput']:
        raise ValueError('unsupported statistic')
    if config.get('runtime_mode', 'explicit') not in ['experiment', 'explicit']:
        raise ValueError('unsupported runtime mode')
    chosen = args.stencil or [s for s, c in config['stencils'].items()
                             if args.suite != 'size-sweep' or c['dim'] == 2]
    if set(chosen)-config['stencils'].keys() or len(chosen) != len(set(chosen)):
        ap.error('unknown or duplicate stencil')
    if args.suite == 'size-sweep' and any(config['stencils'][s]['dim'] == 3 for s in chosen):
        ap.error('EBISU is not evaluated for 3D size sweeps')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((build_root / 'bin/EBISU' / args.arch / 'build.json').read_text())
    if manifest['arch'] != args.arch or config['arch'] != args.arch or manifest['config_sha256'] != sha(config_path):
        raise RuntimeError('build/config mismatch; rebuild the selected architecture')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder = (args.output_dir or root / 'runs' / f'ebisu-{args.arch}-{stamp}').resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        ap.error('output directory must be empty')
    environment = dict(utc=stamp, arch=args.arch, device=args.device, check=args.check,
                       gpu=device_info(args.device, args.arch), build=manifest,
                       config=str(config_path.relative_to(root)), config_sha256=sha(config_path))
    (folder / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    for stencil in chosen:
        c = config['stencils'][stencil]
        b = manifest['binaries'][stencil]
        binary = root / b['path']
        if sha(binary) != b['sha256'] or b['config'] != c:
            raise RuntimeError('binary/config mismatch')
        dims = args.dimensions or (check_dimensions(c, args.arch) if args.check else c['dimensions'])
        steps = args.steps or c['tstep']
        repeats = args.repeats or (1 if args.check else config['repeats'])
        cmd = command(binary, c, dims, steps, args.check)
        runtime_mode = 'explicit'
        if not args.check and args.dimensions is None and args.steps is None and config.get('runtime_mode') == 'experiment':
            cmd = [str(binary), '--experiment', '--verbose']
            runtime_mode = 'experiment'
        trials, logs = [], []
        for i in range(repeats):
            log = folder / f'{stencil}-{i}.log'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS='8')
            (folder / f'{stencil}-{i}-command.json').write_text(json.dumps(dict(
                argv=cmd, CUDA_VISIBLE_DEVICES=str(args.device), OMP_NUM_THREADS='8'), indent=2)+'\n')
            with log.open('w') as stream:
                proc = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            if proc.returncode:
                raise RuntimeError(f'EBISU exit {proc.returncode}: {log}')
            trials.append(parse_output(log.read_text(), c, dims, steps, args.check))
            logs.append(str(log))
        row = dict(implementation='EBISU', arch=args.arch, device=args.device, stencil=stencil,
            variant='native_fp64', experiment='validation' if args.check else args.suite,
            config=str(config_path.relative_to(root)), config_id=stencil, parameters=c,
            dimensions=dims, dimension_order='YX' if c['dim']==2 else 'XYZ', steps=steps,
            repeats=repeats, trials=trials, runtime_mode=runtime_mode, command=cmd, log_paths=logs, binary_sha256=b['sha256'],
            input='native rand()/(RAND_MAX-1)', boundary='native valid interior',
            work_definition='product(dimension-2*radius)*steps; native Valid Performance',
            timing='CUDA events around dependent cooperative launches; allocation/copies/warmup excluded')
        if not args.check:
            rates = [t['gstencils_per_s'] for t in trials]
            row.update(gstencils_per_s=(statistics.mean(rates) if config['statistic'] == 'mean_sample_throughput' else statistics.median(rates)), statistic=config['statistic'],
                       mean_gstencils_per_s=statistics.mean(rates), median_gstencils_per_s=statistics.median(rates))
        with (folder / ('checks.jsonl' if args.check else 'results.jsonl')).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(stencil, trials[0]['validation'] if args.check else row['gstencils_per_s'], flush=True)
    (folder / 'end.json').write_text(json.dumps(dict(completed=len(chosen), check=args.check))+'\n')
    print(f'Run saved: {folder}')


if __name__ == '__main__':
    main()
