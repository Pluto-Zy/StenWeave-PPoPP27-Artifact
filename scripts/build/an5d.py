#!/usr/bin/env python3
"""Build frozen AN5D variants, or all shipped candidates for offline selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
try:
    import tomllib
except ImportError:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', shutil.which('nvcc')))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--all-candidates', action='store_true')
    parser.add_argument('--stencil', action='append', help='build only selected frozen stencils')
    parser.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
    args = parser.parse_args()
    if not args.nvcc or args.jobs < 1:
        parser.error('provide nvcc via --nvcc/NVCC/PATH and positive --jobs')
    root = Path(__file__).resolve().parents[2]
    build_root = (args.build_root or root / 'build').resolve()
    for line in (root / 'provenance/AN5D.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'source snapshot checksum mismatch: {relative}')
    source = build_root / 'src/AN5D' / args.arch
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(root / 'vendor/AN5D', source)
    patchdir = root / 'provenance/patches/AN5D'
    patches = [p.strip() for p in (patchdir / 'series').read_text().splitlines()
               if p.strip() and not p.startswith('#')]
    for patch in patches:
        subprocess.run(['patch', '--batch', '-p1', '-i', str(patchdir / patch)],
                       cwd=source, check=True)
    if args.all_candidates and args.stencil:
        parser.error('--all-candidates cannot be combined with --stencil')
    if args.all_candidates:
        selections = [(p.name.removesuffix('_host.cu'), 0)
                      for p in sorted((source / 'compiled/double').glob('*_host.cu'))]
    else:
        with (root / f'configs/baselines/AN5D/{args.arch}.toml').open('rb') as f:
            config = tomllib.load(f)
        if config['arch'] != args.arch:
            raise RuntimeError('configuration architecture mismatch')
        chosen = args.stencil or list(config['stencils'])
        if len(set(chosen)) != len(chosen) or set(chosen) - config['stencils'].keys():
            parser.error('unknown or duplicate stencil')
        selections = sorted((config['stencils'][s]['variant'], config['stencils'][s].get('maxrregcount', 0))
                            for s in chosen)
    binaries = build_root / 'bin/AN5D' / args.arch
    logs = build_root / 'logs/AN5D' / args.arch
    binaries.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must not leave an old success manifest usable by the runner.
    manifest_path = binaries / 'build.json'
    manifest_path.unlink(missing_ok=True)
    compiler = subprocess.check_output([args.nvcc, '--version'], text=True)

    def compile_one(selection):
        variant, cap = selection
        name = variant + (f'-rreg{cap}' if cap else '')
        command = [args.nvcc, '-DSB_TYPE=double', '-O3', '--use_fast_math',
                   f'-arch=sm_{args.arch[2:]}', '-Xptxas', '-v', '-Xcompiler', '-fopenmp',
                   '-I' + str(source), '-I' + str(source / 'compiled/double')]
        command += [str(source / 'compiled/double' / (variant + suffix))
                    for suffix in ['_host.cu', '_kernel.cu']]
        if cap:
            command += [f'--maxrregcount={cap}']
        command += ['-o', str(binaries / name)]
        with (logs / (name + '.log')).open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name}: build failed; see {logs / (name + ".log")}')
        record = dict(command=command, sha256=hashlib.sha256((binaries / name).read_bytes()).hexdigest())
        print(f'Built {args.arch}: {name}', flush=True)
        return name, record

    records = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_one, selection) for selection in selections]
        for future in as_completed(futures):
            name, record = future.result()
            records[name] = record

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = dict(arch=args.arch, compiler=compiler, source_manifest='provenance/sources.toml',
                    source_files={str(p.relative_to(source)): sha(p)
                                  for p in sorted(source.rglob('*')) if p.is_file()},
                    patches={p: sha(patchdir / p) for p in patches}, binaries=records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Built {len(records)} AN5D variants for {args.arch}', flush=True)


if __name__ == '__main__':
    main()
