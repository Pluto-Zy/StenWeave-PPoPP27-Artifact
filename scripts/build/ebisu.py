#!/usr/bin/env python3
"""Build frozen FP64 EBISU selections from verified snapshots and ordered patches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
try:
    import tomllib
except ImportError:
    import tomli as tomllib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
    ap.add_argument('--nvcc', default=os.environ.get('NVCC', shutil.which('nvcc')))
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--stencil', action='append', help='build only selected frozen stencils')
    ap.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
    args = ap.parse_args()
    if not args.nvcc or args.jobs < 1:
        ap.error('provide nvcc and positive jobs')
    root = Path(__file__).resolve().parents[2]
    build_root = (args.build_root or root / 'build').resolve()
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    for line in (root / 'provenance/EBISU.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        if sha(root / relative) != digest:
            raise RuntimeError(f'snapshot mismatch: {relative}')
    config_path = root / f'configs/baselines/EBISU/{args.arch}.toml'
    config = tomllib.loads(config_path.read_text())
    if config['arch'] != args.arch:
        raise RuntimeError('config architecture mismatch')
    chosen = args.stencil or list(config['stencils'])
    if len(set(chosen)) != len(chosen) or set(chosen) - config['stencils'].keys():
        ap.error('unknown or duplicate stencil')
    source = build_root / 'src/EBISU' / args.arch
    staged = source.with_name(source.name + '.stage')
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(root / 'vendor/EBISU', staged)
    pd = root / 'provenance/patches/EBISU'
    patches = (pd / 'series').read_text().splitlines()
    for patch in patches:
        subprocess.run(['patch', '--batch', '-p1', '-i', str(pd / patch)], cwd=staged, check=True)
    # Preserve mtimes for unchanged inputs so unrelated selected binaries stay intact.
    source.mkdir(parents=True, exist_ok=True)
    for p in staged.rglob('*'):
        dest = source / p.relative_to(staged)
        if p.is_file() and (not dest.exists() or p.read_bytes() != dest.read_bytes()):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    for p in source.rglob('*'):
        if p.is_file() and not (staged / p.relative_to(source)).is_file():
            p.unlink()
    shutil.rmtree(staged)
    bindir = build_root / 'bin/EBISU' / args.arch
    logdir = build_root / 'logs/EBISU' / args.arch
    bindir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    manifest_path = bindir / 'build.json'
    manifest_path.unlink(missing_ok=True)
    nvcc = str(Path(args.nvcc).resolve())
    compiler = subprocess.check_output([nvcc, '--version'], text=True)
    major = int(re.search(r'release (\d+)', compiler)[1])
    def build(item):
        stencil, c = item
        directory = build_root / 'EBISU' / args.arch / stencil
        flags = [f'-DEBISU_TUNE_TSTEP_FP64={c["tstep"]}', f'-DEBISU_TUNE_IPTS_FP64={c["ipts"]}']
        if c['dim'] == 3:
            flags += [f'-DEBISU_TUNE_TILE_X_FP64={c["tile_x"]}']
            if major >= 13:
                flags += ['-static-global-template-stub=false']
        commands = [
            ['cmake', '-S', str(source / f'{c["dim"]}dstencil'), '-B', str(directory),
             '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_COMPILER={nvcc}',
             f'-DCUDA_TOOLKIT_ROOT_DIR={Path(nvcc).parent.parent}',
             f'-DEBISU_CUDA_ARCHITECTURES={args.arch[2:]}', '-DCMAKE_CUDA_STANDARD=17',
             f'-DEBISU_RUNTIME_OUTPUT_DIRECTORY={bindir / stencil}',
             '-DCMAKE_CUDA_FLAGS=' + ' '.join(flags)],
            ['cmake', '--build', str(directory), '--target', c['target'], '--parallel', '2']]
        with (logdir / f'{stencil}.log').open('w') as stream:
            for cmd in commands:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        binary = bindir / stencil / c['target']
        print(f'Built {args.arch} {stencil}', flush=True)
        return stencil, dict(path=os.path.relpath(binary, root), sha256=sha(binary), commands=commands,
                             config=c)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = dict(pool.map(build, [(s, config['stencils'][s]) for s in chosen]))
    manifest_path.write_text(json.dumps(dict(arch=args.arch, compiler=compiler,
        config_sha256=sha(config_path), binaries=records,
        source_files={str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
        patches={p: sha(pd / p) for p in patches}), indent=2) + '\n')


if __name__ == '__main__':
    main()
