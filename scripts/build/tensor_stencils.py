"""Build selected ConvStencil or LoRAStencil FP64 binaries and validation binaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def main(implementation):
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
    for line in (root / f'provenance/{implementation}.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        if sha(root / relative) != digest:
            raise RuntimeError(f'snapshot mismatch: {relative}')
    cp = root / f'configs/baselines/{implementation}/{args.arch}.toml'
    config = tomllib.loads(cp.read_text())
    chosen = args.stencil or list(config['stencils'])
    if len(set(chosen)) != len(chosen) or set(chosen) - config['stencils'].keys():
        ap.error('unknown or duplicate stencil')
    source = build_root / f'src/{implementation}/{args.arch}'
    stage = source.with_name(source.name + '.stage')
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(root / f'vendor/{implementation}', stage)
    pd = root / f'provenance/patches/{implementation}'
    patches = (pd / 'series').read_text().splitlines()
    for patch in patches:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(pd / patch)], cwd=stage, check=True)
    source.mkdir(parents=True, exist_ok=True)
    for p in stage.rglob('*'):
        dest = source / p.relative_to(stage)
        if p.is_file() and (not dest.exists() or p.read_bytes() != dest.read_bytes()):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    shutil.rmtree(stage)
    bindir = build_root / f'bin/{implementation}/{args.arch}'
    logdir = build_root / f'logs/{implementation}/{args.arch}'
    bindir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    manifest_path = bindir / 'build.json'
    manifest_path.unlink(missing_ok=True)
    nvcc = str(Path(args.nvcc).resolve())
    compiler = subprocess.check_output([nvcc, '--version'], text=True)
    def build(item):
        stencil, c = item
        records = {}
        for mode in ['performance', 'check']:
            directory = build_root / f'{implementation}/{args.arch}/{stencil}/{mode}'
            flags = [f'-DBLOCK_SIZE_ROW={c["row"]}', f'-DBLOCK_SIZE_COL={c["col"]}', f'-DWARP_PER_BLOCK={c["warps"]}']
            if mode == 'check':
                flags += ['-DCHECK_ERROR']
            # CUDA 13 defaults template stubs to static, incompatible with these sources.
            if 'release 13.' in compiler:
                flags += ['-static-global-template-stub=false']
            prefix = 'conv' if implementation == 'ConvStencil' else 'lora'
            target = f'{prefix}stencil_{c["dim"]}d'
            commands = [
                ['cmake', '-S', str(source), '-B', str(directory), '-DCMAKE_BUILD_TYPE=Release',
                 f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_ARCHITECTURES={args.arch[2:]}',
                 '-DCMAKE_CUDA_FLAGS=' + ' '.join(flags), f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={bindir / stencil / mode}'],
                ['cmake', '--build', str(directory), '--target', target, '--parallel', '2']]
            with (logdir / f'{stencil}-{mode}.log').open('w') as stream:
                for cmd in commands:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
            binary = bindir / stencil / mode / target
            records[mode] = dict(path=os.path.relpath(binary, root), sha256=sha(binary), commands=commands)
        print(f'Built {implementation} {args.arch} {stencil}', flush=True)
        return stencil, dict(config=c, **records)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = dict(pool.map(build, [(s, config['stencils'][s]) for s in chosen]))
    manifest_path.write_text(json.dumps(dict(implementation=implementation, arch=args.arch, compiler=compiler,
        config_sha256=sha(cp), binaries=records,
        source_files={str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
        patches={p: sha(pd / p) for p in patches}), indent=2) + '\n')
