#!/usr/bin/env python3
"""Build the fixed FlashFFT snapshot and patches for one architecture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arch', choices=['sm80', 'sm90'], required=True)
parser.add_argument('--nvcc', default=os.environ.get('NVCC'))
parser.add_argument('--dim', type=int, choices=(2, 3), action='append', help='build selected dimensions')
parser.add_argument('--build-root', type=Path, help='independent build tree (default: build/)')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build_root = (args.build_root or root / 'build').resolve()
dimensions = list(dict.fromkeys(args.dim or (2, 3)))
nvcc = args.nvcc or (str(Path(os.environ['CUDA_ROOT']) / 'bin/nvcc')
                     if 'CUDA_ROOT' in os.environ else shutil.which('nvcc'))
if not nvcc and Path('/usr/local/cuda/bin/nvcc').is_file():
    nvcc = '/usr/local/cuda/bin/nvcc'
if not nvcc:
    raise SystemExit('nvcc not found; set NVCC or CUDA_ROOT, or pass --nvcc')
source = build_root / 'src/FlashFFTStencil' / args.arch
if source.exists():
    shutil.rmtree(source)
shutil.copytree(root / 'vendor/FlashFFTStencil', source)
patchdir = root / 'provenance/patches/FlashFFTStencil'
patches = [s.strip() for s in (patchdir / 'series').read_text().splitlines()
           if s.strip() and not s.startswith('#')]
for patch in patches:
    subprocess.run(['patch', '--batch', '-p1', '-i', str(patchdir / patch)],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)
binary_dir = build_root / 'bin/FlashFFTStencil' / args.arch
logs = build_root / 'logs/FlashFFTStencil' / args.arch
binary_dir.mkdir(parents=True, exist_ok=True)
logs.mkdir(parents=True, exist_ok=True)
commands = []
for dim in dimensions:
    command = [nvcc, '-std=c++17', '-O3', '--use_fast_math',
               f'-arch=sm_{args.arch[2:]}', '-rdc=true', f'-DDIM={dim}',
               '-I' + str(source), str(root / 'scripts/adapters/flashfft_driver.cu'),
               str(source / f'src/{dim}D/common/fft_overlap_save/fft_{dim}d.cu'),
               '-o', str(binary_dir / f'flashfft{dim}d')]
    log_path = logs / f'{dim}d.log'
    print(f'Building {args.arch} {dim}D; log: {log_path}', flush=True)
    with log_path.open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'build failed ({result.returncode}); see {log_path}')
    commands.append(command)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {
    'arch': args.arch, 'source_manifest': 'provenance/sources.toml',
    'compiler': subprocess.check_output([nvcc, '--version'], text=True),
    'driver_sha256': sha(root / 'scripts/adapters/flashfft_driver.cu'),
    'patches': {p: sha(patchdir / p) for p in patches},
    'commands': commands,
    'binaries': {f'flashfft{d}d': sha(binary_dir / f'flashfft{d}d') for d in dimensions},
}
(binary_dir / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Built FlashFFTStencil {args.arch}: {binary_dir}')
