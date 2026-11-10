"""AN5D command execution and strict parsing; native timing excludes transfers."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


def parse_output(output, repeats, width, steps, stencil, check=False):
    if re.search(r'CUDA error|assertion|Too short stream|nan|\binf\b', output, re.I):
        raise ValueError('AN5D reported an execution error or non-finite value')
    pattern = r'([0-9.eE+-]+)'
    trials = re.findall(r'^\s*(\d+):\s+' + pattern + r' GFLOPS,\s+' + pattern + r' ms\s*$', output, re.M)
    average = re.findall(r'^Average:\s+' + pattern + r' GFLOPS,\s+' + pattern + r' ms\s*$', output, re.M)
    if len(trials) != repeats or [int(t[0]) for t in trials] != list(range(repeats)) or len(average) != 1:
        raise ValueError('missing or duplicate AN5D timing records')
    dim = int(stencil[4] if stencil.startswith('star') else stencil[3])
    radius = int(stencil[-2])
    fpp = 4 * dim * radius + 1 if stencil.startswith('star') else 2 * (2 * radius + 1)**dim - 1
    times = [float(t[2]) for t in trials]
    rates = [float(t[1]) / fpp for t in trials]
    native_gflops, mean_ms = map(float, average[0])
    if any(not math.isfinite(v) or v <= 0 for v in [*times, *rates, native_gflops, mean_ms]):
        raise ValueError('invalid AN5D timing values')
    def consistent_rate(ms, rate):
        expected = width**dim * steps / ms / 1e6
        # Upstream prints both milliseconds and GFLOPS to four decimal places.
        tolerance = expected * 5.1e-5 / max(ms - 5.1e-5, 1e-9) + 5.1e-5 / fpp + 1e-6
        return abs(rate - expected) <= tolerance

    for ms, rate in zip(times, rates):
        if not consistent_rate(ms, rate):
            raise ValueError('AN5D work accounting disagrees with dimensions/time steps')
    if not math.isclose(mean_ms, statistics.mean(times), rel_tol=5e-4, abs_tol=1e-4):
        raise ValueError('AN5D average time disagrees with individual trials')
    if not consistent_rate(mean_ms, native_gflops / fpp):
        raise ValueError('AN5D native average throughput disagrees with average time')
    row = dict(dim=dim, radius=radius, flops_per_update=fpp, trials_ms=times,
               trials_gstencils_per_s=rates, mean_ms=mean_ms,
               native_average_gflops=native_gflops,
               native_average_gstencils_per_s=native_gflops / fpp,
               gstencils_per_s=statistics.mean(rates), statistic='mean_sample_throughput')
    if check:
        errors = {}
        for label in ['Max', 'RMS']:
            matches = re.findall(label + r' Error: ' + pattern, output)
            if len(matches) != 1:
                raise ValueError('missing CPU comparison')
            errors[label.lower()] = float(matches[0])
        if any(not math.isfinite(v) or v < 0 for v in errors.values()) or errors['max'] > 1e-7 or errors['rms'] > 1e-8:
            raise ValueError(f'CPU comparison failed: {errors}')
        row['validation'] = dict(errors, max_tolerance=1e-7, rms_tolerance=1e-8)
    return row


def execute(root, arch, device, variant, width, steps, repeats, folder, check=False, timeout=1800, build_root=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f'{variant}.log'
    if log.exists():
        raise ValueError(f'refusing to overwrite {log}')
    binary = (build_root or root / 'build') / 'bin/AN5D' / arch / variant
    command = [str(binary), '-s', str(width), '-t', str(steps), '-n', str(repeats)]
    if check:
        command.append('-c')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='8')
    (folder / f'{variant}-command.json').write_text(json.dumps(dict(
        argv=command, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='8'), indent=2) + '\n')
    with log.open('w') as stream:
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                 timeout=timeout, cwd=root)
    if process.returncode:
        raise RuntimeError(f'AN5D exit {process.returncode}: {log}')
    row = parse_output(log.read_text(), repeats, width, steps, variant.split('-')[0], check)
    row.update(implementation='AN5D', arch=arch, device=str(device), stencil=variant.split('-')[0],
               variant=variant, width=width, steps=steps, repeats=repeats,
               input='rand_r_index_seed_0_to_1000', boundary='fixed_halo_two_time_slices',
               command=command, log_path=str(log), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    return row


def device_info(device, arch):
    output = subprocess.check_output(['nvidia-smi', '-i', str(device),
        '--query-gpu=index,uuid,name,compute_cap,driver_version,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    if output.split(',')[3].strip() != {'sm80': '8.0', 'sm90': '9.0'}[arch]:
        raise ValueError(f'device architecture does not match {arch}: {output}')
    return output
