"""Native EBISU commands and strict FP64 result parsing."""
import math
import re
import subprocess


def device_info(device, arch):
    output = subprocess.check_output(['nvidia-smi', '-i', str(device),
        '--query-gpu=index,uuid,name,compute_cap,driver_version,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    if output.split(',')[3].strip() != {'sm80': '8.0', 'sm90': '9.0'}[arch]:
        raise ValueError(f'device architecture mismatch: {output}')
    return output


def check_dimensions(c, arch):
    if c['dim'] == 2:
        valid = 256 - 2*c['radius']*c['tstep']
        # Deliberately exercise a short final Y partition as well as two X tiles.
        return [max(valid*3, 2*c['radius']*c['tstep']+32),
                max(valid*2, 2*c['radius']*c['tstep']+32)]
    # Preserve selected XY persistent grid; shorten Z for CPU reference.
    return c['dimensions'][:2] + [280]


def command(binary, c, dimensions, steps, check=False):
    margin = 2*c['radius']*(steps if check else 1)
    if len(dimensions) != c['dim'] or min(dimensions) <= margin:
        raise ValueError('dimensions must leave a nonempty interior')
    if steps < 1 or steps % c['tstep']:
        raise ValueError('steps must be a positive multiple of compiled TSTEP')
    if c['dim'] == 3 and (dimensions[0] % c['tile_x'] or
                          dimensions[1] % (c['ipts']*256//c['tile_x'])):
        raise ValueError('3D X/Y dimensions must be exact tile multiples')
    return [str(binary), *map(str, dimensions), f'--iter={steps}', '--bdim=256',
            '--warmup', '--verbose'] + (['--check', '--blkpsm=1'] if check else [])


def parse_output(output, c, dimensions, steps, check=False):
    if re.search(r'Cuda failure|CUDA Error|illegal|unsupport|inlaunchable|too smal|\bnan\b|\binf\b', output, re.I):
        raise ValueError('EBISU reported an execution error')
    def one(pattern):
        matches = re.findall(pattern, output, re.M)
        if len(matches) != 1:
            raise ValueError(f'missing/duplicate output: {pattern}')
        return matches[0]
    def metric(label):
        v = float(one(r'^\[FORMA\] ' + re.escape(label) + r'\s*:\s*(\S+)'))
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f'invalid {label}')
        return v
    if one(r'^\[FORMA\] Data type: (\w+)') != 'double' or output.count('[FORMA] Finish') != 1:
        raise ValueError('incomplete or non-FP64 output')
    actual = list(map(int, one(r'^\[FORMA\] Domain: ([\d x]+)').split(' x ')))
    expected = dimensions[::-1] if c['dim'] == 2 else dimensions
    if actual != expected:
        raise ValueError('reported domain mismatch')
    iterations = list(map(int, re.findall(r'^\[FORMA\] Iteration: (\d+)', output, re.M)))
    if iterations != [steps, c['tstep']]:
        raise ValueError('reported temporal configuration mismatch')
    grid = list(map(int, one(r'^\[FORMA\] GridDim: (\d+ x \d+ x \d+)').split(' x ')))
    if min(grid) < 1 or metric('BlockDim') != 256:
        raise ValueError('invalid launch configuration')
    if c['dim'] == 3 and metric('Instruction Level Parallelism') != c['ipts']:
        raise ValueError('reported IPTS mismatch')
    ms = metric('Performance (ms)')
    total = metric('Performance (GCells/s)')
    valid = metric('Valid Performance (GCells/s)')
    for rate, count in [(total, math.prod(dimensions)),
                        (valid, math.prod(d-2*c['radius'] for d in dimensions))]:
        derived = count*steps/ms/1e6
        # Native milliseconds and rates have six decimal places.
        tolerance = derived*5.1e-7/max(ms-5.1e-7, 1e-12)+5.1e-7+1e-8
        if abs(rate-derived) > tolerance:
            raise ValueError('native work accounting mismatch')
    result = dict(time_ms=ms, total_gstencils_per_s=total, gstencils_per_s=valid, grid=grid)
    if check:
        errors = {label.lower(): float(one(r'^\[Test\] '+label+r' Error\s*:\s*(\S+)'))
                  for label in ['Max', 'RMS']}
        if any(not math.isfinite(v) or v < 0 for v in errors.values()) or errors['max'] >= 1e-9 or errors['rms'] >= 1e-10:
            raise ValueError(f'CPU comparison failed: {errors}')
        result['validation'] = dict(errors, max_tolerance=1e-9, rms_tolerance=1e-10,
                                    region='interior excluding radius*steps on every face')
    return result
