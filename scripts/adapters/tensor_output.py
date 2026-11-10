"""Parse native ConvStencil/LoRAStencil output without losing its work convention."""
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


def parse_output(text, c, dimensions, iterations, check=False):
    if re.search(r'CUDA Error|illegal memory|failed:|Invalid argument|out of range', text, re.I):
        raise ValueError('native CUDA/runtime failure')
    info = re.findall(r'INFO: shape = \w+, (.*?)times = (\d+)', text)
    if len(info) != 1 or int(info[0][1]) != iterations:
        raise ValueError('missing or mismatched iteration count')
    actual = list(map(int, re.findall(r'= (\d+)', info[0][0])))
    if actual != dimensions:
        raise ValueError('reported dimensions mismatch')
    if check:
        values = re.findall(r'Validation: failures=(\d+) max_error=([^\s]+)', text)
        if len(values) != 1 or int(values[0][0]) or not math.isfinite(float(values[0][1])):
            raise ValueError('missing or failed CPU reference')
        return dict(validation=dict(failures=0, max_error=float(values[0][1]), scope='one native operator, interior', tolerance=1e-7))
    times = re.findall(r'Time\s*=\s*([0-9.eE+-]+)\[ms\]', text)
    rates = re.findall(r'GStencil/s\s*=\s*([0-9.eE+-]+)', text)
    if len(times) != 1 or len(rates) != 1:
        raise ValueError('missing or duplicate native measurements')
    ms, rate = float(times[0]), float(rates[0])
    if not all(math.isfinite(v) and v > 0 for v in [ms, rate]):
        raise ValueError('nonpositive/nonfinite measurement')
    # Time is truncated to integer milliseconds; throughput uses microsecond timing.
    work = math.prod(dimensions) * iterations * c['native_work_factor']
    implied_ms = work / rate / 1e6
    if not ms - 0.001 <= implied_ms <= ms + 1.001:
        raise ValueError('native work/rate/time mismatch')
    return dict(time_ms=ms, native_gstencils_per_s=rate,
                gstencils_per_s=rate / c['normalization_divisor'])
