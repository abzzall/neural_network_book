"""Check local numerical acceptance bounds, optionally against recorded JSON."""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_training


def check(values):
    required = {'cpu_initial_mse', 'cpu_final_mse'}
    if not required <= values.keys() or not all(math.isfinite(v) for v in values.values()):
        return False
    if not (5 < values['cpu_initial_mse'] < 20 and 0 <= values['cpu_final_mse'] < 1e-6):
        return False
    gpu = {'cuda_fp32_final_mse', 'cuda_fp32_gradient_error',
           'cuda_amp_final_mse', 'cuda_amp_gradient_error'}
    if gpu & values.keys():
        if not gpu <= values.keys():
            return False
        return (0 <= values['cuda_fp32_final_mse'] < 1e-5
                and 0 <= values['cuda_amp_final_mse'] < 1e-5
                and 0 <= values['cuda_fp32_gradient_error'] < 1e-5
                and 0 <= values['cuda_amp_gradient_error'] < 0.01)
    return True


if __name__ == '__main__':
    local = cuda_training.run()
    print(json.dumps(local, indent=2))
    ok = check(local)
    if len(sys.argv) > 1:
        recorded = json.loads(Path(sys.argv[1]).read_text())
        ok = ok and recorded.keys() == local.keys() and check(recorded)
        ok = ok and all(math.isclose(recorded[k], local[k], rel_tol=0.05, abs_tol=1e-5)
                        for k in local)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
