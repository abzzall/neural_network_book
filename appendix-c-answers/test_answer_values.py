"""The numbers quoted in appendix C are the numbers the script prints."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def test_answer_values(tmp_path):
    lines = subprocess.run([sys.executable, str(HERE / 'answer_values.py')], cwd=tmp_path,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert lines == [
        '2.1 parameters of a 784-128 layer: 100480',
        '3.1 P(fraud | flagged): 0.0454',
        '3.7 cross-entropy, true class at 0.3: 1.2040',
        '4.10 class weights for 0.5 % positives: 0.5025 and 100.0',
        '5.2 perceptron on OR: mistakes per epoch: [3, 1, 2, 2, 1, 0]',
        '5.2 perceptron on OR: w and b: [2.0, 2.0] -1.0',
        '6.1 parameters of 20-64-32-5: 3589',
        '7.2 updates per epoch: 500',
        '12.1 sizes after conv and pool: 32 then 16',
        '12.2 convolution and dense parameters: 2432 and 393248',
        '15.3 "brains" after six merges: b rain s </w>',
        '16.8 layers of kernel 3 to see a year: 8 (receptive field 511)',
        '17.2 intersection and union: 3.6 and 4.4',
        '18.3 third GCN output: 2.483163',
        '19.1 returns: 1.75 and 2.71',
        '21.2 top-p set for p = 0.9: 0.4444 0.2778 0.1667 0.1111',
        '21.4 GPT-2 cache for 1024 tokens (MB): 75.5',
        '22.3 LoRA rank 4 on 768 x 768: 6144 weights, 1.04%',
        '24.3 quantised codes: [-128, -64, 127], scale 0.007843, zero point -64, '
        'largest error 0.00196',
        '24.6 PSI: 0.4159',
        '25.5 threshold for costs 2 and 3: 0.4',
        '25.6 cost of the policy: 0.4000',
    ]
