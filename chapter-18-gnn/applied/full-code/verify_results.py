"""Check measured records; --rerun additionally repeats the full experiment."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_exercise as exercise
records = json.loads(Path(__file__).with_name('reference_results.json').read_text())
assert len(records) == 15 and exercise.choose_depth(records) == 4
assert all(0 <= r['test_accuracy'] <= 1 for r in records)
if '--rerun' in sys.argv:
    for record in records:
        actual = exercise.run(record['seed'], record['depth'], loss_fn=exercise.masked_loss)
        assert abs(actual['test_accuracy'] - record['test_accuracy']) < .03
print('selected depth:', exercise.choose_depth(records))
print('PASS')
