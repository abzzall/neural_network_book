"""Check measured records; --rerun additionally repeats the full experiment."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_exercise as exercise
records = json.loads(Path(__file__).with_name('reference_results.json').read_text())
assert [r['seed'] for r in records] == [42, 43, 44]
assert all(r['dice'][1] > r['baseline_dice'][1] for r in records)
if '--rerun' in sys.argv:
    for record in records:
        actual = exercise.run(record['seed'], loss_fn=exercise.segmentation_loss,
                              threshold_fn=exercise.choose_threshold, score_fn=exercise.class_dice)
        assert np.allclose(actual['dice'], record['dice'], atol=.025, rtol=0)
print('foreground Dice mean:', round(np.mean([r['dice'][1] for r in records]), 6))
print('PASS')
