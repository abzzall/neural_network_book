# Chapter 12 applied exercise — results

Fill in the second column with the values your code prints, to four
significant figures. Then run, from the repository root:

```bash
.venv/bin/python chapter-12-cnn/applied/full-code/verify_results.py chapter-12-cnn/applied/RESULTS.md
```

| Quantity | Value |
|---|---|
| `mlp_parameters` | |
| `cnn_parameters` | |
| `mlp_test_accuracy` | |
| `mlp_shifted_accuracy` | |
| `cnn_test_accuracy` | |
| `cnn_shifted_accuracy` | |
| `cnn_augmented_test_accuracy` | |
| `cnn_augmented_shifted_accuracy` | |

## What to explain

1. Why does shifting every test image by one pixel hurt the MLP so much more
   than the CNN?
2. The CNN has more parameters than the MLP here. Why would the comparison be
   the other way round for a 224 x 224 colour photograph?
3. Augmentation makes the CNN almost immune to the shift but lowers its accuracy
   on the original images slightly. When is that trade worth making?
