# Chapter 24 applied exercise — results

| Quantity | Value |
|---|---|
| `unchanged_intensity_psi` | |
| `unchanged_confidence_alarm` | |
| `shifted_accuracy` | |
| `shifted_intensity_alarm` | |
| `shifted_confidence_alarm` | |
| `fainter_accuracy` | |
| `fainter_intensity_psi` | |
| `fainter_intensity_alarm` | |
| `swapped_accuracy` | |
| `swapped_confidence_alarm` | |

## What to explain

1. Shifted images reduce accuracy sharply, but the intensity monitor stays
   silent. Why, and which monitor catches it?
2. Fainter scans raise a large alarm, yet accuracy is unchanged. Is the alarm
   wrong? What would you do when it fires?
3. Why can no input or output monitor detect the swapped labels? What does a
   deployed system need in order to detect such concept drift?
4. Why should alarm thresholds be tested on scenarios like these before deployment?
