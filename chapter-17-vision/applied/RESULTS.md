# Measured results

128 training, 32 validation and 64 test images, generated with seeds 10/11/12.
Three model seeds: 42/43/44, 40 epochs, Adam learning rate 0.003, batches of 16.
The threshold baseline gives background/foreground Dice 0.968063/0.698623.
The U-Net means are 0.991828/0.938158, with population seed standard
deviations 0.002135/0.015077. Selected thresholds: 0.6, 0.3, 0.5.
Dice pools pixels within each class across the test images. These repeats
measure initialisation variation on one fixed synthetic test set.
