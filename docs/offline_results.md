# Offline decoder accuracy on recorded EEG

Frequencies [9.0, 10.0, 12.0, 15.0] Hz, chance = 25%. Windows start 0.3 s after the flicker begins, every 0.5 s. No per-person training. Made by `python eval/offline_accuracy.py`.

These are recordings from public datasets, not people wearing our headset.

## Tsinghua benchmark, 35 people

| Window | Method | Electrodes | Mean accuracy | Worst person | People >= 90% | ITR (bits/min) |
|---|---|---|---|---|---|---|
| 1 s | CCA | Oz | 74.5% | 40.1% | 9 of 35 | 46.6 |
| 1 s | CCA | PO7 Oz PO8 (Unicorn has these) | 81.7% | 39.6% | 11 of 35 | 61.5 |
| 1 s | CCA | all 8 | 88.3% | 67.7% | 19 of 35 | 77.7 |
| 1 s | FBCCA | Oz | 77.8% | 43.2% | 12 of 35 | 53.1 |
| 1 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 84.7% | 49.5% | 16 of 35 | 68.3 |
| 1 s | FBCCA | all 8 | 90.5% | 65.6% | 23 of 35 | 83.8 |
| 1.5 s | CCA | Oz | 80.7% | 45.2% | 14 of 35 | 39.5 |
| 1.5 s | CCA | PO7 Oz PO8 (Unicorn has these) | 88.4% | 45.2% | 16 of 35 | 51.9 |
| 1.5 s | CCA | all 8 | 93.6% | 72.0% | 26 of 35 | 62.2 |
| 1.5 s | FBCCA | Oz | 84.0% | 47.6% | 16 of 35 | 44.5 |
| 1.5 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 90.5% | 50.0% | 23 of 35 | 55.9 |
| 1.5 s | FBCCA | all 8 | 95.6% | 78.6% | 29 of 35 | 66.7 |
| 2 s | CCA | Oz | 85.7% | 50.7% | 17 of 35 | 35.4 |
| 2 s | CCA | PO7 Oz PO8 (Unicorn has these) | 91.8% | 45.8% | 26 of 35 | 43.9 |
| 2 s | CCA | all 8 | 96.0% | 77.8% | 30 of 35 | 50.9 |
| 2 s | FBCCA | Oz | 88.4% | 51.4% | 19 of 35 | 38.9 |
| 2 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 93.8% | 55.6% | 26 of 35 | 46.9 |
| 2 s | FBCCA | all 8 | 97.4% | 86.8% | 31 of 35 | 53.5 |
| 3 s | CCA | Oz | 90.7% | 56.2% | 22 of 35 | 28.1 |
| 3 s | CCA | PO7 Oz PO8 (Unicorn has these) | 95.1% | 59.4% | 30 of 35 | 32.8 |
| 3 s | CCA | all 8 | 97.9% | 80.2% | 33 of 35 | 36.4 |
| 3 s | FBCCA | Oz | 92.2% | 56.2% | 24 of 35 | 29.6 |
| 3 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 96.7% | 66.7% | 31 of 35 | 34.8 |
| 3 s | FBCCA | all 8 | 99.0% | 89.6% | 34 of 35 | 38.0 |

## g.tec dataset, 2 people

| Window | Method | Electrodes | Mean accuracy | Worst person | People >= 90% | ITR (bits/min) |
|---|---|---|---|---|---|---|
| 1 s | CCA | Oz | 48.3% | 35.8% | 0 of 2 | 10.9 |
| 1 s | CCA | PO7 Oz PO8 (Unicorn has these) | 58.5% | 39.2% | 0 of 2 | 21.7 |
| 1 s | CCA | all 8 | 59.6% | 44.2% | 0 of 2 | 23.2 |
| 1 s | FBCCA | Oz | 52.1% | 36.9% | 0 of 2 | 14.5 |
| 1 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 64.2% | 40.6% | 0 of 2 | 29.5 |
| 1 s | FBCCA | all 8 | 70.4% | 55.8% | 0 of 2 | 39.2 |
| 1.5 s | CCA | Oz | 55.3% | 40.8% | 0 of 2 | 12.0 |
| 1.5 s | CCA | PO7 Oz PO8 (Unicorn has these) | 67.8% | 47.9% | 0 of 2 | 23.3 |
| 1.5 s | CCA | all 8 | 68.5% | 50.2% | 0 of 2 | 24.1 |
| 1.5 s | FBCCA | Oz | 58.9% | 42.7% | 0 of 2 | 14.8 |
| 1.5 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 71.1% | 49.0% | 1 of 2 | 27.0 |
| 1.5 s | FBCCA | all 8 | 80.3% | 66.7% | 1 of 2 | 38.9 |
| 2 s | CCA | Oz | 59.8% | 44.1% | 0 of 2 | 11.7 |
| 2 s | CCA | PO7 Oz PO8 (Unicorn has these) | 73.3% | 55.5% | 1 of 2 | 22.2 |
| 2 s | CCA | all 8 | 73.4% | 55.5% | 1 of 2 | 22.3 |
| 2 s | FBCCA | Oz | 63.0% | 44.5% | 0 of 2 | 13.9 |
| 2 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 76.6% | 56.1% | 1 of 2 | 25.3 |
| 2 s | FBCCA | all 8 | 85.8% | 75.2% | 1 of 2 | 35.6 |
| 3 s | CCA | Oz | 66.7% | 50.6% | 0 of 2 | 11.1 |
| 3 s | CCA | PO7 Oz PO8 (Unicorn has these) | 77.1% | 60.3% | 1 of 2 | 17.2 |
| 3 s | CCA | all 8 | 78.3% | 62.2% | 1 of 2 | 18.1 |
| 3 s | FBCCA | Oz | 70.4% | 55.8% | 0 of 2 | 13.1 |
| 3 s | FBCCA | PO7 Oz PO8 (Unicorn has these) | 83.1% | 67.2% | 1 of 2 | 21.5 |
| 3 s | FBCCA | all 8 | 91.5% | 84.4% | 1 of 2 | 28.9 |

## What we run

`config.yaml`: FBCCA, 3 s window. Shorter windows have a higher ITR on paper but more wrong picks; for a person who cannot easily undo a mistake we chose accuracy.

- Tsinghua benchmark, 35 people: 99.0% with all 8 electrodes; 0.21% of windows exceed the artifact limit (100 uV) and would be skipped.
- g.tec dataset, 2 people: 91.5% with all 8 electrodes; 0.42% of windows exceed the artifact limit (100 uV) and would be skipped.
