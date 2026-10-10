### cifar10 (final evaluation epoch per condition; mean +/- 95% CI over seeds)

| condition | epoch | seeds | linear 100% | linear 10% | linear 1% | kNN | eff. rank (centred) |
|---|---|---|---|---|---|---|---|
| jepa_block | 50 | 3 | 60.36 +/- 5.56 | 57.25 +/- 5.94 | 45.88 +/- 6.65 | 56.16 +/- 5.19 | 85.9 |
| jepa_multiblock | 50 | 3 | 58.98 +/- 1.53 | 56.16 +/- 0.75 | 46.36 +/- 5.48 | 56.00 +/- 1.27 | 68.6 |
| jepa_random | 50 | 3 | 55.24 +/- 2.52 | 51.71 +/- 3.87 | 40.48 +/- 2.86 | 51.21 +/- 1.38 | 91.8 |
| pixel_multiblock | 50 | 3 | 59.20 +/- 1.83 | 56.69 +/- 1.65 | 47.27 +/- 1.98 | 52.24 +/- 1.99 | 52.1 |
| raw_pixels | 0 | 3 | 41.64 +/- 0.17 | 37.27 +/- 0.39 | 28.04 +/- 1.93 | 45.21 +/- 0.00 | 782.0 |
| untrained | 0 | 3 | 34.49 +/- 1.08 | 33.53 +/- 1.44 | 27.96 +/- 0.88 | 28.59 +/- 0.43 | 11.6 |

### Paired differences (A - B, percentage points, final epoch; same seeds)

_24 comparisons are listed; with 95% intervals roughly 1 in 20 will exclude 0 by chance alone, so treat an isolated 'yes' cautiously and look for consistent patterns across metrics and seeds._

| dataset | A | B | metric | pairs | mean diff | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|---|---|
| cifar10 | jepa_multiblock | untrained | linear_1.0 | 3 | +24.49 | +/- 1.10 | yes |
| cifar10 | jepa_multiblock | untrained | linear_0.1 | 3 | +22.63 | +/- 0.69 | yes |
| cifar10 | jepa_multiblock | untrained | knn | 3 | +27.41 | +/- 0.97 | yes |
| cifar10 | jepa_block | untrained | linear_1.0 | 3 | +25.87 | +/- 4.49 | yes |
| cifar10 | jepa_block | untrained | linear_0.1 | 3 | +23.72 | +/- 5.66 | yes |
| cifar10 | jepa_block | untrained | knn | 3 | +27.57 | +/- 4.97 | yes |
| cifar10 | jepa_random | untrained | linear_1.0 | 3 | +20.76 | +/- 3.36 | yes |
| cifar10 | jepa_random | untrained | linear_0.1 | 3 | +18.18 | +/- 4.83 | yes |
| cifar10 | jepa_random | untrained | knn | 3 | +22.62 | +/- 1.76 | yes |
| cifar10 | pixel_multiblock | untrained | linear_1.0 | 3 | +24.71 | +/- 1.30 | yes |
| cifar10 | pixel_multiblock | untrained | linear_0.1 | 3 | +23.15 | +/- 2.76 | yes |
| cifar10 | pixel_multiblock | untrained | knn | 3 | +23.65 | +/- 2.41 | yes |
| cifar10 | jepa_multiblock | jepa_block | linear_1.0 | 3 | -1.38 | +/- 4.80 | no |
| cifar10 | jepa_multiblock | jepa_block | linear_0.1 | 3 | -1.09 | +/- 5.77 | no |
| cifar10 | jepa_multiblock | jepa_block | knn | 3 | -0.16 | +/- 4.01 | no |
| cifar10 | jepa_multiblock | jepa_random | linear_1.0 | 3 | +3.73 | +/- 2.91 | yes |
| cifar10 | jepa_multiblock | jepa_random | linear_0.1 | 3 | +4.45 | +/- 4.32 | yes |
| cifar10 | jepa_multiblock | jepa_random | knn | 3 | +4.79 | +/- 2.65 | yes |
| cifar10 | jepa_block | jepa_random | linear_1.0 | 3 | +5.12 | +/- 7.69 | no |
| cifar10 | jepa_block | jepa_random | linear_0.1 | 3 | +5.54 | +/- 9.71 | no |
| cifar10 | jepa_block | jepa_random | knn | 3 | +4.95 | +/- 6.48 | no |
| cifar10 | jepa_multiblock | pixel_multiblock | linear_1.0 | 3 | -0.22 | +/- 2.39 | no |
| cifar10 | jepa_multiblock | pixel_multiblock | linear_0.1 | 3 | -0.52 | +/- 2.19 | no |
| cifar10 | jepa_multiblock | pixel_multiblock | knn | 3 | +3.76 | +/- 2.97 | yes |