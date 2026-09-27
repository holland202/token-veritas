condition R  n=40  full-context accuracy 0.975
| policy | budget | tokens (mean) | coverage | accuracy |
|---|---|---|---|---|
| random | 0.1 | 30.6 | 0.287 | 0.050 |
| topk_cos | 0.1 | 30.7 | 0.900 | 0.800 |
| dpp_cos | 0.1 | 28.9 | 0.762 | 0.575 |
| loo | 0.1 | 29.6 | 0.500 | 0.150 |
| dpp_loo | 0.1 | 29.8 | 0.487 | 0.150 |
| dpp_uniform | 0.1 | 26.8 | 0.312 | 0.025 |
| random | 0.2 | 67.2 | 0.625 | 0.325 |
| topk_cos | 0.2 | 61.7 | 1.000 | 0.825 |
| dpp_cos | 0.2 | 69.8 | 0.963 | 0.775 |
| loo | 0.2 | 68.0 | 0.762 | 0.450 |
| dpp_loo | 0.2 | 68.9 | 0.812 | 0.450 |
| dpp_uniform | 0.2 | 68.8 | 0.800 | 0.475 |
| random | 0.3 | 103.5 | 0.863 | 0.525 |
| topk_cos | 0.3 | 105.7 | 1.000 | 0.875 |
| dpp_cos | 0.3 | 103.0 | 1.000 | 0.925 |
| loo | 0.3 | 103.2 | 0.875 | 0.650 |
| dpp_loo | 0.3 | 102.0 | 0.963 | 0.700 |
| dpp_uniform | 0.3 | 97.2 | 0.975 | 0.675 |

condition N  n=40  full-context accuracy 0.425
| policy | budget | tokens (mean) | coverage | accuracy |
|---|---|---|---|---|
| random | 0.1 | 22.6 | 0.050 | 0.000 |
| topk_cos | 0.1 | 25.0 | 0.325 | 0.000 |
| dpp_cos | 0.1 | 24.5 | 0.362 | 0.000 |
| loo | 0.1 | 23.9 | 0.325 | 0.000 |
| dpp_loo | 0.1 | 23.1 | 0.325 | 0.000 |
| dpp_uniform | 0.1 | 24.0 | 0.000 | 0.000 |
| random | 0.2 | 48.0 | 0.087 | 0.000 |
| topk_cos | 0.2 | 47.0 | 0.725 | 0.475 |
| dpp_cos | 0.2 | 49.5 | 0.775 | 0.425 |
| loo | 0.2 | 48.5 | 0.750 | 0.425 |
| dpp_loo | 0.2 | 47.9 | 0.688 | 0.375 |
| dpp_uniform | 0.2 | 48.5 | 0.000 | 0.000 |
| random | 0.3 | 74.2 | 0.338 | 0.075 |
| topk_cos | 0.3 | 75.2 | 0.925 | 0.600 |
| dpp_cos | 0.3 | 74.0 | 0.938 | 0.675 |
| loo | 0.3 | 75.0 | 0.912 | 0.650 |
| dpp_loo | 0.3 | 74.3 | 0.750 | 0.425 |
| dpp_uniform | 0.3 | 75.7 | 0.000 | 0.000 |

## Registered predictions (budget 0.2)
P1 coverage dpp_cos - topk_cos (R) = -0.037  95% CI [-0.087, 0.000]  threshold >= 0.20 -> FAIL
P2 accuracy dpp_cos - topk_cos (R) = -0.050  95% CI [-0.225, 0.125]  wins 5 losses 7 p=0.7744  threshold >= 0.15 & p<0.05 -> FAIL
P3 [control] coverage dpp_cos - topk_cos (N) = 0.050  threshold |d| <= 0.10 -> PASS
P4 [control] coverage dpp_uniform 0.800 vs random 0.625 (R)  threshold <= random+0.10 -> FAIL
P5 LOO coverage N 0.750 -> R 0.762, drop -0.012  threshold >= 0.20 -> FAIL
P6 accuracy dpp_cos 0.775 vs loo 0.450 (R)  threshold dpp >= loo-0.05 -> PASS
seconds: R 4559.8  N 3819.8
