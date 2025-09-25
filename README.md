# Paths
Simulate the extrema of a geometric Brownian motion

```
Simulation parameters:
  Paths:          1000000
  Steps per path: 252
  Log-return mean: 0.000000
  Log-return std:  0.010000
  Initial price:   100.0000
  RNG seed:        None
  Batch size:      100000

Simulated 1000000 paths with 252 steps per path.

Summary statistics (Pandas dataframe):
             mean   median     std      min       q1       q3      max
Maximum  113.3867 110.6745 11.3968 100.0000 104.5747 119.3459 221.3492
Minimum   89.0076  90.3753  8.1339  43.8623  83.7922  95.6354 100.0000
Terminal 101.2707 100.0191 16.1679  46.5662  89.8431 111.2831 219.0190

Probability initial price is path maximum: 3.5688%
Probability initial price is path minimum: 3.5363%
Zero-drift probability that initial price is path minimum: 3.5541%

Probability terminal price exceeds thresholds:
  > 100.00: 50.0509%
  > 104.00: 40.2749%
```
