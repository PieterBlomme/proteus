# Benchmark for MobileNetV2

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   |   score |
|---:|:----------------------|--------:|
|  0 | True                  |    0.59 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  2 | True                  |             2 |               1 |     113.97   |
|  4 | True                  |             4 |               1 |     112.77   |
|  3 | True                  |             1 |               2 |      90.949  |
|  0 | True                  |             1 |               1 |      90.2597 |
|  1 | False                 |             1 |               1 |      44.0438 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |    9.7841 |
|  1 | False                 |   67.8461 |