# Benchmark for MobileNetV2

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   |   score |
|---:|:----------------------|--------:|
|  0 | True                  |    0.67 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  4 | True                  |             4 |               1 |     212.988  |
|  2 | True                  |             2 |               1 |     143.151  |
|  3 | True                  |             1 |               2 |      88.5616 |
|  0 | True                  |             1 |               1 |      80.648  |
|  1 | False                 |             1 |               1 |      35.4668 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |    9.6533 |
|  1 | False                 |   69.8804 |