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
|  4 | True                  |             4 |               1 |     242.076  |
|  2 | True                  |             2 |               1 |     151.542  |
|  3 | True                  |             1 |               2 |      89.2317 |
|  0 | True                  |             1 |               1 |      85.9043 |
|  1 | False                 |             1 |               1 |      40.251  |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |    9.6146 |
|  1 | False                 |   66.7322 |