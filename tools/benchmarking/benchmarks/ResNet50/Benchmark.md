# Benchmark for Resnet50V2

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   |   score |
|---:|:----------------------|--------:|
|  0 | True                  |    0.74 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  4 | True                  |             4 |               1 |     192.942  |
|  2 | True                  |             2 |               1 |     123.461  |
|  3 | True                  |             1 |               2 |      76.5906 |
|  0 | True                  |             1 |               1 |      76.5442 |
|  1 | False                 |             1 |               1 |      64.2332 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   11.8449 |
|  1 | False                 |   18.6604 |