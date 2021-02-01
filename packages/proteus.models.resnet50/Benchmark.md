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
|  4 | True                  |             4 |               1 |     215.474  |
|  2 | True                  |             2 |               1 |     125.248  |
|  0 | True                  |             1 |               1 |      78.5783 |
|  3 | True                  |             1 |               2 |      76.5131 |
|  1 | False                 |             1 |               1 |      65.8347 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   11.7795 |
|  1 | False                 |   18.7786 |