# Benchmark for RetinaNet

## Score

Score on 100 samples from CocoValBBox dataset
|    | triton_optimization   |    score |
|---:|:----------------------|---------:|
|  0 | True                  | 0.353822 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  4 | True                  |             4 |               1 |     23.2188  |
|  2 | True                  |             2 |               1 |     13.9999  |
|  3 | True                  |             1 |               2 |      9.29552 |
|  0 | True                  |             1 |               1 |      9.12073 |
|  1 | False                 |             1 |               1 |      7.6443  |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   111.923 |
|  1 | False                 |   179.619 |