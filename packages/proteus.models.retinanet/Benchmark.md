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
|  4 | True                  |             4 |               1 |     20.7541  |
|  2 | True                  |             2 |               1 |     13.5343  |
|  3 | True                  |             1 |               2 |      8.98524 |
|  0 | True                  |             1 |               1 |      8.89281 |
|  1 | False                 |             1 |               1 |      7.48409 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   117.857 |
|  1 | False                 |   175.281 |