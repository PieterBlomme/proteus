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
|  2 | True                  |             2 |               1 |     10.3175  |
|  4 | True                  |             4 |               1 |     10.2316  |
|  3 | True                  |             1 |               2 |      9.96263 |
|  0 | True                  |             1 |               1 |      9.86101 |
|  1 | False                 |             1 |               1 |      8.06962 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   106.647 |
|  1 | False                 |   159.027 |