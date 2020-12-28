# Benchmark for 

## Score

Score on 100 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  1 | False                 |   692.555 |
|  0 | True                  |   887.146 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  2 | True                  |             2 |               1 |      1.56023 |
|  3 | True                  |             1 |               2 |      1.55379 |
|  1 | False                 |             1 |               1 |      1.47409 |
|  4 | True                  |             4 |               1 |      1.37109 |
|  0 | True                  |             1 |               1 |      1.31099 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |    score |
|---:|:----------------------|---------:|
|  0 | True                  | 0.353822 |