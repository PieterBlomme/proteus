# Benchmark for 

## Score

Score on 100 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  2 | False                 | True       |   872.372 |
|  1 | True                  | False      |   920.479 |
|  3 | False                 | False      |   957.704 |
|  0 | True                  | True       |  1111.72  |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   | dynamic_batching   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|:-------------------|--------------:|----------------:|-------------:|
|  5 | True                  | False      | True               |             1 |               2 |      1.38616 |
|  4 | True                  | False      | True               |             2 |               1 |      1.17546 |
|  1 | True                  | False      | True               |             1 |               1 |      1.16102 |
|  2 | True                  | True       | False              |             1 |               1 |      1.14634 |
|  3 | False                 | False      | False              |             1 |               1 |      1.13746 |
|  0 | True                  | True       | True               |             1 |               1 |      1.1357  |
|  6 | True                  | False      | True               |             4 |               1 |      1.13112 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | quantize   |    score |
|---:|:-----------|---------:|
|  0 | False      | 0.353822 |
|  1 | True       | 0.353822 |