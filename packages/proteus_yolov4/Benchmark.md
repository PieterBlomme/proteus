# Benchmark for YoloV4

## Score

Score on 100 samples from CocoValBBox dataset
|    | quantize   |    score |
|---:|:-----------|---------:|
|  0 | False      | 0.342349 |
|  1 | True       | 0.321724 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   | dynamic_batching   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|:-------------------|--------------:|----------------:|-------------:|
|  6 | True                  | False      | True               |             4 |               1 |     31.1095  |
|  4 | True                  | False      | True               |             2 |               1 |     30.8685  |
|  5 | True                  | False      | True               |             1 |               2 |     28.6526  |
|  2 | True                  | False      | False              |             1 |               1 |     27.5049  |
|  0 | True                  | False      | True               |             1 |               1 |     26.7263  |
|  3 | False                 | False      | False              |             1 |               1 |      9.53335 |
|  1 | True                  | True       | True               |             1 |               1 |      3.88872 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   34.5623 |
|  2 | False                 | False      |  133.429  |
|  1 | True                  | True       |  296.848  |