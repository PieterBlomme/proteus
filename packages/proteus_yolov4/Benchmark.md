# Benchmark for YoloV4

## Score

Score on 100 samples from CocoValBBox dataset
|    | quantize   |    score |
|---:|:-----------|---------:|
|  0 | False      | 0.342349 |
|  1 | True       | 0.320429 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   | dynamic_batching   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|:-------------------|--------------:|----------------:|-------------:|
|  6 | True                  | False      | True               |             4 |               1 |     61.6628  |
|  4 | True                  | False      | True               |             2 |               1 |     42.42    |
|  5 | True                  | False      | True               |             1 |               2 |     25.6574  |
|  2 | True                  | False      | False              |             1 |               1 |     25.1948  |
|  0 | True                  | False      | True               |             1 |               1 |     23.984   |
|  3 | False                 | False      | False              |             1 |               1 |      9.01205 |
|  1 | True                  | True       | True               |             1 |               1 |      3.82275 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   47.9369 |
|  2 | False                 | False      |  122.635  |
|  1 | True                  | True       |  260.167  |