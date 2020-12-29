# Benchmark for YoloV4

## Score

Score on 100 samples from CocoValBBox dataset
|    | quantize   |    score |
|---:|:-----------|---------:|
|  0 | False      | 0.342349 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   | dynamic_batching   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|:-------------------|--------------:|----------------:|-------------:|
|  5 | True                  | False      | True               |             4 |               1 |     25.5334  |
|  3 | True                  | False      | True               |             2 |               1 |     25.0704  |
|  4 | True                  | False      | True               |             1 |               2 |     23.515   |
|  0 | True                  | False      | True               |             1 |               1 |     23.5099  |
|  1 | True                  | False      | False              |             1 |               1 |     22.057   |
|  2 | False                 | False      | False              |             1 |               1 |      9.02872 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   43.9817 |
|  1 | False                 | False      |  126.754  |