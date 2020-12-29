# Benchmark for EfficientNetLite4

## Score

Score on 100 samples from ImageNette dataset
|    | quantize   |   score |
|---:|:-----------|--------:|
|  0 | False      |    0.85 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|--------------:|----------------:|-------------:|
|  3 | True                  | False      |             2 |               1 |      92.3679 |
|  4 | True                  | False      |             4 |               1 |      90.1308 |
|  0 | True                  | False      |             1 |               1 |      76.9962 |
|  2 | True                  | False      |             1 |               2 |      76.6523 |
|  1 | False                 | False      |             1 |               1 |      16.3393 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   13.1624 |
|  1 | False                 | False      |  196.281  |