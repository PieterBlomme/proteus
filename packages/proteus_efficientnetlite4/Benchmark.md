# Benchmark for EfficientNetLite4

## Score

Score on 100 samples from ImageNette dataset
|    | quantize   |   score |
|---:|:-----------|--------:|
|  1 | True       |    0.88 |
|  0 | False      |    0.85 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|--------------:|----------------:|-------------:|
|  5 | True                  | False      |             4 |               1 |    219.105   |
|  4 | True                  | False      |             2 |               1 |    143.642   |
|  3 | True                  | False      |             1 |               2 |     79.0502  |
|  0 | True                  | False      |             1 |               1 |     74.3149  |
|  2 | False                 | False      |             1 |               1 |     18.3322  |
|  1 | True                  | True       |             1 |               1 |      9.24823 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   11.8885 |
|  1 | True                  | True       |  114.111  |
|  2 | False                 | False      |  197.589  |