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
|  4 | True                  | False      |             4 |               1 |     215.22   |
|  3 | True                  | False      |             2 |               1 |     133.814  |
|  2 | True                  | False      |             1 |               2 |      81.008  |
|  0 | True                  | False      |             1 |               1 |      80.4053 |
|  1 | False                 | False      |             1 |               1 |      17.0931 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   11.5929 |
|  1 | False                 | False      |  192.211  |