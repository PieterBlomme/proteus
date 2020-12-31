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
|  5 | True                  | False      |             4 |               1 |    104.541   |
|  4 | True                  | False      |             2 |               1 |    100.454   |
|  0 | True                  | False      |             1 |               1 |     83.2911  |
|  3 | True                  | False      |             1 |               2 |     83.2439  |
|  2 | False                 | False      |             1 |               1 |     16.073   |
|  1 | True                  | True       |             1 |               1 |      7.86415 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   10.6867 |
|  1 | True                  | True       |  129.597  |
|  2 | False                 | False      |  188.911  |