# Benchmark for 

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  3 | False                 | False      |   48.617  |
|  1 | True                  | False      |   52.4732 |
|  0 | True                  | True       |  136.358  |
|  2 | False                 | True       |  167.926  |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|--------------:|----------------:|-------------:|
|  4 | True                  | False      |             2 |               1 |     37.379   |
|  5 | True                  | False      |             4 |               1 |     36.4435  |
|  1 | True                  | False      |             1 |               1 |     30.7522  |
|  3 | True                  | False      |             1 |               2 |     27.6539  |
|  2 | False                 | False      |             1 |               1 |     24.7281  |
|  0 | True                  | True       |             1 |               1 |      6.70238 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | quantize   |   score |
|---:|:-----------|--------:|
|  0 | True       |    0.82 |
|  1 | False      |    0.82 |