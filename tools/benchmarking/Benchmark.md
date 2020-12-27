# Benchmark for 

## Score

Score on 20 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | True       |  161.315  |
|  1 | True                  | False      |   48.5504 |

## Throughput
Average throughput in FPS on 5 samples from ImageNette dataset
|    | triton_optimization   | quantize   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|--------------:|----------------:|-------------:|
|  0 | True                  | False      |             1 |               1 |     40.6558  |
|  1 | False                 | False      |             1 |               1 |     14.7934  |
|  2 | True                  | True       |             1 |               1 |      6.33284 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | quantize   |   score |
|---:|:-----------|--------:|
|  0 | True       |    0.85 |
|  1 | False      |    0.85 |