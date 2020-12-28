# Benchmark for 

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  1 | False                 |   50.5676 |
|  0 | True                  |   68.7578 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  2 | True                  |             2 |               1 |      44.9018 |
|  3 | True                  |             1 |               2 |      32.9203 |
|  0 | True                  |             1 |               1 |      32.1477 |
|  1 | False                 |             1 |               1 |      31.1554 |
|  4 | True                  |             4 |               1 |      28.9737 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   score |
|---:|:----------------------|--------:|
|  0 | True                  |    0.52 |