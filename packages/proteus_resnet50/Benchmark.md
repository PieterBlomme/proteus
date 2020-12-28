# Benchmark for 

## Score

Score on 100 samples from ImageNette dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   19.6216 |
|  1 | False                 |   41.9097 |

## Throughput
Average throughput in FPS on 50 samples from ImageNette dataset
|    | triton_optimization   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|--------------:|----------------:|-------------:|
|  2 | True                  |             2 |               1 |      39.4939 |
|  0 | True                  |             1 |               1 |      36.7478 |
|  4 | True                  |             4 |               1 |      32.9954 |
|  1 | False                 |             1 |               1 |      32.379  |
|  3 | True                  |             1 |               2 |      31.6262 |

## Latency

Average latency in ms on 10 samples from ImageNette dataset
|    | triton_optimization   |   score |
|---:|:----------------------|--------:|
|  0 | True                  |    0.52 |