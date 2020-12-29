# Benchmark for EfficientDetD0

## Score

Score on 100 samples from CocoValBBox dataset
|    |    score |
|---:|---------:|
|  0 | 0.101062 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    |   num_workers |   num_instances |   throughput |
|---:|--------------:|----------------:|-------------:|
|  1 |             2 |               1 |     14.2779  |
|  0 |             1 |               1 |     14.2722  |
|  2 |             2 |               2 |      9.74878 |
|  3 |             1 |               2 |      9.27896 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   216.311 |