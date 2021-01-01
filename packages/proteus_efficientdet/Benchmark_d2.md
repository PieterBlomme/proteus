# Benchmark for EfficientDetD2

## Score

Score on 100 samples from CocoValBBox dataset
|    |   score |
|---:|--------:|
|  0 |  0.2116 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    |   num_workers |   num_instances |   throughput |
|---:|--------------:|----------------:|-------------:|
|  1 |             2 |               1 |      7.55327 |
|  2 |             4 |               1 |      7.26256 |
|  0 |             1 |               1 |      6.80081 |
|  3 |             2 |               2 |      4.96837 |
|  4 |             1 |               2 |      4.26833 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   475.957 |