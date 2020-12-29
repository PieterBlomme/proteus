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
|  1 |             2 |               1 |      6.90026 |
|  0 |             1 |               1 |      6.67888 |
|  2 |             2 |               2 |      4.30257 |
|  3 |             1 |               2 |      4.16758 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   486.266 |