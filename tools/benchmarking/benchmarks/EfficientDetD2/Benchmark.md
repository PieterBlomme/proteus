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
|  2 |             4 |               1 |      7.58956 |
|  1 |             2 |               1 |      7.29533 |
|  0 |             1 |               1 |      6.95794 |
|  3 |             2 |               2 |      4.91367 |
|  4 |             1 |               2 |      4.42691 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   472.207 |