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
|  2 |             4 |               1 |     17.8274  |
|  1 |             2 |               1 |     16.8939  |
|  0 |             1 |               1 |     14.4894  |
|  3 |             2 |               2 |     10.7678  |
|  4 |             1 |               2 |      9.22789 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   238.706 |