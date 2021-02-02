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
|  1 |             2 |               1 |     17.2073  |
|  2 |             4 |               1 |     17.1759  |
|  0 |             1 |               1 |     13.9253  |
|  3 |             2 |               2 |     11.1501  |
|  4 |             1 |               2 |      9.38441 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   215.727 |