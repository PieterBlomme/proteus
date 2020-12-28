# Benchmark for 

## Score

Score on 100 samples from CocoValBBox dataset
|    | triton_optimization   |   latency |
|---:|:----------------------|----------:|
|  0 | True                  |   197.175 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    |   num_workers |   num_instances |   throughput |
|---:|--------------:|----------------:|-------------:|
|  3 |             1 |               2 |      5.26617 |
|  2 |             2 |               2 |      5.16472 |
|  1 |             2 |               1 |      4.84922 |
|  0 |             1 |               1 |      4.65533 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    |    score |
|---:|---------:|
|  0 | 0.316536 |