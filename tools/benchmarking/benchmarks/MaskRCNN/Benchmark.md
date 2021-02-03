# Benchmark for MaskRCNN

## Score

Score on 100 samples from CocoValMask dataset
|    |    score |
|---:|---------:|
|  0 | 0.327063 |

## Throughput
Average throughput in FPS on 50 samples from CocoValMask dataset
|    |   num_workers |   num_instances |   throughput |
|---:|--------------:|----------------:|-------------:|
|  3 |             4 |               1 |      4.02101 |
|  1 |             2 |               1 |      3.9751  |
|  0 |             1 |               1 |      3.56431 |
|  2 |             1 |               2 |      2.94482 |

## Latency

Average latency in ms on 10 samples from CocoValMask dataset
|    |   latency |
|---:|----------:|
|  0 |    452.66 |