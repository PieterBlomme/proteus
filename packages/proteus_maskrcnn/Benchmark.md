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
|  1 |             2 |               1 |      3.70396 |
|  3 |             4 |               1 |      3.65482 |
|  0 |             1 |               1 |      3.62533 |
|  2 |             1 |               2 |      2.93764 |

## Latency

Average latency in ms on 10 samples from CocoValMask dataset
|    |   latency |
|---:|----------:|
|  0 |   446.522 |