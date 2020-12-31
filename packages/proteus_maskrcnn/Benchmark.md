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
|  3 |             4 |               1 |      3.97824 |
|  1 |             2 |               1 |      3.89403 |
|  0 |             1 |               1 |      3.35248 |
|  2 |             1 |               2 |      2.83631 |

## Latency

Average latency in ms on 10 samples from CocoValMask dataset
|    |   latency |
|---:|----------:|
|  0 |   441.466 |