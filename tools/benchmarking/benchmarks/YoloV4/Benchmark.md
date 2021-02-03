# Benchmark for YoloV4

## Score

Score on 100 samples from CocoValBBox dataset
|    | quantize   |    score |
|---:|:-----------|---------:|
|  0 | False      | 0.342349 |

## Throughput
Average throughput in FPS on 50 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   | dynamic_batching   |   num_workers |   num_instances |   throughput |
|---:|:----------------------|:-----------|:-------------------|--------------:|----------------:|-------------:|
|  5 | True                  | False      | True               |             4 |               1 |     59.0431  |
|  3 | True                  | False      | True               |             2 |               1 |     43.8421  |
|  1 | True                  | False      | False              |             1 |               1 |     27.4329  |
|  0 | True                  | False      | True               |             1 |               1 |     27.2565  |
|  4 | True                  | False      | True               |             1 |               2 |     24.9829  |
|  2 | False                 | False      | False              |             1 |               1 |      9.23008 |

## Latency

Average latency in ms on 10 samples from CocoValBBox dataset
|    | triton_optimization   | quantize   |   latency |
|---:|:----------------------|:-----------|----------:|
|  0 | True                  | False      |   42.3883 |
|  1 | False                 | False      |  125.323  |