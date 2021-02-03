========
Benchmarking
========

Small CLI to help with model benchmarking.
Usage: 'proteus.benchmark <file>'
Where <file> is the path to a json file containing the benchmark suite to be run.  Output will be written to a markdown file.
See eg. configs/example.json for an example config file.

Benchmarks on my own workstation available under benchmarks/

Note: the benchmarking needs proteus.datasets.  Install it from source
```
pip install -e ../../packages/proteus.datasets
```