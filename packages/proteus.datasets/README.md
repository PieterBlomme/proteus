========
Proteus Datasets
========

Datasets to be used for testing and benchmarking

Dataset template looks like this:
```
class Dataset:
    """
    Dataset interface:
    - implement __getitem__ to generate files
    - implement eval(preds) to predict a score
    """

    def __getitem__(self, index):
        None

    def __len__(self):
        return 1

    def eval(self, preds):
        return 1.0
```

Implemented datasets:

- ImageNette
- CocoValBBox
- CocoValMask
- BSDSSuperRes
- MPIIPoseEstimation
