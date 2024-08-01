# Simple Waymo Open Dataset Reader

This is a simple file reader for the [Waymo Open Dataset](https://waymo.com/open/) which does not depend on TensorFlow and Bazel. The main goal is to be able to quickly integrate Waymo’s dataset with other deep learning frameworks without having to pull tons of dependencies. It does not aim to replace the [whole framework](https://github.com/waymo-research/waymo-open-dataset), especially the evaluation metrics that they provide.

## Installation

Use the provided `setup.py`:

```
python setup.py install
```

## Usage

Please refer to the examples in `examples/` for how to use the file reader. Refer to [https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb) for more details on Waymo’s dataset.

## License

This code is released under the Apache License, version 2.0. This projects incorporate some parts of the [Waymo Open Dataset code](https://github.com/waymo-research/waymo-open-dataset/blob/master/README.md) (the files `simple_waymo_open_dataset_reader/*.proto`) and is licensed to you under their original license terms. See `LICENSE` file for details.

