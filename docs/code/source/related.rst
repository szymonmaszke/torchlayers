****************
Related projects
****************

Below you can find other projects started by the same author and based on `PyTorch <https://pytorch.org/>`__ as well:

`torchdata: <https://github.com/szymonmaszke/torchdata>`__
==========================================================

**torchdata** extends `torch.utils.data.Dataset` and equips it with
functionalities known from `tensorflow.data <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`__
like `map` or `cache`.
All of that with minimal interference (single call to `super().__init__()`) with original
PyTorch's datasets.

Some functionalities:

* `torch.utils.data.IterableDataset` and `torch.utils.data.Dataset` support
* `map` or `apply` arbitrary functions to dataset
* `memory` or `disk` allows you to cache data (even partially, say `20%`)
* Concrete classes designed for file reading or database support

You can read documentation over at https://szymonmaszke.github.io/torchdata.

`torchlambda: <https://github.com/szymonmaszke/torchlambda>`__
==============================================================

**torchlambda** is a tool to deploy PyTorch models on Amazon's AWS Lambda using AWS SDK for C++ and custom C++ runtime.

* Using statically compiled dependencies whole package is shrunk to only 30MB.
* Due to small size of compiled source code users can pass their models as AWS Lambda layers. Services like Amazon S3 are no longer necessary to load your model.
* torchlambda has it's PyTorch & AWS dependencies always up to date because of continuous deployment run at 03:00 a.m. every day.

You can read project's wiki over at https://github.com/szymonmaszke/torchlambda/wiki

`torchfunc: <https://github.com/szymonmaszke/torchfunc>`__
==========================================================

**torchfunc** is PyTorch oriented library with a goal to help you with:

* Improving and analysing performance of your neural network
* Plotting and visualizing modules
* Record neuron activity and tailor it to your specific task or target
* Get information about your host operating system, CUDA devices and others
* Day-to-day neural network related duties (model size, seeding, performance measurements etc.)

It **is not** directly related with model creation but should be considered more of an environment
around this process.

You can read documentation over at https://szymonmaszke.github.io/torchfunc.
