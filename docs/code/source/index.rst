:github_url: https://github.com/szymonmaszke/torchlayers

***********
torchlayers
***********

**torchlayers** is a PyTorch based library
providing **automatic shape and dimensionality inference of `torch.nn` layers** and  additional
building blocks featured in current SOTA architectures (e.g. `Efficient-Net <https://arxiv.org/abs/1905.11946>`__).

Above requires no user intervention (except single call to `torchlayers.build`)
similarly to the one seen in `Keras <https://www.tensorflow.org/guide/keras>`__.

**Main functionalities:**

* **Shape inference** for most of `torch.nn` module (**convolutional, recurrent, transformer, attention and linear layers**)
* **Dimensionality inference** (e.g. `torchlayers.Conv` working as `torch.nn.Conv1d/2d/3d` based on `input shape`)
* **Shape inference of custom modules** (see `GitHub README <https://github.com/szymonmaszke/torchlayers/blob/86bb6f6fca0a9490f4d7fb4602cf246862150ab9/README.md#examples>`__)
* **Additional** `Keras-like <https://www.tensorflow.org/guide/keras>`__ **layers** (e.g. `torchlayers.Reshape` or `torchlayers.StandardNormalNoise`)
* **Additional SOTA layers** mostly from ImageNet competitions (e.g. `PolyNet <https://arxiv.org/abs/1608.06993>`__, `Squeeze-And-Excitation <https://arxiv.org/abs/1709.01507>`__, `StochasticDepth <www.arxiv.org/abs/1512.03385>`__)
* **Useful defaults** (`"same"` padding and default `kernel_size=3` for `Conv`, dropout rates etc.)
* **Zero overhead and** `torchscript <https://pytorch.org/docs/stable/jit.html>`__ **support**

Modules
#######

If you import classes from modules listed belows using `torchlayers` you will get
shape inferrable version, e.g. `torchlayers.Conv` instead of `torchlayers.convolution.Conv`.

If you wish to use those without shape inferrence capabilities use fully qualified module name,
e.g. `torchlayers.convolution.SqueezeExcitation`.

.. toctree::
   :glob:
   :maxdepth: 1

   packages/*

.. toctree::
   :hidden:

   related


Installation
############

Following installation methods are available:

`pip: <https://pypi.org/project/torchlayers/>`__
================================================

To install latest release:

.. code-block:: shell

  pip install --user torchlayers

To install `nightly` version:

.. code-block:: shell

  pip install --user torchlayers-nightly


`Docker: <https://cloud.docker.com/repository/docker/szymonmaszke/torchlayers>`__
=================================================================================

Various `torchlayers` images are available both CPU and GPU-enabled.
You can find them at Docker Cloud at `szymonmaszke/torchlayers`

CPU
---

CPU image is based on `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ and
official release can be pulled with:

.. code-block:: shell

  docker pull szymonmaszke/torchlayers:18.04

For `nightly` release:

.. code-block:: shell

  docker pull szymonmaszke/torchlayers:nightly_18.04

This image is significantly lighter due to lack of GPU support.

GPU
---

All images are based on `nvidia/cuda <https://hub.docker.com/r/nvidia/cuda/>`__ Docker image.
Each has corresponding CUDA version tag ( `10.1`, `10` and `9.2`) CUDNN7 support
and base image ( `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ ).

Following images are available:

- `10.1-cudnn7-runtime-ubuntu18.04`
- `10.1-runtime-ubuntu18.04`
- `10.0-cudnn7-runtime-ubuntu18.04`
- `10.0-runtime-ubuntu18.04`
- `9.2-cudnn7-runtime-ubuntu18.04`
- `9.2-runtime-ubuntu18.04`

Example pull:

.. code-block:: shell

  docker pull szymonmaszke/torchlayers:10.1-cudnn7-runtime-ubuntu18.04

You can use `nightly` builds as well, just prefix the tag with `nightly_`, for example
like this:

.. code-block:: shell

  docker pull szymonmaszke/torchlayers:nightly_10.1-cudnn7-runtime-ubuntu18.04
