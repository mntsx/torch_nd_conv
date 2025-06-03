# PyTorch ND Convolution Repository

[![LICENSE: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-%23C2A000.svg)](https://github.com/python/cpython)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-202020.svg?style=flat)](https://github.com/psf/black)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mntsx/torch_nd_conv)
[![GitHub stars](https://img.shields.io/github/stars/mntsx/torch_nd_conv.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/mntsx/torch_nd_conv/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/mntsx/torch_nd_conv.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/mntsx/torch_nd_conv/network/)


## Introduction

**torch_nd_conv** contains a fully Python written implementation of n-dimensional convolution operation using **PyTorch** as its only dependency.

### Key Features

- **Arbitrary Dimension Convolution**: implements convolution (`Conv`) operation for input spaces any number of dimensions.
- **Auxiliary Classes**: provides `Fold` and `Unfold` classes - instrumental for the manipulation of N-dimensional feature spaces.
- **PyTorch Integration**: all classes inherit from `torch.nn.Module` - ensuring seamless integration with existing PyTorch workflows.
- **Python 3.12+**: developed using Python version 3.12 - taking advantage of some its new features and optimizations.

## Usage Notes

While the modules provided in this repository are built upon PyTorch's architecture and conventions, there are some key differences in their usage compared to PyTorch's native modules. Below are important considerations and guidelines for effectively utilizing the `Fold`, `Unfold`, and `Conv` classes.

### Differences with respect to PyTorch's (`Conv`, `Fold`, `Unfold`) Native Modules

#### 1. **Additional Initialization Parameters for N-Dimensional Support**

Both `Fold` and `Unfold` accept optional arguments that precompute shape-related data during construction, reducing work in each `forward` call:

* **`input_size`** (`Fold`, `Unfold`)

  * **Purpose**: If you know the non-batched spatial dimensions (e.g., `(C, D, H, W)` for a batch of 3D volumes), passing `input_size` lets the module build its index masks and compute dilations/strides once in `__init__`. This avoids repeating those calculations at runtime.
  * **Usage**: Provide a tuple (or integer) matching the input’s convolutional dimensions. The module checks at `forward` that the incoming tensor’s shape aligns with this size.

* **`output_size`** (`Fold` only)

  * **Purpose**: Validates at construction that your kernel parameters (kernel size, stride, dilation, padding) will fold an unfolded tensor back to the correct shape, catching mismatches early.
  * **Usage**: Supply the expected output-volume shape (excluding batch), e.g. `(C, D′, H′, W′)`. If the parameters wouldn’t produce that size, `Fold` raises an error in `__init__`.

* **`kernel_position`** (`Fold` only)

  * **Purpose**: Lets you specify whether the kernel dimensions come before or after the convolutional-output axes in the unfolded input:

    * **`"last"`** (default): Input to `forward` is `(..., D_out, H_out, W_out, K_d, K_h, K_w)`.
    * **`"first"`**: Input to `forward` is `(..., K_d, K_h, K_w, D_out, H_out, W_out)`.
  * **Usage**: Match it to how your data pipeline organizes those axes so you don’t need extra `permute` calls.

#### 2. **Differences in Input/Output Formats Compared to PyTorch**

Unlike PyTorch’s 2D-only modules, which flatten all kernel dims into one channel axis, `torch_nd_conv` keeps kernel dimensions separate for clarity in N-D operations:

* **`Unfold`**

  * **PyTorch 2D**: Takes `(N, C, H, W)` → `(N, C×K_h×K_w, L)`, collapsing the `K_h×K_w` patch into a single channel and listing `L` sliding-window positions.
  * **torch\_nd\_conv N-D**: From `(N, C, D, H, W)` with kernel `(K_d, K_h, K_w)`, returns `(N, C, D_out, H_out, W_out, K_d, K_h, K_w)`.

    1. Convolutional output axes `(D_out, H_out, W_out)` remain distinct.
    2. Kernel dims `(K_d, K_h, K_w)` stay separate, so each patch element’s location is obvious.

* **`Fold`**

  * **PyTorch 2D**: Expects `(N, C×K_h×K_w, L)` → reconstructs `(N, C, H, W)` by summing overlaps.
  * **torch\_nd\_conv N-D**: Takes `(N, C, D_out, H_out, W_out, K_d, K_h, K_w)` → reconstructs `(N, C, D′, H′, W′)`.

    1. Gathers each of the `K_d×K_h×K_w` elements from their `(D_out, H_out, W_out)` positions.
    2. Sums them along the reconstruction axes to rebuild the original volume.

By preserving kernel dimensions, `torch_nd_conv` makes it straightforward to generalize beyond 2D. No manual reshaping or axis permutation is needed when moving to 3D, 4D, or higher.

### Example Usage

Below is a basic example demonstrating how to utilize the `Conv` module alongside `Fold` and `Unfold` for a 3D convolution operation:

```python
import torch
from torch_nd_conv import Conv, Fold, Unfold

# Define input dimensions: (batch_size, channels, depth, height, width)
input_tensor = torch.randn(8, 3, 8, 16, 16)

# Initialize FoldND and UnfoldND
fold = Fold(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), kernel_position="last")
unfold = Unfold(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

# Initialize ConvND
conv = Conv(input_channels=3, output_channels=2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

# Perform unfold
unfolded = unfold(input_tensor)

# Perform fold
folded_output = fold(unfolded)

# Perform convolution
output = conv(unfolded)
```

## Installation

To get started with the ConvND repository, follow these steps to set up your development environment:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mntsx/torch_nd_conv.git
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment to manage dependencies.

   - **Windows:**

     ```bash
     py -3.12 -m venv .venv
     .venv\Scripts\activate
     python -m pip install --upgrade pip
     cd torch_nd_conv
     python -m pip install -r requirements.txt
     ```

   - **macOS/Linux:**

     ```bash
     python3.12 venv .venv
     source .venv/bin/activate
     python3 -m pip install --upgrade pip
     cd torch_nd_conv
     python3 -m pip install -r requirements.txt
     ```

## Running Benchmarks

To evaluate the performance of the custom n-dimensional convolution against PyTorch's native convolution functions, execute the `benchmark.conv` submodule:

```bash
cd torch_nd_conv
python -m benchmarks.conv
```

This will output the execution times and performance ratios for both 2D and 3D convolution operations.

## Testing

Ensure that all modules are functioning correctly by running the test suites using `pytest`:

```bash
cd torch_nd_conv
pytest .
```


## File Structure

The repository is organized aiming for functional separation. Below is an overview of the primary directories and their respective contents:

```
torch_nd_conv/
│
├── benchmarks/
│   ├── conv.py
│   └── __init__.py
│
├── src/
│   ├── conv.py
│   ├── fold.py
│   ├── internal_types.py
│   ├── utils.py
│   └── __init__.py
│
├── tests/
│   ├── pytest.ini
│   ├── test_conv.py
│   ├── test_fold.py
│   ├── test_unfold.py
│   └── __init__.py
│
├── __init__.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

### Directory and File Descriptions

#### `benchmarks/`

This directory contains benchmarking scripts that compare the performance of the custom n-dimensional convolution functions against PyTorch's native convolution operations in the dimensions where PyTorch provides built-in support.

- **`conv.py`**: Contains benchmarks that evaluate the performance of the n-dimensional convolution (`Conv`) against PyTorch's 2D (`conv2d`) and 3D (`conv3d`) convolution functions.

#### `src/`

The core implementations of the convolution operations and their auxiliary classes reside in this directory.

- **`conv.py`**: Defines the `Conv` module, implementing the n-dimensional convolution operation as a subclass of `torch.nn.Module`.
- **`fold.py`**: Contains the definitions for the n-dimensional `Fold` and `Unfold` classes, which are essential for preparing and reconstructing data for convolution operations in arbitrary dimensions.
- **`internal_types.py`**: Includes custom type definitions that enhance readability and maintainability through improved type hinting.
- **`utils.py`**: Provides utility functions used for validating hyperparameters and inputs for the `Fold`, `Unfold`, and `Conv` classes.
- **`__init__.py`**: Initializes the `src` package, facilitating easy imports of the modules within.

#### `tests/`

This directory houses test suites that ensure the correctness and reliability of the implemented modules using `pytest`.

- **`test_conv.py`**: Contains tests verifying the functionality and performance of the `Conv` module.
- **`test_fold.py`**: Includes tests for the `Fold` class, ensuring accurate data folding operations.
- **`test_unfold.py`**: Comprises tests for the `Unfold` class, validating the unfolding process.
- **`__init__.py`**: Initializes the `tests` package, enabling straightforward test discovery and execution.

