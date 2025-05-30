# PyTorch ND Convolution Repository

[![LICENSE: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)](https://github.com/python)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mntsx/torch_nd_conv)
[![GitHub stars](https://img.shields.io/github/stars/mntsx/torch_nd_conv.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/mntsx/torch_nd_conv/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/mntsx/torch_nd_conv.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/mntsx/torch_nd_conv/network/)


## Introduction

Welcome to the **torch_nd_conv** repository, a Python-based implementation of n-dimensional convolution operations using PyTorch. This repository introduces a flexible and generalized approach to convolution operations beyond the native capabilities of PyTorch, enabling convolutions in arbitrary dimensions.

### Key Features

- **Arbitrary-Dimensional Convolution (`Conv`)**: Implements convolution operations in any number of dimensions, extending beyond PyTorch's native 2D and 3D convolutions.
- **Auxiliary Classes (`Fold` and `Unfold`)**: Provides `Fold` and `Unfold` classes necessary for the implementation and manipulation of n-dimensional convolutions.
- **PyTorch Integration**: All classes inherit from `torch.nn.Module`, ensuring seamless integration with existing PyTorch workflows.
- **Python 3.12+**: Developed using Python version 3.12, ensuring compatibility with the latest Python features and optimizations.

**Note**: PyTorch does not offer built-in modules for n-dimensional convolutions. This repository fills that gap by providing robust and efficient implementations tailored for high-dimensional data processing.

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

## Usage Notes

While the modules provided in this repository are built upon PyTorch's architecture and conventions, there are some key differences in their usage compared to PyTorch's native modules. Below are important considerations and guidelines for effectively utilizing the `Fold`, `Unfold`, and `Conv` classes.

### Differences from PyTorch's Native Modules

#### 1. **N-Dimensional Operations**

- **PyTorch Limitation**: PyTorch's native `torch.nn.Fold` and `torch.nn.Unfold` are limited to 2D operations.
- **Conv ND Advantage**: The `Fold` and `Unfold` classes extend these operations to n-dimensions, allowing for more versatile and generalized convolution processes.

#### 2. **Class Initialization Parameters**

Both `Fold` and `Unfold` offer additional parameters to handle n-dimensional data effectively:

- **`kernel_position` (for `Fold`)**:
  - **`"last"` (default)**: Expects input dimensions in the order `(*batch_dims, *conv_output_dims, *kernel_dims)`.
  - **`"first"`**: Expects input dimensions in the order `(*batch_dims, *kernel_dims, *conv_output_dims)`.
  - **Purpose**: Provides flexibility in how input dimensions are arranged, catering to different data formats and convolution configurations.

- **`input_size` and `output_size`**:
  - **Purpose**: Allow pre-calculations during initialization, potentially accelerating the `forward` method by avoiding repetitive computations.
  - **Usage**: When provided, these parameters should correspond to the non-batched input size (excluding batch dimensions). This ensures consistency and correctness in the unfolding and folding processes.

#### 3. **Input and Output Formats**

- **`Fold` and `Unfold` Output Structures**:
  - **`Fold`**: Reconstructs the original input from its unfolded representation, maintaining separate dimensions for kernel elements.
  - **`Unfold`**: Unfolds the input data into a shape that retains kernel dimensions separately, as opposed to collapsing them into a single dimension in PyTorch's native `Unfold`.

- **Comparison with PyTorch**:
  - **PyTorch's `Fold`**: Typically handles input in the shape `(N, C * kernel_height * kernel_width, L)`, where `L` is the number of sliding windows per convolution input.
  - **Conv's `Fold`**: Maintains separate kernel dimensions, resulting in an output shape of `(*batch_dims, *conv_output_dims, *kernel_dims)` for more intuitive handling in n-dimensional spaces.

#### 4. **Validation and Error Handling**

- **Utility Functions**: The `utils.py` module provides functions such as `param_check`, `fold_input_check`, and `unfold_input_check` to validate hyperparameters and input sizes, ensuring that the convolution operations are configured correctly.
- **Error Messages**: Comprehensive error handling is implemented to notify users of mismatches in input sizes or incorrect parameter configurations, enhancing the robustness of the modules.

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
   cd torch_nd_conv
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   - **Windows:**

     ```bash
     py -3.12 -m venv .venv
     ```

   - **macOS/Linux:**

     ```bash
     python3.12 venv .venv
     ```

3. **Activate the Virtual Environment**

   - **Windows:**

     ```bash
     .venv\Scripts\activate
     ```

   - **macOS/Linux:**

     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running Benchmarks

To evaluate the performance of the custom n-dimensional convolution against PyTorch's native convolution functions, execute the `benchmark.conv` submodule:

```bash
python -m benchmarks.conv
```

This will output the execution times and performance ratios for both 2D and 3D convolution operations.

## Testing

Ensure that all modules are functioning correctly by running the test suites using `pytest`:

```bash
pytest .
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes. Ensure that all new features are accompanied by appropriate tests.

---
