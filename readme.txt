# xLSTM Model Implementation

This repository contains an implementation of an xLSTM model using TensorFlow. The model includes various LSTM cell types, specifically sLSTM and mLSTM, and is designed for sequence prediction tasks.

## Features

- Implementation of `sLSTMCell` and `mLSTMCell` classes.
- Flexible `xLSTMBlock` that can use different LSTM cell types.
- `xLSTMModel` class for building and training the model.
- Easy to extend and modify for different sequence prediction tasks.

## Requirements

To run this project, you will need:

- Python 3.6 or higher
- TensorFlow 2.x
- NumPy (optional, depending on your data processing)

You can install the required packages using pip:

```bash
pip install tensorflow numpy