import tensorflow as tf
from slstm_cell import sLSTMCell
from mlstm_cell import mLSTMCell

class xLSTMBlock(tf.keras.layers.Layer):
    def __init__(self, units, block_type='sLSTM'):
        super(xLSTMBlock, self).__init__()
        self.block_type = block_type

        if block_type == 'sLSTM':
            self.cell = sLSTMCell(units)
        elif block_type == 'mLSTM':
            self.cell = mLSTMCell(units)
        else:
            raise ValueError("block_type must be 'sLSTM' or 'mLSTM'")

    def call(self, inputs, states):
        return self.cell(inputs, states)