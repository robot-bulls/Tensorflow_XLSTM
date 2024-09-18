import tensorflow as tf
from xlstm_block import xLSTMBlock

class xLSTMModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers, block_types):
        super(xLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.blocks = [xLSTMBlock(hidden_dim, block_types[i]) for i in range(num_layers)]
        self.fc = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        h = [tf.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        c = [tf.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]

        ta = tf.TensorArray(dtype=tf.float32, size=seq_len)

        for t in tf.range(seq_len):
            xt = inputs[:, t, :]
            for i in range(self.num_layers):
                h[i], [h[i], c[i]] = self.blocks[i](xt, [h[i], c[i]])
                xt = h[i]
            ta = ta.write(t, h[-1])

        hidden_states = ta.stack()
        hidden_states = tf.transpose(hidden_states, [1, 0, 2])

        out = self.fc(hidden_states[:, -1, :])
        return out