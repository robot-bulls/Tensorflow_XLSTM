import tensorflow as tf
from exponential_gate import ExponentialGate

class sLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(sLSTMCell, self).__init__()
        self.units = units

        self.input_gate = ExponentialGate(units)
        self.forget_gate = ExponentialGate(units)
        self.output_gate = tf.keras.layers.Dense(units)

        self.input_transform = tf.keras.layers.Dense(units)
        self.hidden_transform = tf.keras.layers.Dense(units)

    def call(self, inputs, states):
        h_prev, c_prev = states
        i_t = self.input_gate(inputs)
        f_t = tf.sigmoid(self.forget_gate(inputs))
        o_t = tf.sigmoid(self.output_gate(inputs))

        c_t = f_t * c_prev + i_t * tf.tanh(self.input_transform(inputs))
        h_t = o_t * tf.tanh(c_t)
        return h_t, [h_t, c_t]