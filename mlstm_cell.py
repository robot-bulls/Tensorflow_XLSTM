import tensorflow as tf
from exponential_gate import ExponentialGate

class mLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(mLSTMCell, self).__init__()
        self.units = units

        self.key_transform = tf.keras.layers.Dense(units)
        self.value_transform = tf.keras.layers.Dense(units)
        self.query_transform = tf.keras.layers.Dense(units)

        self.input_gate = ExponentialGate(units)
        self.forget_gate = ExponentialGate(units)
        self.output_gate = tf.keras.layers.Dense(units)

    def call(self, inputs, states):
        h_prev, c_prev = states
        k_t = self.key_transform(inputs)
        v_t = self.value_transform(inputs)
        q_t = self.query_transform(inputs)

        i_t = self.input_gate(inputs)
        f_t = tf.sigmoid(self.forget_gate(inputs))
        o_t = tf.sigmoid(self.output_gate(inputs))

        c_t = tf.matmul(tf.expand_dims(v_t, axis=2), tf.expand_dims(k_t, axis=1))
        h_t = tf.matmul(c_t, tf.expand_dims(q_t, axis=2))
        h_t = tf.squeeze(h_t, axis=-1)
        
        return h_t, [h_t, c_t]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)