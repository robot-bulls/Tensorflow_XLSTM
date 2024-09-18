import tensorflow as tf

class ExponentialGate(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ExponentialGate, self).__init__()
        self.units = units
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return tf.exp(self.dense(inputs))