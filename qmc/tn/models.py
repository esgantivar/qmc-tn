import tensorflow as tf
from tensornetwork.tn_keras.dense import DenseDecomp

from qmc.tf.layers import QMeasureDensityEig


class DenseCompClassifierQMC(tf.keras.Model):
    
    def __init__(self, n_sites=28 ** 2, n_output=100, num_classes=10, num_eig=100, decomp_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_decomp = DenseDecomp(n_output, decomp_size=decomp_size, input_shape=(n_sites,),activation='softmax')
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensityEig(n_output, num_eig))

    def call(self, inputs, **kwargs):
        psi_x = self.dense_decomp(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors /
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def get_config(self):
        return super(DenseCompClassifierQMC, self).get_config()
