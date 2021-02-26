import tensorflow as tf

from qmc.tf.layers import QMeasureDensityEig
from qmc.tf.tt.layers import MPSLayer, MPSFeature


class MPSClassifier(tf.keras.Model):

    def __init__(self, n_sites=28 ** 2, d_bond=2, n_output=10, dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mps = MPSLayer(n_sites=n_sites, d_bond=d_bond, n_output=n_output, dim=dim)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, **kwargs):
        return self.softmax(self.mps(inputs))

    def get_config(self):
        return super(MPSClassifier, self).get_config()


class MPSClassifierQMC(tf.keras.Model):

    def __init__(self, n_sites=28 ** 2, d_bond=2, n_output=10, num_classes=10, dim=2, num_eig=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mps = MPSFeature(n_sites=n_sites, d_bond=d_bond, n_output=n_output, dim=dim)
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(QMeasureDensityEig(n_output, num_eig))

    def call(self, inputs, **kwargs):
        psi_x = self.mps(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors /
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def get_config(self):
        return super(MPSClassifierQMC, self).get_config()
