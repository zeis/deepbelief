import tensorflow as tf
from . import rbm
from deepbelief import preprocessing
import deepbelief.config as c


class BGRBM(rbm.RBM):
    """Bernoulli-Gaussian RBM layer.

    The visible units are binary, and the hidden ones are Gaussian.

    Args:
        num_v: Number of visible units.
        num_h: Number of hidden units.
        W_std: Weight initialization standard deviation. Default: 0.001.
        lr: Initial learning rate. Default: 0.001.
        temperature: Parameter of Concrete units.
            If None, standard bernoulli units are used. Default: None.
        bottom: A layer object to use as the bottom layer. Default: None.
        loss_plotter: GraphPlotter object. Default: None.
        distr_plotter: ImageRowPlotter object. Default: None.
        filter_plotter: ImageGridPlotter object. Default: None.
        latent_space_explorer: LatentSpaceExplorer or LatentSpaceExplorer2D
            object. Default: None.
        session: Session object.
        name: Name of the layer. Default: 'BGRBM'.
    """

    def __init__(self,
                 num_v,
                 num_h,
                 W_std=0.001,
                 lr=0.001,
                 temperature=None,
                 bottom=None,
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='BGRBM'):

        if not hasattr(self, 'sigma_h'):
            with tf.variable_scope(name):
                sigma_h_init = tf.ones(shape=(1, num_h), dtype=c.float_type)
                sigma_h_init = preprocessing.inverse_softplus(sigma_h_init)
                self.opt_sigma_h = tf.get_variable(initializer=sigma_h_init,
                                                   name='opt_sigma_h')
                # Constrain sigma_h parameter to be positive
                self.sigma_h = tf.nn.softplus(self.opt_sigma_h)

        super().__init__(num_v=num_v,
                         num_h=num_h,
                         W_std=W_std,
                         lr=lr,
                         temperature=temperature,
                         bottom=bottom,
                         loss_plotter=loss_plotter,
                         distr_plotter=distr_plotter,
                         filter_plotter=filter_plotter,
                         latent_sample_plotter=latent_sample_plotter,
                         latent_space_explorer=latent_space_explorer,
                         session=session,
                         name=name)

        self.params['sigma_h'] = self.sigma_h

    def h_net_input(self, v_batch):
        """Return the network input afferent to the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        matmul = tf.matmul(v_batch, self.W)
        mul = tf.multiply(self.sigma_h, matmul)
        add = tf.add(mul, self.c)
        return add

    def h_probs(self, v_batch):
        """Return the probability of activation of the hidden units.

        Args:
            v_batch: A tensor of shape NxV, where N is the batch size and V is
                the number of visible units.

        Returns:
            A tensor of shape NxH, where H is the number of hidden units.
        """
        return self.h_net_input(v_batch)

    def sample_h(self, h_probs, sample=True, summary=True):
        """Return a sample from the hidden units.

        Args:
            h_probs: A tensor of probabilities of shape NxH, where N is the
                batch size and H is the number of hidden units.

        Returns:
            A tensor of shape NxH.
        """
        if sample:
            batch_size = tf.shape(h_probs)[0]
            gaussian_noise = tf.random_normal(shape=(batch_size, self.num_h))
            mul = tf.multiply(self.sigma_h, gaussian_noise)
            h_batch = tf.add(h_probs, mul)
        else:
            h_batch = h_probs
        tf.summary.histogram(self.name + '_hidden_activations', h_batch)
        return h_batch

    def v_net_input(self, h_batch):
        """Return the network input afferent to the visible units.

        Args:
            h_batch: A tensor of shape NxH, where N is the batch size and H is
                the number of hidden units.

        Returns:
            A tensor of shape NxV, where V is the number of visible units.
        """
        div = tf.div(h_batch, self.sigma_h)
        transpose1 = tf.transpose(div)
        matmul = tf.matmul(self.W, transpose1)
        add = tf.add(matmul, self.b)
        transpose2 = tf.transpose(add)
        return transpose2

    def free_energy(self, v_batch):
        """Calculate the free energy of each datapoint in a batch.

        Args:
            v_batch: A tensor of datapoints of shape NxV. N is the batch size
                and V is the number of visible units.

        Returns:
            A tensor of shape Nx1.
        """
        matmul1 = tf.matmul(v_batch, self.b)
        matmul2 = tf.matmul(v_batch, self.W)
        square = tf.square(matmul2)
        mul1 = tf.multiply(0.5, square)
        div = tf.div(self.c, self.sigma_h)
        mul2 = tf.multiply(div, matmul2)
        add = tf.add(mul1, mul2)
        reduce_sum = tf.reduce_sum(add, axis=1, keepdims=True)
        add2 = tf.add(matmul1, reduce_sum)
        neg = tf.negative(add2)
        return neg

    def _grad_var_list(self, lower_layers=False):
        var_list = [self.W, self.b, self.c, self.opt_sigma_h]
        if lower_layers and self.bottom:
            var_list = var_list + self.bottom._grad_var_list(lower_layers=True)
        return var_list
