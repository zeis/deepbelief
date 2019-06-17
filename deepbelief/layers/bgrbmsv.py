import tensorflow as tf
import deepbelief.config as c
from . import bgrbm


class BGRBMSV(bgrbm.BGRBM):
    """Bernoulli-Gaussian RBM layer with shared variance parameter.

    The visible units are binary, and the hidden ones are Gaussian.

    Args:
        num_v: Number of visible units.
        num_h: Number of hidden units.
        W_std: Weight initialization standard deviation. Default: 0.001.
        lr: Initial learning rate. Default: 0.001.
        sigma_h_ph: Placeholder for the sigma parameter of the hidden units.
        temperature: Parameter of Concrete units.
            If None, standard bernoulli units are used. Default: None.
        bottom: A layer object to use as the bottom layer. Default: None.
        loss_plotter: GraphPlotter object. Default: None.
        distr_plotter: ImageRowPlotter object. Default: None.
        filter_plotter: ImageGridPlotter object. Default: None.
        latent_space_explorer: LatentSpaceExplorer or LatentSpaceExplorer2D
            object. Default: None.
        session: Session object.
        name: Name of the layer. Default: 'bgrbmsv'.
    """

    def __init__(self,
                 num_v,
                 num_h,
                 W_std=0.001,
                 lr=0.001,
                 sigma_h_ph=None,
                 temperature=None,
                 bottom=None,
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='BGRBMSV'):

        with tf.variable_scope(name):
            if sigma_h_ph is None:
                sigma_h_init = tf.ones(shape=1, dtype=c.float_type)
                self.opt_sigma_h = tf.get_variable(initializer=sigma_h_init,
                                                   name='opt_sigma_h')
                # Constrain sigma_h parameter to be positive
                self.sigma_h = tf.nn.softplus(self.opt_sigma_h)
            else:
                self.sigma_h = sigma_h_ph

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
