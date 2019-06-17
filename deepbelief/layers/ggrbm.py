from . import gbrbm
from . import bgrbm


class GGRBM(gbrbm.GBRBM, bgrbm.BGRBM):
    """Gaussian-Gaussian RBM layer.

    The visible and hidden units are Gaussian. The variance vector of the
    visible units is assumed to be fixed to 1.

    Args:
        num_v: Number of visible units.
        num_h: Number of hidden units.
        W_std: Weight initialization standard deviation. Default: 0.001.
        lr: Initial learning rate. Default: 0.001.
        bottom: A layer object to use as the bottom layer. Default: None.
        loss_plotter: GraphPlotter object. Default: None.
        distr_plotter: ImageRowPlotter object. Default: None.
        filter_plotter: ImageGridPlotter object. Default: None.
        latent_space_explorer: LatentSpaceExplorer or LatentSpaceExplorer2D
            object. Default: None.
        session: Session object.
        name: Name of the layer. Default: 'GGRBM'.
    """

    def __init__(self,
                 num_v,
                 num_h,
                 W_std=0.001,
                 lr=0.001,
                 bottom=None,
                 loss_plotter=None,
                 distr_plotter=None,
                 filter_plotter=None,
                 latent_sample_plotter=None,
                 latent_space_explorer=None,
                 session=None,
                 name='GGRBM'):
        super().__init__(num_v=num_v,
                         num_h=num_h,
                         W_std=W_std,
                         lr=lr,
                         bottom=bottom,
                         loss_plotter=loss_plotter,
                         distr_plotter=distr_plotter,
                         filter_plotter=filter_plotter,
                         latent_sample_plotter=latent_sample_plotter,
                         latent_space_explorer=latent_space_explorer,
                         session=session,
                         name=name)
