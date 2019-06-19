import os
import pickle
from scipy import misc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import preprocessing


class Plotter:
    def __init__(self, output_dir, bbox_inches, fig_ext):
        self.output_dir = output_dir
        self.bbox_inches = bbox_inches
        self.fig_ext = fig_ext
        self.FIG_DPI = 72
        self.fig = None

        if self.output_dir is not None:
            assert os.path.isdir(self.output_dir) is True

    def save_fig(self, name):
        filename = name + self.fig_ext
        self.fig.savefig(
            os.path.join(self.output_dir, filename),
            dpi=self.FIG_DPI,
            bbox_inches=self.bbox_inches)

    def save_axis(self, name):
        filename = name + '.pickle'
        pickle.dump(self.fig.axes[0],
                    open(os.path.join(self.output_dir, filename), 'wb'))


class GraphPlotter(Plotter):
    def __init__(self,
                 xlim,
                 ylim,
                 xlabel=None,
                 ylabel=None,
                 line_style='bo-',
                 bbox_inches='tight',
                 output_dir=None,
                 fig_ext='.svg'):
        super().__init__(output_dir, bbox_inches, fig_ext)

        self.fig, ax = plt.subplots(1, 1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.plot([], [], line_style)

    def set_data(self, x_values, y_values, name):
        self.fig.axes[0].lines[0].set_data(x_values, y_values)
        self.save_fig(name)


class ImageRowPlotter(Plotter):
    def __init__(self,
                 num_images,
                 image_shape,
                 bbox_inches='tight',
                 output_dir=None,
                 fig_ext='.png'):
        super().__init__(output_dir, bbox_inches, fig_ext)

        self.num_images = num_images
        self.image_shape = image_shape

        self.fig, axes = plt.subplots(1, self.num_images)
        zeros = np.empty(self.image_shape)
        for i in range(self.num_images):
            axes[i].set_axis_off()
            self.fig.axes[i].imshow(zeros,
                                    cmap=plt.cm.Greys_r,
                                    interpolation='none',
                                    vmin=0,
                                    vmax=1)

    def set_data(self, images, name):
        for i in range(self.num_images):
            data = preprocessing.normalize(images[i])
            data = np.reshape(data, self.image_shape)
            self.fig.axes[i].images[0].set_data(data)
        self.save_fig(name)


class ImageGridPlotter(Plotter):
    def __init__(self,
                 num_images,
                 image_shape,
                 num_columns=10,
                 clip_min=None,
                 clip_max=None,
                 bbox_inches=None,
                 output_dir=None,
                 fig_ext='.png'):
        super().__init__(output_dir, bbox_inches, fig_ext)

        self.num_images = num_images
        self.image_shape = image_shape
        self.clip_min = clip_min
        self.clip_max = clip_max

        num_rows = np.ceil(num_images / num_columns).astype(int)

        content_height_pixels = self.image_shape[0] * num_rows
        content_width_pixels = self.image_shape[1] * num_columns

        white_space_height_pixels = 0
        white_space_width_pixels = 0
        # white_space_height_pixels = 2 * (num_rows - 1)
        # white_space_width_pixels = 2 * (num_columns - 1)

        fig_height_pixels = content_height_pixels + white_space_height_pixels
        fig_width_pixels = content_width_pixels + white_space_width_pixels

        white_space_height_percentage = (white_space_height_pixels /
                                         fig_height_pixels)
        white_space_width_percentage = (white_space_width_pixels /
                                        fig_width_pixels)

        self.fig, _ = plt.subplots(num_rows, num_columns)

        fig_width_inches = fig_width_pixels / self.FIG_DPI
        fig_height_inches = fig_height_pixels / self.FIG_DPI

        self.fig.set_size_inches(fig_width_inches, fig_height_inches)

        for i in range(num_rows * num_columns):
            self.fig.axes[i].set_axis_off()

        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=white_space_height_percentage,
                            wspace=white_space_width_percentage)

        empty_img = np.empty(self.image_shape)
        for i in range(num_images):
            self.fig.axes[i].imshow(empty_img,
                                    cmap=plt.cm.Greys_r,
                                    interpolation='none',
                                    vmin=0,
                                    vmax=1)

    def set_data(self, image_batch, name):
        if self.clip_min is not None or self.clip_max is not None:
            image_batch = np.clip(image_batch, self.clip_min, self.clip_max)
        image_batch = np.reshape(
            image_batch,
            (self.image_shape[0], self.image_shape[1], self.num_images))
        for i in range(self.num_images):
            data = preprocessing.normalize(image_batch[..., i])
            self.fig.axes[i].images[0].set_data(data)
        self.save_fig(name)


class LatentSpacePlotter(Plotter):
    def __init__(self,
                 xlim,
                 ylim,
                 bbox_inches,
                 output_dir=None,
                 fig_ext='.svg'):
        super().__init__(output_dir, bbox_inches, fig_ext)

        self.xlim = xlim
        self.ylim = ylim

    def build_grid(self, dx, dy, xlim, ylim, yflipud=False):
        x = np.arange(xlim[0], xlim[1] + dx, dx)
        y = np.arange(ylim[0], ylim[1] + dy, dy)
        xx, yy = np.meshgrid(x, y)
        xx = xx.ravel()[:, None]
        if yflipud:
            yy = np.flipud(yy.ravel())[:, None]
        else:
            yy = yy.ravel()[:, None]
        grid = np.hstack([xx, yy])
        return grid, x, y


class LatentPointPlotter(LatentSpacePlotter):
    # label_color_dict example: {2: '#1f77b4', 3: '#f22c40'}
    def __init__(self,
                 xlim,
                 ylim,
                 delta,
                 bbox_inches='tight',
                 label_color_dict=None,
                 show_plot=False,  # If True, a plot is shown and
                 # no figure is saved.
                 save_axis_plot=False,
                 output_dir=None,
                 fig_ext='.svg'):
        super().__init__(xlim=xlim,
                         ylim=ylim,
                         bbox_inches=bbox_inches,
                         output_dir=output_dir,
                         fig_ext=fig_ext)
        self.fig, ax = plt.subplots(1, 1)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        self.delta = delta
        dx = dy = self.delta
        self.grid, x, y = self.build_grid(
            dx, dy, self.xlim, self.ylim, yflipud=False)
        self.space_image_shape = (x.shape[0], y.shape[0])
        self.label_color_dict = label_color_dict
        self.show_plot = show_plot
        self.save_axis_plot = save_axis_plot
        if self.show_plot:
            self.open_figs = []  # List of figs to keep open

        zeros = np.empty(self.space_image_shape)
        self.fig.tight_layout()
        self.fig.axes[0].imshow(zeros,
                                cmap='gray_r',
                                vmin=0,
                                vmax=1,
                                extent=[x.min(), x.max(), y.min(), y.max()])

        if self.show_plot:
            self.open_figs.append(self.fig)

        if self.label_color_dict:
            labels = self.label_color_dict.keys()
            rectangles = []
            for label in labels:
                rectangles.append(matplotlib.patches.Rectangle(
                    (0, 0), 1, 1, fc=self.label_color_dict[label]))
                self.fig.axes[0].legend(rectangles, labels)

    def plot(self, X, var_diag=None, fig_name=None, labels=None):
        if var_diag is not None:
            z = np.reshape(var_diag, self.space_image_shape, order='C')
            z = np.flipud(z)
            z = preprocessing.normalize(z)
        else:
            z = np.zeros(self.space_image_shape)

        self.fig.axes[0].images[0].set_data(z)

        colors = '#1f77b4'
        if labels is not None:
            labels = labels.astype(int)
            assert self.label_color_dict is not None
            colors = [self.label_color_dict[int(label)]
                      for label in labels[:, 0]]

        scat = self.fig.axes[0].scatter(X[:, 0],
                                        X[:, 1],
                                        alpha=0.7,
                                        c=colors,
                                        edgecolors='none')
        if fig_name and self.show_plot == False:
            self.save_fig(fig_name)
            if self.save_axis_plot:
                self.save_axis(fig_name)
        elif self.show_plot:
            figs = list(map(plt.figure, plt.get_fignums()))
            for fig in figs:
                if fig not in self.open_figs:
                    plt.close(fig)
            plt.show()

        scat.remove()  # Remove scatter plot


class LatentSpaceExplorer:
    def __init__(self,
                 sample_shape,
                 num_sliders=3,
                 slider_min_val=-1.0,
                 slider_max_val=1.0,
                 add_variance_slider=True,
                 slider_variance_max_val=10.0,
                 slider_variance_init_val=3.0,
                 num_samples_to_average=100):

        # TODO: Dim Check
        self.num_sliders = num_sliders
        self.sample_shape = sample_shape
        self.num_samples_to_average = num_samples_to_average
        self.sample_fn = None
        self.variance_fn = None

        self.sliders = []
        self.slider_min_val = slider_min_val
        self.slider_max_val = slider_max_val
        self.latent_point = np.zeros(self.num_sliders)
        self.add_variance_slider = add_variance_slider

        assert self.slider_min_val < self.slider_max_val

        if self.add_variance_slider:
            self.slider_height = 1.0 / (self.num_sliders + 1)
        else:
            self.slider_height = 1.0 / self.num_sliders

        fig = plt.figure(figsize=(5, self.num_sliders * 0.3))
        fig.canvas.set_window_title('Latent Space Explorer')

        for i in range(self.num_sliders):
            self._add_slider(i, 'Dim ' + str(i + 1))

        if self.add_variance_slider:
            self._add_slider(pos=i + 1,
                             name='Pred Var',
                             valmin=0.0,
                             valmax=slider_variance_max_val,
                             valinit=slider_variance_init_val,
                             facecolor='orange',
                             on_change=False)

        self.fig2 = plt.figure(figsize=(2, 2))
        self.ax2 = self.fig2.add_axes([0, 0, 1, 1])
        self.ax2.set_axis_off()

        self.fig2.axes[0].imshow(np.zeros(shape=self.sample_shape),
                                 cmap=plt.cm.Greys_r,
                                 interpolation='none',
                                 vmin=0,
                                 vmax=1)
        self.fig2.canvas.set_window_title('Output')

    def _add_slider(self,
                    pos,
                    name,
                    valmin=None,
                    valmax=None,
                    valinit=None,
                    facecolor=None,
                    on_change=True):
        valmin = valmin if valmin is not None else self.slider_min_val
        valmax = valmax if valmax is not None else self.slider_max_val
        valinit = valinit if valinit is not None else \
            (self.slider_min_val + self.slider_max_val) / 2

        left = 0.15
        bottom = 0.1 * self.slider_height + pos * self.slider_height
        width = 0.75
        height = 0.8 * self.slider_height
        ax = plt.axes([left, bottom, width, height])
        slider = matplotlib.widgets.Slider(ax,
                                           name,
                                           valmin=valmin,
                                           valmax=valmax,
                                           valinit=valinit,
                                           facecolor=facecolor)
        self.sliders.append(slider)
        if on_change:
            slider.on_changed(self._update)

    def _update(self, val):
        for i in range(self.num_sliders):
            self.latent_point[i] = self.sliders[i].val
        self._plot_sample()

    def _plot_sample(self):
        assert self.sample_fn is not None

        datapoint = self.sample_fn(
            self.latent_point, self.num_samples_to_average)

        if self.add_variance_slider:
            pred_var = self.variance_fn(self.latent_point)
            self.sliders[-1].set_val(pred_var[0][0])

        datapoint = np.reshape(datapoint, self.sample_shape)
        datapoint = preprocessing.normalize(datapoint)

        self.ax2.images[0].set_data(datapoint)
        self.fig2.canvas.draw()

    def set_sample_callback(self, fn):
        # 'fn' must take as parameters a latent point (numpy.ndarray),
        # and an integer (number of samples to average) and return an
        # image (numpy.ndarray).
        self.sample_fn = fn

    def set_variance_callback(self, fn):
        # 'fn' must take as parameters a latent point (numpy.ndarray),
        # and return the predictive variance value (note this is a
        # two-dimensional numpy.ndarray containing a numeric value).
        self.variance_fn = fn

    def plot(self):
        plt.show()


class LatentSpaceExplorer2D(LatentPointPlotter):
    def __init__(self,
                 xlim,
                 ylim,
                 delta,
                 sample_shape,
                 output_dir=None,
                 label_color_dict=None,
                 num_samples_to_average=100):
        super().__init__(xlim=xlim,
                         ylim=ylim,
                         delta=delta,
                         label_color_dict=label_color_dict,
                         output_dir=output_dir,
                         show_plot=True)

        self.sample_shape = sample_shape
        self.num_samples_to_average = num_samples_to_average
        self.sample_fn = None

        self.fig2 = plt.figure(figsize=(1, 1))
        self.ax2 = self.fig2.add_axes([0, 0, 1, 1])
        self.ax2.set_axis_off()

        self.datapoint = None

        self.fig2.axes[0].imshow(np.zeros(shape=self.sample_shape),
                                 cmap=plt.cm.Greys_r,
                                 interpolation='none',
                                 vmin=0,
                                 vmax=1)
        self.open_figs.append(self.fig2)

    def set_sample_callback(self, fn):
        # 'fn' must take as parameters a latent point (numpy.ndarray),
        # and an integer (number of samples to average) and return an
        # image (numpy.ndarray).
        self.sample_fn = fn
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_move(self, event):
        if event.inaxes is not None and event.inaxes == self.fig.axes[0]:
            x, y = event.xdata, event.ydata

            latent_point = np.array([x, y])

            assert self.sample_fn is not None
            self.datapoint = self.sample_fn(
                latent_point, self.num_samples_to_average)

            self.datapoint = np.reshape(self.datapoint, self.sample_shape)
            self.datapoint = preprocessing.normalize(self.datapoint)

            self.ax2.images[0].set_data(self.datapoint)
            self.fig2.canvas.draw()

    def _on_click(self, event):
        x, y = event.xdata, event.ydata
        misc.imsave(
            os.path.join(self.output_dir, str(x) + '.' + str(y) + '.png'),
            self.datapoint)


class LatentSamplePlotter(LatentSpacePlotter):
    def __init__(self,
                 image_shape,
                 xlim,
                 ylim,
                 delta=0.5,
                 num_samples_to_average=100,
                 bbox_inches=None,
                 output_dir=None,
                 fig_ext='.png'):
        super().__init__(xlim, ylim, bbox_inches, output_dir, fig_ext)

        self.delta = delta
        dx = dy = self.delta
        self.image_shape = image_shape
        self.grid, x, _ = self.build_grid(dx,
                                          dy,
                                          self.xlim,
                                          self.ylim,
                                          yflipud=True)
        self.num_samples_to_average = num_samples_to_average
        self.output_dir = output_dir

        self.num_images = self.grid.shape[0]
        self.fig_ext = fig_ext
        self.grid_plotter = ImageGridPlotter(num_images=self.num_images,
                                             image_shape=image_shape,
                                             num_columns=len(x),
                                             output_dir=output_dir,
                                             fig_ext=fig_ext)

    def rebuild_grid(self, dx, dy, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.grid, x, _ = self.build_grid(dx,
                                          dy,
                                          self.xlim,
                                          self.ylim,
                                          yflipud=True)

        self.num_images = self.grid.shape[0]
        self.grid_plotter = ImageGridPlotter(num_images=self.num_images,
                                             image_shape=self.image_shape,
                                             num_columns=len(x),
                                             output_dir=self.output_dir,
                                             fig_ext=self.fig_ext)

    def plot(self, image_vectors, fig_name):
        assert image_vectors.shape[1] == \
               self.image_shape[0] * self.image_shape[1]
        image_vectors = np.transpose(image_vectors)
        self.grid_plotter.set_data(image_vectors, fig_name)
