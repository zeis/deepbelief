import os
import json
import logging
import numpy as np
import h5py


class Data:
    """Data layer.

    This layer is represents an interface to sample from a HDF5 dataset.
    The data is automatically randomly shuffled at each new epoch (except
    the very first epoch, unless 'shuffle_first=True').

    Args:
        data_fname: File name of the dataset.
        has_labels: Boolean. If set to True labels will be loaded.
            Default: False.
        shuffle_first: Boolean. If set to True the data will be shuffled
            even in the first epoch. Default: False.
        batch_size: A number that specifies the size of a batch. Default: 1.
        log_epochs: If 'False' no messages about epochs will be logged.
            Default: 'True'.
        name: A string that specifies the name of the layer. Default: 'Data'.
    """
    # TODO: Make clear that labels are assumed to be numeric

    def __init__(self,
                 data_fname,
                 has_labels=False,
                 shuffle_first=False,
                 batch_size=1,
                 log_epochs=True,
                 name='Data'):
        self.name = name
        self.datapoints = None
        self.labels = None
        self.datapoint_size = None
        self.num_datapoints = 0
        self.epoch = 0
        self.next_batch_start = 0
        self.has_labels = has_labels
        self.batch_size = batch_size
        self.log_epochs = log_epochs

        assert os.path.isfile(data_fname), \
            'The file does not exist: %s' % data_fname

        self.logger = logging.getLogger(self.name)

        # Load HDF5 dataset
        data = h5py.File(data_fname, mode='r', driver='core')
        self.datapoints = data['datapoints']
        self.datapoint_size = self.datapoints.shape[1]
        self.num_datapoints = self.datapoints.shape[0]
        if has_labels:
            self.labels = data['labels']

        assert self.batch_size <= self.num_datapoints

        # Index map
        self._index_map = np.arange(self.num_datapoints)

        if shuffle_first:
            # Shuffle index map for the first time
            np.random.shuffle(self._index_map)

        self.logger.debug(self)

    def batch_shape(self):
        # TODO: Remove this function
        """Return the shape of a batch."""
        return self.batch_size, self.datapoints.shape[1]

    def next_batch(self):
        # TODO: Change this docsctring and include labels
        """Return the next batch of datapoints drawn from the dataset.

        The datapoints are automatically shuffled at each new epoch.
        """
        batch_start = self.next_batch_start
        self.next_batch_start += self.batch_size

        if self.next_batch_start > self.num_datapoints:
            self.epoch += 1  # Increment epoch counter
            batch_start = 0
            self.next_batch_start = self.batch_size
            if self.log_epochs:
                self.logger.info('Epoch: %d', self.epoch)
            np.random.shuffle(self._index_map)  # Shuffle index map

        batch_indices = sorted(list(
            self._index_map[batch_start:self.next_batch_start]))
        datapoint_batch = self.datapoints[batch_indices]

        if self.has_labels:
            label_batch = self.labels[batch_indices]
            return datapoint_batch, label_batch

        return datapoint_batch

    def _data_stat_dict(self, arr, arr_str):
        d = {}
        d[arr_str + '_array_dtype'] = str(arr.dtype)
        d[arr_str + '_array_min'] = np.asscalar(np.min(arr))
        d[arr_str + '_array_max'] = np.asscalar(np.max(arr))
        return d

    def __str__(self):
        d = self._data_stat_dict(self.datapoints, 'datapoint')
        d['batch_size'] = self.batch_size
        d['num_datapoints'] = self.num_datapoints
        d['datapoint_size'] = self.datapoint_size
        if self.has_labels:
            d.update(self._data_stat_dict(self.labels, 'label'))
        return json.dumps(d, indent=4, sort_keys=True)
