from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from src.mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """
    Wrapper for a concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concatenates the aspect ratio group indicator for the image.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


@DATASETS.register_module()
class RepeatDataset(object):
    """
    Wrapper for a repeated dataset.

    The length of the repeated dataset will be `times` times larger than the original dataset.
    This is useful when data loading time is long but the dataset is small. Using RepeatDataset
    can reduce the data loading time between epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to repeat.
        times (int): Number of repetitions.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Gets an item from the original dataset."""

        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``."""

        return self.times * self._ori_len
