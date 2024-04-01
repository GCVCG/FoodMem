from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from src.mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """
    Envoltura de un conjunto de datos concatenado.

    Igual que :obj:`torch.utils.data.dataset.ConcatDataset`, pero
    concatena el indicador de grupo para la proporción de aspecto de la imagen.

    Args:
        datasets (list[:obj:`Dataset`]): Una lista de conjuntos de datos.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


@DATASETS.register_module()
class RepeatDataset(object):
    """
    Envoltura de un conjunto de datos repetido.

    La longitud del conjunto de datos repetido será `times` veces mayor que el conjunto
    de datos original. Esto es útil cuando el tiempo de carga de datos es largo pero
    el conjunto de datos es pequeño. Usar RepeatDataset puede reducir el tiempo de carga
    de datos entre épocas.

    Args:
        dataset (:obj:`Dataset`): El conjunto de datos a repetir.
        times (int): Número de repeticiones.
    """
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Obtiene un elemento del conjunto de datos original."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """La longitud se multiplica por ``times``"""
        return self.times * self._ori_len
