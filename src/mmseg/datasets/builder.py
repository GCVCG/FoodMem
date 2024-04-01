import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from torch.utils.data import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    """Construye un ConcatDataset."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Construye conjuntos de datos."""
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     **kwargs):
    """
    Construye DataLoader de PyTorch.

    En el entrenamiento distribuido, cada GPU/proceso tiene un DataLoader.
    En el entrenamiento no distribuido, solo hay un DataLoader para todas las GPUs.

    Args:
        dataset (Dataset): Un conjunto de datos de PyTorch.
        samples_per_gpu (int): Número de muestras de entrenamiento en cada GPU, es decir,
            tamaño del lote de cada GPU.
        workers_per_gpu (int): Cuántos subprocesos usar para la carga de datos
            para cada GPU.
        num_gpus (int): Número de GPUs. Solo se usa en el entrenamiento no distribuido.
        dist (bool): Entrenamiento/prueba distribuido o no. Predeterminado: True.
        shuffle (bool): Si mezclar los datos en cada época.
            Predeterminado: True.
        seed (int | None): Semilla a usar. Predeterminado: None.
        drop_last (bool): Si omitir el último lote incompleto en la época.
            Predeterminado: False
        pin_memory (bool): Si usar pin_memory en DataLoader.
            Predeterminado: True
        dataloader_type (str): Tipo de DataLoader. Predeterminado: 'PoolDataLoader'
        kwargs: cualquier argumento de palabra clave para inicializar DataLoader

    Returns:
        DataLoader: Un DataLoader de PyTorch.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """
    Función de inicialización del trabajador para el DataLoader.

    La semilla de cada trabajador es igual a num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): ID del trabajador.
        num_workers (int): Número de trabajadores.
        rank (int): El rango del proceso actual.
        seed (int): La semilla aleatoria a usar.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
