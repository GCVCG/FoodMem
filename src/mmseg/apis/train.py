import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from src.mmseg.core.evaluation import DistEvalHook, EvalHook
from src.mmseg.datasets import build_dataloader, build_dataset
from src.mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """
    Establece la semilla aleatoria.

    Esta función establece la semilla aleatoria para garantizar la reproducibilidad de los experimentos.

    Args:
        seed (int): Semilla a utilizar.
        deterministic (bool): Si se debe establecer la opción determinista para el backend de CUDNN,
            es decir, establecer `torch.backends.cudnn.deterministic` en True y `torch.backends.cudnn.benchmark`
            en False. Por defecto: False.
    """
    # Establece la semilla para la generación de números aleatorios en Python
    random.seed(seed)
    # Establece la semilla para la generación de números aleatorios en NumPy
    np.random.seed(seed)
    # Establece la semilla para la generación de números aleatorios en PyTorch
    torch.manual_seed(seed)
    # Establece la semilla para la generación de números aleatorios en todas las GPUs disponibles
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Establece el modo determinista en el backend de CUDNN
        torch.backends.cudnn.deterministic = True
        # Desactiva el modo de referencia (benchmark) en el backend de CUDNN
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """
    Inicia el entrenamiento del segmentador.

    Args:
        model (nn.Module): Modelo a ser entrenado.
        dataset (Dataset or list[Dataset]): Conjunto de datos de entrenamiento. Puede ser una sola instancia
            de Dataset o una lista de instancias de Dataset si se está utilizando un conjunto de datos de
            entrenamiento múltiple.
        cfg (dict): Configuración del entrenamiento.
        distributed (bool): Indica si se está utilizando el entrenamiento distribuido. Por defecto: False.
        validate (bool): Indica si se debe realizar la validación durante el entrenamiento. Por defecto: False.
        timestamp (str): Marca de tiempo. Por defecto: None.
        meta (dict): Metadatos asociados con el entrenamiento. Por defecto: None.
    """
    logger = get_root_logger(cfg.log_level)
    # Prepara los cargadores de datos
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # Coloca el modelo en las GPUs
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Establece el parámetro `find_unused_parameters` en DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # Construye el runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # Registra los hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # Una solución temporal para que los nombres de los archivos .log y .log.json sean los mismos
    runner.timestamp = timestamp

    # Registra los hooks de evaluación
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
