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
    Set the random seed.

    This function sets the random seed to ensure experiment reproducibility.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for the CuDNN backend,
            i.e., setting `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark`
            to False. Default: False.
    """

    # Set the random seed for Python
    random.seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)

    # Set the random seed for all available GPUs
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Set the deterministic mode for the CuDNN backend
        torch.backends.cudnn.deterministic = True
        # Disable benchmark mode in the CuDNN backend
        torch.backends.cudnn.benchmark = False


# Function: Train Segmentor
def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """
    Initiates training for the segmentor.

    Args:
        model (nn.Module): Model to be trained.
        dataset (Dataset or list[Dataset]): Training dataset. It can be a single instance
            of Dataset or a list of Dataset instances if using multiple training datasets.
        cfg (dict): Training configuration.
        distributed (bool): Indicates whether distributed training is used. Default: False.
        validate (bool): Indicates whether validation should be performed during training. Default: False.
        timestamp (str): Timestamp. Default: None.
        meta (dict): Metadata associated with the training. Default: None.
    """

    logger = get_root_logger(cfg.log_level)

    # Prepare data loaders
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

    # Place the model on GPUs
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Set the `find_unused_parameters` parameter in DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # Build the runner
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

    # Register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # A temporary fix to make .log and .log.json file names the same
    runner.timestamp = timestamp

    # Register evaluation hooks
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
