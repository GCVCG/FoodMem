from mmcv.utils import Registry, build_from_cfg

PIXEL_SAMPLERS = Registry('pixel sampler')


def build_pixel_sampler(cfg, **default_args):
    """Construye un muestreador de píxeles para mapas de segmentación."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
