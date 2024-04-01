from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import src.mmseg


def collect_env():
    """
    Recopila la información de los entornos en ejecución.
    
    Returns:
        dict: Información del entorno.
    """ 
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{src.mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
