from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import src.mmseg


def collect_env():
    """
    Collects information about the running environment.

    Returns:
        dict: Environment information.
    """

    # Collects basic environment information
    env_info = collect_base_env()

    # Adds specific information about MMSegmentation, including the version and git hash
    env_info['MMSegmentation'] = f'{src.mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    # Collects environment information and prints it
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
