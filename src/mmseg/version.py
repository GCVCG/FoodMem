# Copyright (c) Open-MMLab. All rights reserved.

__version__ = '0.11.0'


def parse_version_info(version_str):
    """
    Analiza la cadena de versión y la convierte en una tupla de números
    enteros y cadenas.

    Args:
        version_str (str): La cadena de versión a analizar.

    Returns:
        tuple: Una tupla que contiene los números enteros y cadenas que
            representan la versión.
    """
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
