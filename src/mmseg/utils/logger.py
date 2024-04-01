import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """
    Obtiene el logger raíz.

    El logger se inicializará si no ha sido inicializado previamente. Por
    defecto, se añadirá un StreamHandler. Si se especifica `log_file`, también
    se añadirá un FileHandler. El nombre del logger raíz es el nombre del
    paquete de nivel superior, por ejemplo, "mmseg".

    Args:
        log_file (str | None): El nombre del archivo de registro. Si se
            especifica, se añadirá un FileHandler al logger raíz.
        log_level (int): El nivel del logger raíz. Tenga en cuenta que solo
            el proceso de rango 0 se verá afectado, mientras que otros
            procesos establecerán el nivel en "Error" y permanecerán en silencio
            la mayor parte del tiempo.

    Returns:
        logging.Logger: El logger raíz.
    """
    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger
