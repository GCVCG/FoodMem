def add_prefix(inputs, prefix):
    """
    Agrega un prefijo a un diccionario.

    Args:
        inputs (dict): El diccionario de entrada con claves de tipo str.
        prefix (str): El prefijo a agregar.

    Returns:
        dict: El diccionario con las claves actualizadas con el ``prefijo``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
