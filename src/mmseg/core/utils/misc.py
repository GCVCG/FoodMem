def add_prefix(inputs, prefix):
    """
    Adds a prefix to a dictionary.

    Args:
        inputs (dict): The input dictionary with keys of type str.
        prefix (str): The prefix to add.

    Returns:
        dict: The dictionary with keys updated with the ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
