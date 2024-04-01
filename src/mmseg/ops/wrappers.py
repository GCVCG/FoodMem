import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """
    Redimensiona la entrada según el tamaño especificado o el factor de escala.

    Args:
        input (torch.Tensor): Datos de entrada.
        size (int or tuple, optional): Tamaño de salida. Si es un entero, especifica la altura y el ancho; si es una tupla, especifica (height, width). Por defecto es None.
        scale_factor (float or tuple, optional): Factor de escala. Si es un flotante, especifica el factor de escala; si es una tupla, especifica (scale_factor_height, scale_factor_width). Por defecto es None.
        mode (str, optional): Método de interpolación. Por defecto es 'nearest'.
        align_corners (bool, optional): Si es True, alinea las esquinas de la entrada y la salida, lo que puede ser inconsistente con la interpolación 'nearest'. Por defecto es None.
        warning (bool, optional): Si es True, emite una advertencia si es necesario. Por defecto es True.

    Returns:
        torch.Tensor: Datos de salida redimensionados.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):
    """
    Capa de aumento de tamaño.

    Args:
        size (int or tuple, optional): Tamaño de salida. Si es un entero, especifica la altura y el ancho; si es una tupla, especifica (height, width). Por defecto es None.
        scale_factor (float or tuple, optional): Factor de escala. Si es un flotante, especifica el factor de escala; si es una tupla, especifica (scale_factor_height, scale_factor_width). Por defecto es None.
        mode (str, optional): Método de interpolación. Por defecto es 'nearest'.
        align_corners (bool, optional): Si es True, alinea las esquinas de la entrada y la salida, lo que puede ser inconsistente con la interpolación 'nearest'. Por defecto es None.
    """
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        """
        Inicializa la capa de aumento de tamaño.

        Args:
            size (int or tuple, optional): Tamaño de salida. Si es un entero, especifica la altura y el ancho; si es una tupla, especifica (height, width). Por defecto es None.
            scale_factor (float or tuple, optional): Factor de escala. Si es un flotante, especifica el factor de escala; si es una tupla, especifica (scale_factor_height, scale_factor_width). Por defecto es None.
            mode (str, optional): Método de interpolación. Por defecto es 'nearest'.
            align_corners (bool, optional): Si es True, alinea las esquinas de la entrada y la salida, lo que puede ser inconsistente con la interpolación 'nearest'. Por defecto es None.
        """
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """
        Realiza la propagación hacia adelante de la red.

        Args:
            x (torch.Tensor): Datos de entrada.

        Returns:
            torch.Tensor: Datos de salida redimensionados.
        """
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
