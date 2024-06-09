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
    Resize the input according to the specified size or scale factor.

    Args:
        input (torch.Tensor): Input data.
        size (int or tuple, optional): Output size. If an integer, it specifies the height and width; if a tuple, it specifies (height, width). Default is None.
        scale_factor (float or tuple, optional): Scale factor. If a float, it specifies the scale factor; if a tuple, it specifies (scale_factor_height, scale_factor_width). Default is None.
        mode (str, optional): Interpolation method. Default is 'nearest'.
        align_corners (bool, optional): If True, align the corners of input and output, which may be inconsistent with 'nearest' interpolation. Default is None.
        warning (bool, optional): If True, issue a warning if necessary. Default is True.

    Returns:
        torch.Tensor: Resized output data.
    """

    if warning:
        # Issue a warning if align_corners is True and resizing may lead to misalignment
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

    # Convert size to a tuple of integers if it's a torch.Size object
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    # Use F.interpolate for actual resizing
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):
    """
    Upsampling layer.

    Args:
        size (int or tuple, optional): Output size. If an integer, it specifies the height and width; if a tuple, it specifies (height, width). Default is None.
        scale_factor (float or tuple, optional): Scale factor. If a float, it specifies the scale factor; if a tuple, it specifies (scale_factor_height, scale_factor_width). Default is None.
        mode (str, optional): Interpolation method. Default is 'nearest'.
        align_corners (bool, optional): If True, align the corners of input and output, which may be inconsistent with 'nearest' interpolation. Default is None.
    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        """
        Initializes the upsample layer.

        Args:
            size (int or tuple, optional): Output size. If an integer, it specifies the height and width; if a tuple, it specifies (height, width). Default is None.
            scale_factor (float or tuple, optional): Scale factor. If a float, it specifies the scale factor; if a tuple, it specifies (scale_factor_height, scale_factor_width). Default is None.
            mode (str, optional): Interpolation method. Default is 'nearest'.
            align_corners (bool, optional): If True, align the corners of input and output, which may be inconsistent with 'nearest' interpolation. Default is None.
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
        Performs forward propagation of the network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Resized output data.
        """

        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size

        return resize(x, size, None, self.mode, self.align_corners)