import torch
from torch import nn as nn
from torch.nn import functional as F


class Encoding(nn.Module):
    """
    Capa de codificación: un codificador residual aprendible.

    La entrada tiene forma (batch_size, canales, alto, ancho).
    La salida tiene forma (batch_size, num_codes, canales).

    Args:
        channels: dimensión de las características o canales de características
        num_codes: número de palabras de código
    """
    def __init__(self, channels, num_codes):
        """
        Inicializa la capa de codificación.

        Args:
            channels (int): Dimensión de las características o canales de características.
            num_codes (int): Número de palabras de código.
        """
        super(Encoding, self).__init__()
        # Inicialización de palabras de código y factor de suavizado
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)


    @staticmethod
    def scaled_l2(x, codewords, scale):
        """
        Calcula la distancia L2 escalada entre los datos y las palabras de código.

        Args:
            x (torch.Tensor): Datos de entrada.
            codewords (torch.Tensor): Palabras de código.
            scale (torch.Tensor): Factor de escala.

        Returns:
            torch.Tensor: Distancia L2 escalada.
        """
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm


    @staticmethod
    def aggregate(assigment_weights, x, codewords):
        """
        Agrega las características codificadas.

        Args:
            assigment_weights (torch.Tensor): Pesos de asignación.
            x (torch.Tensor): Datos de entrada.
            codewords (torch.Tensor): Palabras de código.

        Returns:
            torch.Tensor: Características codificadas agregadas.
        """
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)

        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assigment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        """
        Realiza la propagación hacia adelante de la red.

        Args:
            x (torch.Tensor): Datos de entrada.

        Returns:
            torch.Tensor: Características codificadas.
        """
        assert x.dim() == 4 and x.size(1) == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        # assignment_weights: [batch_size, channels, num_codes]
        assigment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        encoded_feat = self.aggregate(assigment_weights, x, self.codewords)
        return encoded_feat


    def __repr__(self):
        """
        Representación de la capa de codificación.

        Returns:
            str: Representación de la capa de codificación.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'
        return repr_str
