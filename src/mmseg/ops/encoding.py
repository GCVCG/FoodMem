import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoding(nn.Module):
    """
    Learnable residual encoder layer.

    The input shape is (batch_size, channels, height, width).
    The output shape is (batch_size, num_codes, channels).

    Args:
        channels (int): Dimension of features or channels of features.
        num_codes (int): Number of code words.
    """

    def __init__(self, channels, num_codes):
        """
        Initializes the encoding layer.

        Args:
            channels (int): Dimension of features or channels of features.
            num_codes (int): Number of code words.
        """

        super(Encoding, self).__init__()

        # Initializing code words and scale factor
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
        Computes the scaled L2 distance between the data and the code words.

        Args:
            x (torch.Tensor): Input data.
            codewords (torch.Tensor): Code words.
            scale (torch.Tensor): Scale factor.

        Returns:
            torch.Tensor: Scaled L2 distance.
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
        Aggregates the encoded features.

        Args:
            assigment_weights (torch.Tensor): Assignment weights.
            x (torch.Tensor): Input data.
            codewords (torch.Tensor): Code words.

        Returns:
            torch.Tensor: Aggregated encoded features.
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
        Performs forward propagation of the network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Encoded features.
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
        Representation of the encoding layer.

        Returns:
            str: Representation of the encoding layer.
        """

        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'

        return repr_str
