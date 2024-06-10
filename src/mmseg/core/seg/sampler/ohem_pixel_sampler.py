import torch
import torch.nn.functional as F

from src.mmseg.core.seg.builder import PIXEL_SAMPLERS
from src.mmseg.core.seg.sampler.base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
    """
    Online Hard Example Mining (OHEM) sampler for segmentation.

    Args:
        context (nn.Module): The context of the sampler, subclass of
            :obj:`BaseDecodeHead`.
        thresh (float, optional): The threshold for selecting hard examples.
            Below this are predictions with low confidence. If not specified,
            hard examples will be pixels with the top ``min_kept`` losses.
            Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
    """

    def __init__(self, context, thresh=None, min_kept=100000):
        super(OHEMPixelSampler, self).__init__()
        self.context = context
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept


    def sample(self, seg_logit, seg_label):
        """
        Samples pixels with high loss or low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """

        with torch.no_grad():
            # Ensure shapes match
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1

            # Reduce label dimension
            seg_label = seg_label.squeeze(1).long()

            # Calculate number of samples kept per batch
            batch_kept = self.min_kept * seg_label.size(0)

            # Mask for valid pixels
            valid_mask = seg_label != self.context.ignore_index

            # Initialize segmentation weight tensor
            seg_weight = seg_logit.new_zeros(size=seg_label.size())

            # Extract valid segmentation weight
            valid_seg_weight = seg_weight[valid_mask]

            # Calculate segmentation weight based on threshold or top loss
            if self.thresh is not None:
                # Calculate softmax probability
                seg_prob = F.softmax(seg_logit, dim=1)

                # Clone and expand label
                tmp_seg_label = seg_label.clone().unsqueeze(1)

                # Replace ignore_index with 0
                tmp_seg_label[tmp_seg_label == self.context.ignore_index] = 0

                # Gather probabilities for each pixel class
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)

                # Sort probabilities for valid pixels
                sort_prob, sort_indices = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    # Calculate threshold based on batch_kept
                    min_threshold = sort_prob[min(batch_kept,
                                                  sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0

                # Ensure threshold is not lower than min_threshold or specified threshold
                threshold = max(min_threshold, self.thresh)

                # Set weight to 1 for pixels with probability below threshold
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:
                # Calculate loss
                losses = self.context.loss_decode(
                    seg_logit,
                    seg_label,
                    weight=None,
                    ignore_index=self.context.ignore_index,
                    reduction_override='none')

                # Sort losses for valid pixels
                _, sort_indices = losses[valid_mask].sort(descending=True)

                # Set weight to 1 for top losses of batch_kept
                valid_seg_weight[sort_indices[:batch_kept]] = 1.

            # Set segmentation weight for valid pixels
            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight
