import torch
import torch.nn.functional as F

from src.mmseg.core.seg.builder import PIXEL_SAMPLERS
from src.mmseg.core.seg.sampler.base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
    """
    Muestreador de Ejemplos Difíciles en Línea (OHEM) para segmentación.

    Args:
        context (nn.Module): El contexto del muestreador, subclase de
            :obj:`BaseDecodeHead`.
        thresh (float, opcional): El umbral para la selección de ejemplos difíciles.
            Por debajo de esto, hay predicciones con baja confianza. Si no se
            especifica, los ejemplos difíciles serán píxeles de las mejores
            ``min_kept`` pérdidas. Por defecto: None.
        min_kept (int, opcional): El número mínimo de predicciones a mantener.
            Por defecto: 100000.
    """
    def __init__(self, context, thresh=None, min_kept=100000):
        super(OHEMPixelSampler, self).__init__()
        self.context = context
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept


    def sample(self, seg_logit, seg_label):
        """
        Muestra píxeles con alta pérdida o con baja confianza de predicción.

        Args:
            seg_logit (torch.Tensor): logits de segmentación, forma (N, C, H, W)
            seg_label (torch.Tensor): etiqueta de segmentación, forma (N, 1, H, W)

        Returns:
            torch.Tensor: peso de segmentación, forma (N, H, W)
        """
        with torch.no_grad():
            # Asegura que las formas coincidan
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1
            # Reduce la dimensión de la etiqueta
            seg_label = seg_label.squeeze(1).long()
            # Calcula el número de muestras mantenidas por lote
            batch_kept = self.min_kept * seg_label.size(0)
            # Máscara para píxeles válidos
            valid_mask = seg_label != self.context.ignore_index
            # Inicializa el tensor de peso de segmentación
            seg_weight = seg_logit.new_zeros(size=seg_label.size())
            # Extrae el peso de segmentación válido
            valid_seg_weight = seg_weight[valid_mask]
            
            # Calcula el peso de segmentación basado en el umbral o la pérdida superior
            if self.thresh is not None:
                # Calcula la probabilidad softmax
                seg_prob = F.softmax(seg_logit, dim=1)
                # Clona y expande la etiqueta
                tmp_seg_label = seg_label.clone().unsqueeze(1)
                # Reemplaza ignore_index con 0
                tmp_seg_label[tmp_seg_label == self.context.ignore_index] = 0
                # Recoge las probabilidades para cada clase de píxel
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
                # Ordena las probabilidades para los píxeles válidos
                sort_prob, sort_indices = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    # Calcula el umbral basado en batch_kept
                    min_threshold = sort_prob[min(batch_kept,
                                                  sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                # Asegura que el umbral no sea menor que min_threshold o el umbral especificado
                threshold = max(min_threshold, self.thresh)
                # Establece el peso en 1 para los píxeles con probabilidad por debajo del umbral
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:
                # Calcula la pérdida
                losses = self.context.loss_decode(
                    seg_logit,
                    seg_label,
                    weight=None,
                    ignore_index=self.context.ignore_index,
                    reduction_override='none')
                # Ordena las pérdidas para los píxeles válidos
                _, sort_indices = losses[valid_mask].sort(descending=True)
                # Establece el peso en 1 para las principales pérdidas de batch_kept
                valid_seg_weight[sort_indices[:batch_kept]] = 1.
            # Establece el peso de segmentación para los píxeles válidos
            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight
