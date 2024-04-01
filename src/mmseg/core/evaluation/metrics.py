import mmcv
import numpy as np


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """
    Calcula la intersección y la unión entre las predicciones y las etiquetas.

    Args:
        pred_label (ndarray or str): Mapa de segmentación de las predicciones.
        label (ndarray or str): Mapa de segmentación de las etiquetas verdaderas.
        num_classes (int): Número de clases.
        ignore_index (int): Índice que será ignorado en la evaluación.
        label_map (dict): Mapeo de etiquetas antiguas a nuevas etiquetas. El parámetro
            solo funcionará cuando la etiqueta sea de tipo str. Por defecto: dict().
        reduce_zero_label (bool): Si se debe ignorar la etiqueta cero. El parámetro
            solo funcionará cuando la etiqueta sea de tipo str. Por defecto: False.

     Returns:
         ndarray: La intersección de los histogramas de predicción y etiqueta verdadera
             en todas las clases.
         ndarray: La unión de los histogramas de predicción y etiqueta verdadera en todas
             las clases.
         ndarray: El histograma de predicción en todas las clases.
         ndarray: El histograma de etiqueta verdadera en todas las clases.
    """
    # Carga los mapas de segmentación si son archivos de ruta
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    
    # Modifica las etiquetas si hay un mapeo personalizado
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
            
    # Reduce la etiqueta cero si es necesario
    if reduce_zero_label:
        # Evita la conversión de underflow
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    
    # Filtra las áreas de la predicción y la etiqueta
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    # Calcula la intersección
    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    # Calcula los histogramas de predicción y etiqueta
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    # Calcula la unión
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """
    Calcula la Intersección y Unión Total.

    Args:
        results (list[ndarray]): Lista de mapas de segmentación de predicción.
        gt_seg_maps (list[ndarray]): Lista de mapas de segmentación de las etiquetas verdaderas.
        num_classes (int): Número de categorías.
        ignore_index (int): Índice que se ignorará en la evaluación.
        label_map (dict): Mapeo de etiquetas antiguas a nuevas etiquetas. Por defecto: dict().
        reduce_zero_label (bool): Si se debe ignorar la etiqueta cero. Por defecto: False.

     Returns:
         ndarray: La intersección del histograma de predicción y etiqueta verdadera
             en todas las clases.
         ndarray: La unión del histograma de predicción y etiqueta verdadera en todas
             las clases.
         ndarray: El histograma de predicción en todas las clases.
         ndarray: El histograma de etiqueta verdadera en todas las clases.
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        # Calcula la intersección y unión para cada imagen
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        # Suma las áreas de intersección, unión, predicción y etiqueta para obtener el total.
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """
    Calcula la Media de Intersección y Unión (mIoU).

    Args:
        results (list[ndarray]): Lista de mapas de segmentación de predicción.
        gt_seg_maps (list[ndarray]): Lista de mapas de segmentación de las etiquetas verdaderas.
        num_classes (int): Número de categorías.
        ignore_index (int): Índice que se ignorará en la evaluación.
        nan_to_num (int, opcional): Si se especifica, los valores NaN serán reemplazados
            por los números definidos por el usuario. Por defecto: None.
        label_map (dict): Mapeo de etiquetas antiguas a nuevas etiquetas. Por defecto: dict().
        reduce_zero_label (bool): Si se debe ignorar la etiqueta cero. Por defecto: False.

     Returns:
         float: Precisión general en todas las imágenes.
         ndarray: Precisión por categoría, forma (num_classes, ).
         ndarray: IoU por categoría, forma (num_classes, ).
    """
    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """
    Calcula el Dado Medio (mDice).

    Args:
        results (list[ndarray]): Lista de mapas de segmentación de predicción.
        gt_seg_maps (list[ndarray]): Lista de mapas de segmentación de las etiquetas verdaderas.
        num_classes (int): Número de categorías.
        ignore_index (int): Índice que se ignorará en la evaluación.
        nan_to_num (int, opcional): Si se especifica, los valores NaN serán reemplazados
            por los números definidos por el usuario. Por defecto: None.
        label_map (dict): Mapeo de etiquetas antiguas a nuevas etiquetas. Por defecto: dict().
        reduce_zero_label (bool): Si se debe ignorar la etiqueta cero. Por defecto: False.

     Returns:
         float: Precisión general en todas las imágenes.
         ndarray: Precisión por categoría, forma (num_classes, ).
         ndarray: Dado por categoría, forma (num_classes, ).
    """
    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """
    Calcula las métricas de evaluación.
    
    Args:
        results (list[ndarray]): Lista de mapas de segmentación de predicción.
        gt_seg_maps (list[ndarray]): Lista de mapas de segmentación de las etiquetas verdaderas.
        num_classes (int): Número de categorías.
        ignore_index (int): Índice que se ignorará en la evaluación.
        metrics (list[str] | str): Métricas a evaluar, 'mIoU' y 'mDice'.
        nan_to_num (int, opcional): Si se especifica, los valores NaN serán reemplazados
            por los números definidos por el usuario. Por defecto: None.
        label_map (dict): Mapeo de etiquetas antiguas a nuevas etiquetas. Por defecto: dict().
        reduce_zero_label (bool): Si se debe ignorar la etiqueta cero. Por defecto: False.
        
    Returns:
        float: Precisión general en todas las imágenes.
        ndarray: Precisión por categoría, forma (num_classes, ).
        ndarray: Métricas de evaluación por categoría, forma (num_classes, ).
    """
    # Verifica si metrics es una cadena y conviértela en una lista si es necesario
    if isinstance(metrics, str):
        metrics = [metrics]
    # Lista de métricas permitidas
    allowed_metrics = ['mIoU', 'mDice']
    # Verifica si las métricas solicitadas están permitidas
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    # Calcula las áreas de intersección y unión para todas las imágenes
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    # Calcula la precisión general (macc)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    # Calcula la precisión por categoría (acc)
    acc = total_area_intersect / total_area_label
    # Inicializa la lista de métricas de retorno con macc y acc
    ret_metrics = [all_acc, acc]                            
    # Calcula las métricas de interés (mIoU o mDice)
    for metric in metrics:
        if metric == 'mIoU':
            # Calcula el IoU por categoría
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            # Calcula el Dice por categoría
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    # Reemplaza los valores NaN por el número especificado, si es necesario
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
        
    return ret_metrics
