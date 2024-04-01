import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from src.mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Resize(object):
    """
    Redimensiona imágenes y segmentaciones.

    Esta transformación redimensiona la imagen de entrada a una escala determinada. Si el diccionario de entrada
    contiene la clave "scale", entonces se utiliza la escala en el diccionario de entrada,
    de lo contrario se utiliza la escala especificada en el método init.

    ``img_scale`` puede ser Nong, una tupla (una sola escala) o una lista de tuplas
    (escala múltiple). Hay 4 modos de escala múltiple:

    - ``ratio_range is not None``:
    1. Cuando img_scale es None, img_scale es la forma de la imagen en los resultados
        (img_scale = results['img'].shape[:2]) y la imagen se redimensiona en función
        del tamaño original. (modo 1)
    2. Cuando img_scale es una tupla (una sola escala), se muestrea aleatoriamente una relación de
        el rango de relaciones y se multiplica con la escala de la imagen. (modo 2)

    - ``ratio_range is None and multiscale_mode == "range"``: se muestrea aleatoriamente una
    escala de un rango. (modo 3)

    - ``ratio_range is None and multiscale_mode == "value"``: se muestrea aleatoriamente una
    escala de múltiples escalas. (modo 4)

    Args:
        img_scale (tuple or list[tuple]): Escalas de imágenes para el redimensionamiento.
        multiscale_mode (str): Ya sea "range" o "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Si mantener la relación de aspecto al redimensionar la
            imagen.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # Modo 1: Dado img_scale=None y un rango de relación de imagen
            # Modo 2: Dado una escala y un rango de relación de imagen
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # Modo 3 y 4: Dadas múltiples escalas o un rango de escalas
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio


    @staticmethod
    def random_select(img_scales):
        """
        Selecciona aleatoriamente una escala de imagen de los candidatos dados.

        Args:
            img_scales (list[tuple]): Escalas de imágenes para la selección.

        Returns:
            (tuple, int): Devuelve una tupla ``(img_scale, scale_dix)``,
                donde ``img_scale`` es la escala de imagen seleccionada y
                ``scale_idx`` es el índice seleccionado en los candidatos dados.
        """
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx


    @staticmethod
    def random_sample(img_scales):
        """
        Muestra aleatoriamente una escala de imagen cuando ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Rango de escala de imágenes para muestrear.
                Deben haber dos tuplas en img_scales, que especifican el límite inferior
                y el límite superior de las escalas de imagen.

        Returns:
            (tuple, None): Devuelve una tupla ``(img_scale, None)``, donde
                ``img_scale`` es la escala muestreada y None es solo un marcador de posición
                para ser coherente con :func:`random_select`.
        """
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None


    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """
        Muestrea aleatoriamente una escala de imagen cuando se especifica ``ratio_range``.

        Se muestrea aleatoriamente una relación del rango especificado por
        ``ratio_range``. Luego se multiplicaría con ``img_scale`` para
        generar la escala muestreada.

        Args:
            img_scale (tuple): Escala base de imágenes para multiplicar con la relación.
            ratio_range (tuple[float]): La relación mínima y máxima para escalar
                la ``img_scale``.

        Returns:
            (tuple, None): Devuelve una tupla ``(escala, None)``, donde
                ``escala`` es la relación muestreada multiplicada por ``img_scale`` y
                None es solo un marcador de posición para ser coherente con
                :func:`random_select`.
        """
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None


    def _random_scale(self, results):
        """
        Muestrea aleatoriamente una escala de imagen de acuerdo con ``ratio_range`` y
        ``multiscale_mode``.

        Si se especifica ``ratio_range``, se muestreará una relación y
        se multiplicará con ``img_scale``.
        Si se especifican múltiples escalas mediante ``img_scale``, se muestreará una escala
        según ``multiscale_mode``.
        De lo contrario, se utilizará una sola escala.

        Args:
            results (dict): Diccionario de resultados de :obj:`dataset`.

        Returns:
            dict: Se agregan dos nuevas claves 'scale` y 'scale_idx` en
                ``results``, que serían utilizadas por las canalizaciones subsecuentes.
        """
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx


    def _resize_img(self, results):
        """Redimensiona las imágenes con ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # La w_scale y h_scale tiene una diferencia menor
            # se debería hacer una corrección real en el futuro en mmcv.imrescale
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # En caso de que no haya relleno
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio


    def _resize_seg(self, results):
        """Redimensiona el mapa de segmentación semántica con ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg


    def __call__(self, results):
        """
        Llama a la función para redimensionar imágenes, cajas delimitadoras, máscaras, segmentación semántica.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados redimensionados, se agregan las claves 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' al diccionario de resultados.
        """
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """
    Voltea la imagen y la segmentación.

    Si el diccionario de entrada contiene la clave "flip", entonces se usará el indicador,
    de lo contrario, se decidirá aleatoriamente según una proporción especificada en el método init.

    Args:
        prob (float, opcional): La probabilidad de volteo. Por defecto: None.
        direction(str, opcional): La dirección de volteo. Las opciones son
            'horizontal' y 'vertical'. Por defecto: 'horizontal'.
    """
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']


    def __call__(self, results):
        """Llama a la función para voltear las cajas delimitadoras, máscaras, mapas de segmentación semántica.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados volteados, se agregan las claves 'flip', 'flip_direction' al
                diccionario de resultados.
        """
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # Voltear imagen
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # Voltear segmentaciones
            for key in results.get('seg_fields', []):
                # Usar copy() para hacer positivo el paso de numpy
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Pad(object):
    """
    Rellena la imagen y la máscara.

    Hay dos modos de relleno: (1) rellenar a un tamaño fijo y (2) rellenar al
    tamaño mínimo que es divisible por algún número.
    Se agregan las claves "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, opcional): Tamaño de relleno fijo.
        size_divisor (int, opcional): El divisor del tamaño de relleno.
        pad_val (float, opcional): Valor de relleno. Por defecto: 0.
        seg_pad_val (float, opcional): Valor de relleno del mapa de segmentación.
            Por defecto: 255.
    """
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # Solo uno de size y size_divisor debería ser válido
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None


    def _pad_img(self, results):
        """Rellena las imágenes según ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


    def _pad_seg(self, results):
        """Rellena las máscaras según ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """
        Llama a la función para rellenar imágenes, máscaras, mapas de segmentación semántica.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Diccionario de resultados actualizado.
        """
        self._pad_img(results)
        self._pad_seg(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """
    Normaliza la imagen.

    Se agrega la clave "img_norm_cfg".

    Args:
        mean (sequence): Valores medios de los 3 canales.
        std (sequence): Valores de desviación estándar de los 3 canales.
        to_rgb (bool): Si convertir la imagen de BGR a RGB,
            el valor predeterminado es verdadero.
    """
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """
        Llama a la función para normalizar imágenes.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados normalizados, se agrega la clave 'img_norm_cfg' al
                diccionario de resultados.
        """
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Rerange(object):
    """
    Reajusta el valor de los píxeles de la imagen.

    Args:
        min_value (float or int): Valor mínimo de la imagen reajustada.
            Por defecto: 0.
        max_value (float or int): Valor máximo de la imagen reajustada.
            Por defecto: 255.
    """
    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """
        Llama a la función para reajustar imágenes.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.
        Returns:
            dict: Resultados reajustados.
        """
        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # Reajusta a [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # Reajusta a [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@PIPELINES.register_module()
class CLAHE(object):
    """
    Utiliza el método CLAHE para procesar la imagen.

    Consulte `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` para obtener más información.

    Args:
        clip_limit (float): Umbral para la limitación de contraste. Por defecto: 40.0.
        tile_grid_size (tuple[int]): Tamaño de la cuadrícula para la ecualización del histograma.
            La imagen de entrada se dividirá en rectángulos de tamaño igual.
            Define el número de rectángulos en fila y columna. Por defecto: (8, 8).
    """
    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size


    def __call__(self, results):
        """
        Llama a la función para utilizar el método CLAHE para procesar imágenes.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados procesados.
        """
        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """
    Recorta aleatoriamente la imagen y la segmentación.

    Args:
        crop_size (tuple): Tamaño esperado después del recorte, (h, w).
        cat_max_ratio (float): La proporción máxima que una sola categoría podría
            ocupar.
    """
    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index


    def get_crop_bbox(self, img):
        """Obtiene aleatoriamente una caja de recorte."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2


    def crop(self, img, crop_bbox):
        """Recorta de ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img


    def __call__(self, results):
        """
        Llama a la función para recortar imágenes de manera aleatoria y mapas de segmentación semántica.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados recortados aleatoriamente, la clave 'img_shape' en el diccionario de resultados se actualiza según el tamaño del recorte.
        """
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repetir 10 veces
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # Recortar la imagen
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # Recortar seg semántica
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate(object):
    """
    Rota la imagen y la segmentación.

    Args:
        prob (float): La probabilidad de rotación.
        degree (float, tuple[float]): Rango de grados para seleccionar. Si
            degree es un número en lugar de una tupla como (min, max),
            el rango de grados será (``-degree``, ``+degree``)
        pad_val (float, opcional): Valor de relleno de la imagen. Por defecto: 0.
        seg_pad_val (float, opcional): Valor de relleno del mapa de segmentación.
            Por defecto: 255.
        center (tuple[float], opcional): Punto central (w, h) de la rotación en
            la imagen fuente. Si no se especifica, se utilizará el centro de la imagen.
            Por defecto: None.
        auto_bound (bool): Si ajustar el tamaño de la imagen para cubrir toda la
            imagen rotada. Por defecto: False
    """
    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound


    def __call__(self, results):
        """
        Llama a la función para rotar la imagen, los mapas de segmentación.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados rotados.
        """
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # Rotar imagen
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # Rotar segmentación
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class RGB2Gray(object):
    """
    Convierte una imagen RGB en una imagen en escala de grises.

    Esta transformación calcula la media ponderada de los canales de imagen de entrada con
    ``weights`` y luego expande los canales a ``out_channels``. Cuando
    ``out_channels`` es None, el número de canales de salida es el mismo que
    los canales de entrada.

    Args:
        out_channels (int): Número esperado de canales de salida después de
            la transformación. Por defecto: None.
        weights (tuple[float]): Los pesos para calcular la media ponderada.
            Por defecto: (0.299, 0.587, 0.114).
    """
    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights


    def __call__(self, results):
        """
        Llama a la función para convertir una imagen RGB en una imagen en escala de grises.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Diccionario de resultados con imagen en escala de grises.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@PIPELINES.register_module()
class AdjustGamma(object):
    """
    Utiliza la corrección gamma para procesar la imagen.

    Args:
        gamma (float or int): Valor gamma utilizado en la corrección gamma.
            Por defecto: 1.0.
    """
    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')


    def __call__(self, results):
        """Llama a la función para procesar la imagen con corrección gamma.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Resultados procesados.
        """
        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results


    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@PIPELINES.register_module()
class SegRescale(object):
    """
    Reescala los mapas de segmentación semántica.

    Args:
        scale_factor (float): El factor de escala de la salida final.
    """
    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor


    def __call__(self, results):
        """
        Llama a la función para escalar el mapa de segmentación semántica.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Diccionario de resultados con el mapa de segmentación semántica escalado.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """
    Aplica distorsión fotométrica a la imagen secuencialmente, cada transformación
    se aplica con una probabilidad de 0.5. La posición del contraste aleatorio es en
    el segundo o penúltimo lugar.

    1. brillo aleatorio
    2. contraste aleatorio (modo 0)
    3. convertir color de BGR a HSV
    4. saturación aleatoria
    5. tono aleatorio
    6. convertir color de HSV a BGR
    7. contraste aleatorio (modo 1)
    8. intercambiar canales al azar

    Args:
        brightness_delta (int): delta de brillo.
        contrast_range (tuple): rango de contraste.
        saturation_range (tuple): rango de saturación.
        hue_delta (int): delta de tono.
    """
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta


    def convert(self, img, alpha=1, beta=0):
        """Multiplica por alpha y suma beta con recorte."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


    def brightness(self, img):
        """Distorsión de brillo."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img


    def contrast(self, img):
        """Distorsión de contraste."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img


    def saturation(self, img):
        """Distorsión de saturación."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img


    def hue(self, img):
        """Distorsión de tono."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img


    def __call__(self, results):
        """
        Llama a la función para realizar distorsión fotométrica en las imágenes.

        Args:
            results (dict): Diccionario de resultados de la canalización de carga.

        Returns:
            dict: Diccionario de resultados con imágenes distorsionadas.
        """
        img = results['img']
        # Brillo aleatorio
        img = self.brightness(img)

        # Modo == 0 --> Hacer contraste aleatorio primero
        # Modo == 1 --> Hacer contraste aleatorio al final
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # Saturación aleatoria
        img = self.saturation(img)

        # Tono aleatorio
        img = self.hue(img)

        # Contraste aleatorio
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
