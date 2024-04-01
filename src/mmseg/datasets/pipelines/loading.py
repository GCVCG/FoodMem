import os.path as osp

import mmcv
import numpy as np

from src.mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """
    Carga una imagen desde un archivo.

    Las claves requeridas son "img_prefix" e "img_info" (un diccionario que debe contener la
    clave "filename"). Las claves agregadas o actualizadas son "filename", "img", "img_shape",
    "ori_shape" (igual a `img_shape`), "pad_shape" (igual a `img_shape`),
    "scale_factor" (1.0) y "img_norm_cfg" (means=0 y stds=1).

    Args:
        to_float32 (bool): Si convertir la imagen cargada a un array numpy float32.
            Si se establece en False, la imagen cargada es un array uint8.
            Por defecto: False.
        color_type (str): El argumento de bandera para :func:`mmcv.imfrombytes`.
            Por defecto: 'color'.
        file_client_args (dict): Argumentos para instanciar un FileClient.
            Ver :class:`mmcv.fileio.FileClient` para más detalles.
            Por defecto: ``dict(backend='disk')``.
        imdecode_backend (str): Backend para :func:`mmcv.imdecode`. Por defecto:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend


    def __call__(self, results):
        """
        Llama a las funciones para cargar la imagen y obtener información de metaimagen.

        Args:
            results (dict): Diccionario de resultados de :obj:`mmseg.CustomDataset`.

        Returns:
            dict: El diccionario contiene la imagen cargada e información de metaimagen.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """
    Carga anotaciones para segmentación semántica.

    Args:
        reduce_zero_label (bool): Si reducir todos los valores de etiqueta en 1.
            Usualmente se utiliza para conjuntos de datos donde 0 es la etiqueta de fondo.
            Por defecto: False.
        file_client_args (dict): Argumentos para instanciar un FileClient.
            Ver :class:`mmcv.fileio.FileClient` para más detalles.
            Por defecto: ``dict(backend='disk')``.
        imdecode_backend (str): Backend para :func:`mmcv.imdecode`. Por defecto:
            'pillow'
    """
    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """
        Llama a la función para cargar anotaciones de varios tipos.

        Args:
            results (dict): Diccionario de resultados de :obj:`mmseg.CustomDataset`.

        Returns:
            dict: El diccionario contiene las anotaciones de segmentación semántica cargadas.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        if len(gt_semantic_seg.shape) == 3:
            gt_semantic_seg = gt_semantic_seg[:,:,2]

        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
