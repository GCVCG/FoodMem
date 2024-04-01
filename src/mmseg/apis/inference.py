import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from src.mmseg.datasets.pipelines import Compose
from src.mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """
    Inicializa un segmentador a partir de un archivo de configuración.

    Args:
        config (str or :obj:`mmcv.Config`): Ruta del archivo de configuración o el objeto de configuración.
        checkpoint (str, opcional): Ruta del punto de control. Si se deja en None, el modelo no cargará ningún peso.
        device (str, opcional): Opción del dispositivo CPU/CUDA. Por defecto 'cuda:0'. Usa 'cpu' para cargar el modelo en la CPU.

    Returns:
        nn.Module: El segmentador construido.
    """
    # Verifica si la configuración es una cadena de texto (ruta de archivo) o un objeto Config
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    # Establece los atributos pretrained y train_cfg de la configuración del modelo como None para evitar cargar pesos o configuraciones de entrenamiento
    config.model.pretrained = None
    config.model.train_cfg = None
    # Construye el segmentador a partir de la configuración del modelo
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    # Si se proporciona un punto de control, carga los pesos del modelo y los metadatos
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    # Guarda la configuración en el modelo para mayor conveniencia
    model.cfg = config
    # Mueve el modelo al dispositivo especificado y lo pone en modo de evaluación
    model.to(device)
    model.eval()
    
    return model


class LoadImage:
    """Un pipeline simple para cargar imágenes."""

    def __call__(self, results):
        """
        Función de llamada para cargar imágenes en los resultados.

        Args:
            results (dict): Un diccionario de resultados que contiene el nombre de archivo
                de la imagen que se va a leer.

        Returns:
            dict: Se devolverán los ``results`` que contienen la imagen cargada.
        """
        # Verifica si la imagen en los resultados es una cadena de texto (ruta de archivo)
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        # Si no es una cadena de texto, establece los nombres de archivo como None
        else:
            results['filename'] = None
            results['ori_filename'] = None
        # Lee la imagen utilizando mmcv
        img = mmcv.imread(results['img'])
        results['img'] = img
        # Guarda la forma original y la forma de la imagen cargada en los resultados
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        
        return results


def inference_segmentor(model, img):
    """
    Infiere imagen(es) con el segmentador.

    Args:
        model (nn.Module): El segmentador cargado.
        img (str/ndarray or list[str/ndarray]): Archivos de imagen o imágenes cargadas.

    Returns:
        (list[Tensor]): El resultado de segmentación.
    """
    # Obtiene la configuración del modelo
    cfg = model.cfg
    # Obtiene el dispositivo del modelo
    device = next(model.parameters()).device
    # Construye el pipeline de datos de prueba
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # Prepara los datos
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # Si el modelo está en GPU, distribuye los datos en la GPU especificada
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    # Si el modelo está en CPU, establece los metadatos de la imagen
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # Realiza la inferencia con el modelo
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        
    return result


def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10)):
    """
    Visualiza los resultados de segmentación en la imagen.

    Args:
        model (nn.Module): El segmentador cargado.
        img (str or np.ndarray): Nombre del archivo de imagen o imagen cargada.
        result (list): El resultado de segmentación.
        palette (list[list[int]]] | None): La paleta del mapa de segmentación.
            Si se proporciona None, se generará una paleta aleatoria. 
            Por defecto: None
        fig_size (tuple): Tamaño de la figura de pyplot. Por defecto: (15, 10)
    """
    if hasattr(model, 'module'):
        model = model.module
    # Muestra los resultados de segmentación en la imagen
    img = model.show_result(img, result, palette=palette, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    #plt.show()
    plt.savefig("demo.png")
