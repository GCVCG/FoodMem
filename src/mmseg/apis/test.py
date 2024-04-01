import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None):
    """
    Guarda una matriz ndarray en un archivo numpy local.

    Args:
        array (ndarray): Matriz ndarray que se va a guardar.
        temp_file_name (str): Nombre del archivo numpy. Si 'temp_file_name' es None,
            esta función generará un nombre de archivo utilizando tempfile.NamedTemporaryFile
            para guardar la matriz ndarray. Por defecto, es None.

    Returns:
        str: El nombre del archivo numpy generado.
    """
    # Si no se proporciona un nombre de archivo temporal, se genera uno
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    # Guarda la matriz ndarray en el archivo numpy
    np.save(temp_file_name, array)
    
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    color_list_path,
                    show=False,
                    out_dir=None,
                    efficient_test=False,):
    """
    Realiza la prueba con un solo GPU.

    Args:
        model (nn.Module): Modelo que se va a probar.
        data_loader (utils.data.Dataloader): Cargador de datos de Pytorch.
        color_list_path (str): Ruta al archivo de lista de colores.
        show (bool): Indica si mostrar resultados durante la inferencia. Por defecto: False.
        out_dir (str, opcional): Si se especifica, los resultados se guardarán en
            el directorio para guardar los resultados de salida.
        efficient_test (bool): Indica si guardar los resultados como archivos numpy locales para
            ahorrar memoria de la CPU durante la evaluación. Por defecto: False.

    Returns:
        list: Los resultados de predicción.
    """
    # Pone el modelo en modo de evaluación
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # Itera sobre los datos del cargador de datos
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Realiza la inferencia con el modelo
            result = model(return_loss=False, **data)
        # Muestra los resultados o los guarda en el directorio especificado
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'].split('.')[0])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    color_list_path=color_list_path,
                    show=show,
                    out_file=out_file)
                
        # Guarda los resultados como archivos numpy locales si efficient_test es True
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)
        
        # Actualiza la barra de progreso
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
            
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """
    Prueba del modelo con múltiples GPUs.

    Esta función prueba un modelo con múltiples GPUs y recopila los resultados
    en dos modos diferentes: gpu y cpu. Al establecer 'gpu_collect=True', se
    codifican los resultados en tensores de gpu y se utiliza la comunicación
    gpu para la recopilación de resultados. En el modo cpu, se guardan los
    resultados en diferentes GPUs en 'tmpdir' y se recopilan por el trabajador
    de rango 0.

    Args:
        model (nn.Module): El modelo a probar.
        data_loader (utils.data.Dataloader): Cargador de datos de Pytorch.
        tmpdir (str): Ruta del directorio para guardar los resultados temporales
            de las diferentes GPUs en modo cpu.
        gpu_collect (bool): Opción para usar ya sea gpu o cpu para recopilar
            los resultados.
        efficient_test (bool): Si se deben guardar los resultados como archivos
            numpy locales para ahorrar memoria de la CPU durante la evaluación.
            Por defecto: False.

    Returns:
        list: Los resultados de la predicción.
    """
    # Pone el modelo en modo de evaluación
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    # Si el rango es 0, crea una barra de progreso
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    # Itera sobre los datos del cargador de datos
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Realiza la inferencia con el modelo
            result = model(return_loss=False, rescale=True, **data)

        # Guarda los resultados como archivos numpy locales si efficient_test es True
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        # Actualiza la barra de progreso si el rango es 0
        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # Recopila los resultados de todos los rangos
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
        
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """
    Recopila resultados con CPU.

    Args:
        result_part (list): Lista de resultados parciales de la evaluación.
        size (int): Tamaño total esperado de los resultados.
        tmpdir (str): Ruta del directorio para guardar los resultados temporales.
            Si no se especifica, se creará un directorio temporal.

    Returns:
        list: Los resultados ordenados de la evaluación.
    """
    # Obtiene el rango y el tamaño del mundo (número total de procesos)
    rank, world_size = get_dist_info()
    # Crea un directorio temporal si no se especifica
    if tmpdir is None:
        MAX_LEN = 512
        # 32 es un espacio en blanco
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            # Crea un directorio temporal
            tmpdir = tempfile.mkdtemp()
            # Convierte el nombre del directorio en un tensor de bytes en CUDA
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            # Copia el nombre del directorio al tensor
            dir_tensor[:len(tmpdir)] = tmpdir
        # Transmite el tensor a todos los procesos
        dist.broadcast(dir_tensor, 0)
        # Convierte el tensor en un string y lo asigna a tmpdir
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    # Crea el directorio si no existe
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # Guarda los resultados parciales en el directorio temporal
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # Barrera para asegurar que todos los procesos hayan guardado sus resultados
    dist.barrier()
    
    # Recolecta todos los resultados parciales
    if rank != 0:
        return None
    # Carga los resultados de todos los procesos desde el directorio temporal
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # Ordena los resultados
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # Algunos procesos pueden haber cargado más resultados que el tamaño esperado así que los recortamos
        ordered_results = ordered_results[:size]
        # Elimina el directorio temporal
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """
    Recopila los resultados con GPU.

    Args:
        result_part (list): Lista de resultados parciales.
        size (int): Tamaño total esperado de los resultados.

    Returns:
        list: Lista ordenada de resultados.
    """
    # Obtiene el rango y el tamaño del mundo (número total de procesos)
    rank, world_size = get_dist_info()
    # Convierte la lista de resultados parciales a un tensor en GPU
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # Recolecta la forma de cada tensor de resultado parcial
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # Rellena el tensor de resultados parciales para tener la misma longitud
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # Recolecta todos los resultados parciales
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            # Carga los resultados de cada tensor y los agrega a una lista
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # Ordena los resultados
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # Algunos procesos pueden haber cargado más resultados que el tamaño esperado así que los recortamos
        ordered_results = ordered_results[:size]
        return ordered_results
