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
    Saves a ndarray to a local numpy file.

    Args:
        array (ndarray): The ndarray to be saved.
        temp_file_name (str): The name of the numpy file. If 'temp_file_name' is None,
            this function will generate a file name using tempfile.NamedTemporaryFile
            to save the ndarray. Defaults to None.

    Returns:
        str: The generated numpy file name.
    """

    # If a temporary file name is not provided, generate one
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name

    # Save the ndarray to the numpy file
    np.save(temp_file_name, array)

    return temp_file_name


def single_gpu_test(model, data_loader, color_list_path, show=False, out_dir=None, efficient_test=False):
    """
    Perform testing with a single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch DataLoader.
        color_list_path (str): Path to the color list file.
        show (bool): Whether to display results during inference. Default: False.
        out_dir (str, optional): If specified, results will be saved in
            the directory for saving output results.
        efficient_test (bool): Whether to save results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: Prediction results.
    """

    # Set the model to evaluation mode
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    # Iterate over the data loader
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Perform inference with the model
            result = model(return_loss=False, **data)

        # Display or save results in the specified directory
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

        # Save results as local numpy files if efficient_test is True
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        # Update the progress bar
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, efficient_test=False):
    """
    Test the model with multiple GPUs.

    This function tests a model with multiple GPUs and collects results
    in two different modes: GPU and CPU. Setting 'gpu_collect=True' encodes
    results into gpu tensors and uses gpu communication for result collection.
    In the cpu mode, results on different GPUs are saved in 'tmpdir' and
    collected by the worker of rank 0.

    Args:
        model (nn.Module): The model to test.
        data_loader (utils.data.Dataloader): Pytorch DataLoader.
        tmpdir (str): Path of the directory to save temporary results
            from different GPUs in cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu for result collection.
        efficient_test (bool): Whether to save results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: Prediction results.
    """

    # Set the model to evaluation mode
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    # If rank is 0, create a progress bar
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    # Iterate over the data loader
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # Perform inference with the model
            result = model(return_loss=False, rescale=True, **data)

        # Save results as local numpy files if efficient_test is True
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        # Update the progress bar if rank is 0
        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # Collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """
    Collect results using CPU.

    Args:
        result_part (list): List of partial evaluation results.
        size (int): Total expected size of the results.
        tmpdir (str): Path of the directory to save temporary results.
            If not specified, a temporary directory will be created.

    Returns:
        list: Ordered evaluation results.
    """

    # Get the rank and world size (total number of processes)
    rank, world_size = get_dist_info()

    # Create a temporary directory if not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is a whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            # Create a temporary directory
            tmpdir = tempfile.mkdtemp()
            # Convert the directory name to a byte tensor on CUDA
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            # Copy the directory name to the tensor
            dir_tensor[:len(tmpdir)] = tmpdir

        # Broadcast the tensor to all processes
        dist.broadcast(dir_tensor, 0)
        # Convert the tensor to a string and assign it to tmpdir
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()

    # Create the directory if it does not exist
    else:
        mmcv.mkdir_or_exist(tmpdir)

    # Save the partial results in the temporary directory
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # Barrier to ensure all processes have saved their results
    dist.barrier()

    # Collect all partial results
    if rank != 0:
        return None
    # Load the results from all processes from the temporary directory
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))

        # Order the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # Some processes might have loaded more results than the expected size so trim them
        ordered_results = ordered_results[:size]

        # Remove the temporary directory
        shutil.rmtree(tmpdir)

        return ordered_results



def collect_results_gpu(result_part, size):
    """
    Collect results using GPU.

    Args:
        result_part (list): List of partial results.
        size (int): Total expected size of the results.

    Returns:
        list: Ordered list of results.
    """

    # Get the rank and world size (total number of processes)
    rank, world_size = get_dist_info()

    # Convert the list of partial results to a tensor on GPU
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')

    # Collect the shape of each partial result tensor
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    # Pad the partial result tensor to have the same length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]

    # Collect all partial results
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            # Load the results from each tensor and append them to a list
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))

        # Order the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # Some processes might have loaded more results than the expected size so trim them
        ordered_results = ordered_results[:size]

        return ordered_results
