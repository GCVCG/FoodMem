import os
import os.path as osp
import sys
import tempfile

import mmcv
import numpy as np
import torch

sys.path.append('.')
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel
from src.mmseg.apis.inference import inference_segmentor, init_segmentor
from src.mmseg.datasets.builder import build_dataloader, build_dataset
from src.mmseg.models.builder import build_segmentor
from torchview import draw_graph


def save_result(img_path,
                result,
                show_vis,
                color_list_path,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                vis_save_name='pred_vis',
                mask_save_name='pred_mask.png'):
    """
    Save and visualize the segmentation results on the given image.

    Args:
        img_path (str): Path to the input image.
        result (tuple): Segmentation result.
        show_vis (bool): Whether to save visualization image.
        color_list_path (str): Path to the numpy array file containing the color list.
        win_name (str, optional): Window name for displaying the image. Defaults to ''.
        show (bool, optional): Whether to display the image. Defaults to False.
        wait_time (int, optional): Wait time for display window. Defaults to 0.
        out_file (str, optional): Path to save the segmentation result. Defaults to None.
        vis_save_name (str, optional): Name for saving the visualization image. Defaults to 'pred_vis'.
        mask_save_name (str, optional): Name for saving the mask image. Defaults to 'pred_mask.png'.

    Returns:
        np.ndarray: The resulting image with segmentation overlay if not shown or saved.
    """

    # Read the image from the given path
    img = mmcv.imread(img_path)
    img = img.copy()

    # Extract the segmentation result
    seg = result[0]

    # Create an empty array for the color segmentation overlay
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

    # Load the color list from the given path and set the background color
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]

    # Apply colors to the segmentation result
    for label, color in enumerate(color_list):
        color_seg[seg == label, :] = color_list[label]

    # Overlay the color segmentation on the original image
    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    # If an output file path is provided, do not display the image
    if out_file is not None:
        show = False

    # Display the image if requested
    if show:
        mmcv.imshow(img, win_name, wait_time)

    # Save the visualization image if requested
    if show_vis:
        file_name = os.path.basename(img_path)
        file_stem, _ = os.path.splitext(file_name)
        vis_save_name = vis_save_name + "_" + f"{file_stem}" + ".png"
        mmcv.imwrite(img, os.path.join(out_file, "vis", vis_save_name))

    # Save the segmentation request if an output file is provided
    if out_file is not None:
        file_name = os.path.basename(img_path)
        file_stem, _ = os.path.splitext(file_name)
        out_file = os.path.join(out_file, f"{file_stem}.png")
        mmcv.imwrite(seg, out_file)

    # If neither displaying nor saving the image, return the image with overlay
    if not (show or out_file):
        print('show==False and out_file is not specified, only '
              'result image will be returned')

        return img


def np2tmp(array, temp_file_name=None):
    """
    Save a NumPy array to a temporary file.

    Args:
        array (np.ndarray): The NumPy array to save.
        temp_file_name (str, optional): The name of the temporary file. If None, a temporary file
                                        will be created with a .npy suffix. Defaults to None.

    Returns:
        str: The path to the temporary file where the array is saved.
    """

    # If no temp file name is provided, create a new temporary file with a .npy suffix
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name

    # Save the NumPy array to the temporary file
    np.save(temp_file_name, array)

    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    color_list_path,
                    show=False,
                    out_dir=None,
                    efficient_test=False, ):
    """
    Test a model on a dataset using a single GPU and save or show the results.

    Args:
        model (torch.nn.Module): The model to be tested.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        color_list_path (str): Path to the numpy array file containing the color list.
        show (bool, optional): Whether to display the images with segmentation results. Defaults to False.
        out_dir (str, optional): Directory to save the output images. Defaults to None.
        efficient_test (bool, optional): Whether to save results in a memory-efficient manner. Defaults to False.

    Returns:
        list: A list of results for the entire dataset.
    """

    # Set the model to evaluation mode
    model.eval()
    results = []
    dataset = data_loader.dataset

    # Initialize a progress bar
    prog_bar = mmcv.ProgressBar(len(dataset))

    # Iterate over the dataset
    for i, data in enumerate(data_loader):
        # Perform inference without gradient calculation
        with torch.no_grad():
            result = model(return_loss=False, **data)

        # If showing or saving results is requested
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

                # Save or show the result image
                save_result(
                    img_show,
                    result,
                    color_list_path=color_list_path,
                    show=show,
                    out_file=out_file)

        # Append the result to the results list
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)

        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)

        # Update the progress bar
        for _ in range(batch_size):
            prog_bar.update()

    return results


def semantic_predict(data_root, img_dir, ann_dir, config, options, aug_test, checkpoint, eval_options,
                     color_list_path, show_vis,
                     img_path=None, output_path=None):
    """
    Perform semantic segmentation prediction on a single image or a dataset.

    Args:
        data_root (str): The root directory of the dataset.
        img_dir (str): The directory containing images.
        ann_dir (str): The directory containing annotations.
        config (str): Path to the configuration file.
        options (dict): Additional configuration options to merge.
        aug_test (bool): Whether to use test-time augmentation.
        checkpoint (str): Path to the model checkpoint.
        eval_options (dict): Evaluation options.
        color_list_path (str): Path to the numpy array file containing the color list.
        show_vis (bool): Whether to save visualization images.
        img_path (str, optional): Path to a single image for prediction. Defaults to None.
        output_path (str, optional): Directory to save output images. Defaults to None.

    Returns:
        None
    """

    # Load the configuration file
    cfg = mmcv.Config.fromfile(config)

    # Merge additional options into the configuration if provided
    if options is not None:
        cfg.merge_from_dict(options)

    # Enable cuDNN benchmark if specified in the configuration
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Adjust test-time augmentation settings if enabled
    if aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True

    # Disable pretrained model loading and set test mode
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Predict on a single image
    if img_path:
        # No need to use load_checkpoint as it is done in init_segmentor
        model = init_segmentor(config, checkpoint)

        # Draw and visualize the model graph (if needed)
        #model.to('cuda:0')
        #model.forward = model.forward_dummy
        #model_graph = draw_graph(model, input_size=(1, 3, 768, 768), device='meta')
        #model_graph.visual_graph
        #model.forward = model.forward_dummy
        #summary(model, (3, 768, 768))
        #print(model)

        # Perform inference on the image
        result = inference_segmentor(model, img_path)

        # Save the result
        save_result(
            img_path,
            result,
            show_vis,
            color_list_path=color_list_path,
            show=False,
            out_file=output_path)

    # Predict on a dataset
    else:
        # No need to use load_checkpoint as it is done in init_segmentor
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

        # Draw and visualize the model graph (if needed)
        #model.to('cuda:0')
        #model.forward = model.forward_dummy
        #model_graph = draw_graph(model, input_size=(1, 3, 768, 768), device='meta')
        #model_graph.visual_graph
        #model.forward = model.forward_dummy
        #summary(model, (3, 768, 768))
        #print(model)

        # Configure the test dataset and data loader
        cfg.data.test.data_root = data_root
        cfg.data.test.img_dir = img_dir
        cfg.data.test.ann_dir = ann_dir
        dataset = build_dataset(cfg.data.test)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        efficient_test = False

        if eval_options is not None:
            efficient_test = eval_options.get('efficient_test', False)

        model = MMDataParallel(model, device_ids=[0])

        # Perform single GPU testing and save results
        single_gpu_test(model, data_loader, color_list_path, out_dir=output_path,
                                  efficient_test=efficient_test)
