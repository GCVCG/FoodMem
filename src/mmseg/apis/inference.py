import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from src.mmseg.datasets.pipelines import Compose
from src.mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """
    Initialize a segmentor from a configuration file.

    Args:
        config (str or :obj:`mmcv.Config`): Path to the configuration file or the configuration object.
        checkpoint (str, optional): Path to the checkpoint file. If None, the model will not load any weights.
        device (str, optional): Device option CPU/CUDA. Default is 'cuda:0'. Use 'cpu' to load the model on CPU.

    Returns:
        nn.Module: The constructed segmentor.
    """

    # Check if config is a string (file path) or a Config object
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))

    # Set the pretrained and train_cfg attributes to None to avoid loading weights or training configurations
    config.model.pretrained = None
    config.model.train_cfg = None

    # Build the segmentor from the model configuration
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))

    # If a checkpoint is provided, load the model weights and metadata
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location=device)
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

    # Save the configuration in the model for convenience
    model.cfg = config

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    return model


class LoadImage:
    """A simple pipeline to load images."""

    def __call__(self, results):
        """
        Call function to load images into the results.

        Args:
            results (dict): A results dictionary containing the filename
                of the image to be read.

        Returns:
            dict: The ``results`` dictionary updated with the loaded image and its metadata.
        """

        # Check if the image path in results is a string (file path)
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        # If it is not a string, set the filenames to None
        else:
            results['filename'] = None
            results['ori_filename'] = None

        # Read the image using mmcv
        img = mmcv.imread(results['img'])
        results['img'] = img

        # Save the original shape and the shape of the loaded image in results
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        return results


def inference_segmentor(model, img):
    """
    Infer image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str/ndarray or list[str/ndarray]): Image file(s) or loaded images.

    Returns:
        list[Tensor]: The segmentation result.
    """

    # Get the model configuration
    cfg = model.cfg

    # Get the device of the model
    device = next(model.parameters()).device

    # Build the test data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # Prepare the data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    # If the model is on GPU, distribute the data to the specified GPU
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    # If the model is on CPU, set image metadata
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # Perform inference with the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result


def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10)):
    """
    Visualize segmentation results on the image using pyplot.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image file name or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of the segmentation map.
            If None is provided, a random palette will be generated.
            Default is None.
        fig_size (tuple): Size of the pyplot figure. Default is (15, 10).
    """

    # If the model is wrapped in nn.DataParallel, unwrap it
    if hasattr(model, 'module'):
        model = model.module

    # Show segmentation results on the image
    img = model.show_result(img, result, palette=palette, show=False)

    # Plot the image using pyplot
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
    plt.savefig("demo.png")  # Save the plot to a file
