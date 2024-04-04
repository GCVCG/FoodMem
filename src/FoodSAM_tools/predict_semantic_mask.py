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
from mmcv.runner import load_checkpoint
from src.mmseg.apis.inference import inference_segmentor, init_segmentor
from src.mmseg.datasets.builder import build_dataloader, build_dataset
from src.mmseg.models.builder import build_segmentor


def save_result(img_path,
                result,
                color_list_path,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                vis_save_name='pred_vis.png',
                mask_save_name='pred_mask.png'):
    img = mmcv.imread(img_path)
    img = img.copy()
    seg = result[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    color_list = np.load(color_list_path)
    color_list[0] = [238, 239, 20]

    for label, color in enumerate(color_list):
        color_seg[seg == label, :] = color_list[label]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)

    # TODO Hacer un condicional para que se muestre la visualizacion, poner argumento en la linea de entrada
    if out_file is not None:
        # mmcv.imwrite(img, os.path.join(out_file, vis_save_name))
        mmcv.imwrite(seg, out_file)

    if not (show or out_file):
        print('show==False and out_file is not specified, only '
              'result image will be returned')

        return img


def np2tmp(array, temp_file_name=None):
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name

    np.save(temp_file_name, array)

    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    color_list_path,
                    show=False,
                    out_dir=None,
                    efficient_test=False, ):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

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

                save_result(
                    img_show,
                    result,
                    color_list_path=color_list_path,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)

        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)

        for _ in range(batch_size):
            prog_bar.update()

    return results


def semantic_predict(data_root, img_dir, ann_dir, config, options, aug_test, checkpoint, eval_options, output,
                     color_list_path,
                     img_path=None, output_path=None):
    cfg = mmcv.Config.fromfile(config)

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if img_path:
        # No need to use load_checkpoint as it is done in init_segmentor
        model = init_segmentor(config, checkpoint)
        # load_checkpoint(model, checkpoint, map_location='cpu')
        result = inference_segmentor(model, img_path)
        # output_dir = os.path.join(output, os.path.basename(img_path).split('.')[0])
        print(output_path)
        save_result(
            img_path,
            result,
            color_list_path=color_list_path,
            show=False,
            out_file=output_path)

    else:
        # No need to use load_checkpoint as it is done in init_segmentor
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        # load_checkpoint(model, checkpoint, map_location='cpu')
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
        single_gpu_test(model, data_loader, color_list_path, out_dir=output,
                                  efficient_test=efficient_test)
