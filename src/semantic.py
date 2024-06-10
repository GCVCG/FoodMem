import argparse

from mmcv.utils import DictAction

from src.FoodSAM_tools.predict_semantic_mask import semantic_predict

parser = argparse.ArgumentParser(
    description=(
        "Runs FoodSAM on one frame"
    )
)
parser.add_argument(
    "--data_root",
    type=str,
    default='dataset/FoodSeg103/Images',
    help="Path to folder of images and masks.",
)
parser.add_argument(
    "--img_dir",
    type=str,
    default='img_dir/test',
    help="dir name of images",
)
parser.add_argument(
    "--ann_dir",
    type=str,
    default='ann_dir/test',
    help="dir name of gt masks.",
)
parser.add_argument(
    '--semantic_config',
    default="configs/SETR_MLA_768x768_80k_base.py",
    help='test config file path of mmseg'
)
parser.add_argument(
    '--options',
    nargs='+',
    action=DictAction,
    help='custom options'
)
parser.add_argument(
    '--aug-test',
    action='store_true',
    help='Use Flip and Multi scale aug'
)
parser.add_argument(
    '--semantic_checkpoint',
    default="ckpts/SETR_MLA/iter_80000.pth",
    help='checkpoint file of mmseg'
)
parser.add_argument(
    '--eval-options',
    nargs='+',
    action=DictAction,
    help='custom options for evaluation'
)
parser.add_argument(
    '--color_list_path',
    type=str,
    default="src/FoodSAM_tools/color_list.npy",
    help='the color used to draw for each label'
)
parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="dir name of imgs.",
)
parser.add_argument(
    "--out_path",
    type=str,
    help="output file path",
)

parser.add_argument('--video', type=str,
                    help='Path to the video file or directory with .jpg video frames to process')
parser.add_argument('--masks', type=str,
                    help='Path to the directory with individual .png masks for corresponding video frames')
parser.add_argument('--frame_number', type=int,
                    help='Frame number to process')
parser.add_argument(
    "--show_vis",
    type=bool,
    default=False,
    help="vis pred mask",
)


def main(args: argparse.Namespace) -> None:
    semantic_predict(data_root=args.data_root, img_dir=args.img_dir, ann_dir=args.ann_dir, config=args.semantic_config,
                     options=args.options, aug_test=args.aug_test,
                     checkpoint=args.semantic_checkpoint, eval_options=args.eval_options,
                     color_list_path=args.color_list_path, show_vis=args.show_vis, img_path=args.img_path,
                     output_path=args.out_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
