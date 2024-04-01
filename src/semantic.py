import argparse

from mmcv.utils import DictAction

from FoodSAM_tools.predict_semantic_mask import semantic_predict

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
    "--output",
    type=str,
    default='Output/Semantic_Results',
    help="Path to the directory where results will be output. Output will be a folder "
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


def main(args: argparse.Namespace) -> None:
    semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test,
                     args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.img_path)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.img_path)
    print(args.output)
    main(args)

# semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test,
# args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.img_path)
