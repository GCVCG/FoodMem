import argparse


def semantic_predict():
    return

parser = argparse.ArgumentParser(
    description='Segmenting FoodSAM for a frame')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/1641173_2291260800.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/tag2text_swin_14m.pth')
parser.add_argument('--specified-tags',
                    default='None',
                    help='User input specified tags')

args = parser.parse_args()


if __name__ == '__main__':
    print(args.image)

# semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test,
# args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.img_path)