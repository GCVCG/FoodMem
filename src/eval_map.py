import argparse
import os
import numpy as np
from sklearn.metrics import average_precision_score, recall_score
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import time
import cv2

def load_masks(directory):
    masks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            mask = imread(os.path.join(directory, filename))
            # Verificar si la imagen ya está en formato RGB
            if mask.shape[-1] == 3:
                # Convertir píxeles etiquetados como 1 a 255
                mask[mask == 1] = 255
                masks[filename] = mask
            else:
                # Convertir la imagen en escala de grises a RGB
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                # Convertir píxeles etiquetados como 1 a 255
                mask_rgb[mask_rgb == 1] = 255
                masks[filename] = mask_rgb
    return masks

def generate_error_image(submit_mask, truth_mask):
    # False negative
    error_mask = np.zeros_like(submit_mask)
    error_indices = ((truth_mask == 255) & (submit_mask == 0))
    error_mask[error_indices] = 255  # Red color
    error_mask[:, :, 1] = 0
    error_mask[:, :, 2] = 0
    overlaid_image = submit_mask.copy()
    overlaid_image[error_mask[:, :, 0] == 255] = [255, 0, 0]

    # False positive
    error_mask = np.zeros_like(submit_mask)
    error_indices = ((truth_mask == 0) & (submit_mask != 0))
    error_mask[error_indices] = 255
    error_mask[:, :, 1] = 0
    error_mask[:, :, 2] = 0
    overlaid_image[error_mask[:, :, 0] == 255] = [0, 0, 255]

    return overlaid_image


def mean_average_precision(submit_dir, truth_dir, output_dir, show_error):
    submit_masks = load_masks(submit_dir)
    truth_masks = load_masks(truth_dir)

    os.makedirs(output_dir, exist_ok=True)
    if len(submit_masks) != len(truth_masks):
        raise ValueError("El número de máscaras en submit_dir y truth_dir no coincide.")

    average_precisions = {}
    for filename in submit_masks:
        submit_mask = submit_masks[filename].flatten()
        truth_mask = truth_masks.get(filename, np.zeros_like(submit_mask)).flatten()
        average_precision = average_precision_score(truth_mask, submit_mask, pos_label=255)
        average_precisions[filename] = average_precision

        if show_error:
            error_image = generate_error_image(submit_masks[filename], truth_masks.get(filename, np.zeros_like(submit_masks[filename])))
            error_masks_dir = os.path.join(output_dir, "error_masks")
            os.makedirs(error_masks_dir, exist_ok=True)
            error_image_path = os.path.join(error_masks_dir, filename)
            imsave(error_image_path, error_image)

    mean_average_precision = np.mean(list(average_precisions.values()))

    output_file = os.path.join(output_dir, "mean_average_precision.txt")
    with open(output_file, "w") as f:
        for filename, ap in average_precisions.items():
            f.write("{}: {:.4f}\n".format(filename, ap))
        f.write("Mean Average Precision: {:.4f}\n".format(mean_average_precision))


def calculate_recall(submit_dir, truth_dir, output_dir):
    submit_masks = load_masks(submit_dir)
    truth_masks = load_masks(truth_dir)
    recalls = {}
    for filename in submit_masks:
        submit_mask = submit_masks[filename]
        truth_mask = truth_masks.get(filename, np.zeros_like(submit_mask))
        recall = recall_score(truth_mask.flatten(), submit_mask.flatten(), average='micro')
        recalls[filename] = recall

    mean_recall = np.mean(list(recalls.values()))
    output_file = os.path.join(output_dir, "recall_score.txt")
    with open(output_file, "w") as f:
        for filename, recall in recalls.items():
            f.write("{}: {:.4f}\n".format(filename, recall))
        f.write("Recall: {:.4f}\n".format(mean_recall))


def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')

    parser.add_argument('--submit_dir', '-i', type=str, help='test output directory', required=True)

    parser.add_argument(
        '--truth_dir',
        type=str,
        default='../VIPSeg/VIPSeg_720P/panomasksRGB',
        help='ground truth directory. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panomasksRGB '
             'after running the conversion script')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output',
        help='Output directory'
    )

    parser.add_argument(
        '--show_error',
        default=False,
        action="store_true",
        help='Show error mask'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = args.output_dir
    start_time = time.time()
    mean_average_precision(submit_dir, truth_dir, output_dir, args.show_error)
    end_time = time.time()
    print("Mean Average Precision execution time:", end_time - start_time, "seconds")
    start_time = time.time()
    calculate_recall(submit_dir, truth_dir, output_dir)
    end_time = time.time()
    print("Recall Score execution time:", end_time - start_time, "seconds")
