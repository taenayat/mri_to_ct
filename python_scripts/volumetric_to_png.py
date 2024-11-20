
import argparse
import logging

import SimpleITK as sitk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmapy

def compare_image(image_a, image_a_filename, image_b, image_b_filename, output_compare):
    fill_image = np.zeros_like(image_a)
    comp_image = cv2.merge((image_b, fill_image, image_a))

    filename_a = os.path.basename(image_a_filename).split(".")[0]
    filename_b = os.path.basename(image_b_filename).split(".")[0]

    if output_compare is None:
        output_dir = os.path.dirname(image_a_filename)
    else:
        output_dir = output_compare
        
    filename = "compare_"+filename_a+"_"+filename_b+".png"
    cv2.imwrite(os.path.join(output_dir,filename), comp_image)

def img_to_heatmap(img, color_mode):
    if color_mode == 1:
        heatmap = cv2.applyColorMap(img, cmapy.cmap('bwr'))
    elif color_mode == 2:
        heatmap = cv2.applyColorMap(img, cmapy.cmap('jet'))
    else:
        raise RuntimeError("Error: color mode {} not recognized".format(color_mode))

    return heatmap

def process_image(image_path, n_partitions, is_seg, color_mode, clipping_window):

    image = sitk.ReadImage(image_path)
    img_size = image.GetSize()

    image_np = sitk.GetArrayFromImage(image)
    
    if color_mode != 0:
        image_np = np.clip(image_np, -clipping_window//2, clipping_window//2)

    if not is_seg:
        if color_mode == 0:
            img_min = np.min(image_np)
            img_max = np.max(image_np)
        if color_mode == 1:
            img_min = -clipping_window//2 #image_np.min()
            img_max = clipping_window//2 #image_np.max()
        elif color_mode == 2:
            img_min = 0 #image_np.min()
            img_max = clipping_window//2 #image_np.max()

        image_np -= img_min
        image_np /= (img_max - img_min)
    
    image_np = image_np*255
    image_np = image_np.astype(np.uint8)

    img_to_show = [[], [], []]
    for axis in [0, 1, 2]:

        slices = np.linspace(0, img_size[axis], num=n_partitions+1, endpoint=False).astype(np.uint16)[1:]
        for slice in slices:
            if axis == 0:
                img = image_np[:,:,slice]
                if color_mode != 0:
                    img = img_to_heatmap(img, color_mode)
                img = cv2.flip(img, 0)
                img_to_show[0].append(img)
            elif axis == 1:
                img = image_np[:,slice,:]
                if color_mode != 0:
                    img = img_to_heatmap(img, color_mode)
                img = cv2.flip(img, 0)
                img_to_show[1].append(img)
            elif axis == 2:
                img = image_np[slice,:,:]
                if color_mode != 0:
                    img = img_to_heatmap(img, color_mode)
                img_to_show[2].append(img)
    
    axis_images = []
    for image_axis in img_to_show:
        img_concat = cv2.vconcat(image_axis)
        axis_images.append(img_concat)

    max_height = max(img.shape[0] for img in axis_images)
    
    padded_images = []
    for img in axis_images:
        height, width = img.shape[:2]
        top = 0
        bottom = max_height - height
        padded_img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_images.append(padded_img)
        
    # Concatenate images horizontally
    hconcat_image = cv2.hconcat(padded_images)

    filename = os.path.abspath(image_path).split(".")[0]

    cv2.imwrite(filename+".png", hconcat_image)

    return hconcat_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Image to render in png",
        required=False
    )

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Folder containing all the images to render in png",
        required=False
    )

    parser.add_argument(
        "-c",
        "--compare",
        type=str,
        help="Image to render in the same png as the --image argument",
        required=False
    )

    parser.add_argument(
        "-o",
        "--output-compare",
        type=str,
        help="Path to a directory to save the resulting comparison image. If not provided, the image will be saved in the same directory where the --image argument is located",
        required=False
    )

    parser.add_argument(
        "-n",
        "--n-partitions",
        type=int,
        help="Number of slices to render per axis",
        required=False,
        default=1
    )

    parser.add_argument(
        "-cw",
        "--clipping-window",
        type=int,
        help="Double the value of the clipping window. The intensity values of the resulting 2D images will be clipped between -clipping_window//2 and clipping_window//2",
        required=False,
        default=2000
    )


    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        help="Verbosity level. 0: WARNING, 1: INFO, 2: DEBUG",
        default=0
    )

    parser.add_argument(
        "--seg",
        action="store_true",
        help="Indicates whether the image is a segmentation mask or not",
        required=False,
        default=False
    )

    parser.add_argument(
        "--color_mode",
        type=int,
        help="Indicates the color map to use for rendering. 0: original color (HU), 1: blue-white-red, 2: jet",
        required=False,
        default=0
    )

    args = parser.parse_args()

    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    assert args.image is not None or args.folder is not None

    if args.folder is not None:
        for root, folders, files in os.walk(args.folder):
            for file in files:
                if file.endswith(".nii.gz"):
                    process_image(os.path.join(root, file), args.n_partitions, args.seg, args.color_mode, args.clipping_window)
    
    elif args.compare is not None:
        image_a = process_image(args.image, args.n_partitions, args.seg, args.color_mode, args.clipping_window)
        image_b = process_image(args.compare, args.n_partitions, args.seg, args.color_mode, args.clipping_window)

        if image_a.shape != image_b.shape:
            raise RuntimeError("Error, both images must have same shape")

        compare_image(image_a, args.image, image_b, args.compare, args.output_compare)

    elif args.image is not None:
        process_image(args.image, args.n_partitions, args.seg, args.color_mode, args.clipping_window)