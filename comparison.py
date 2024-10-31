import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.widgets as mpwidgets
import argparse


class Visualization:
    def __init__(self, image_parent_dir: str, img1_name: str = 'mr', img2_name: str = 'ct') -> None:
        self.img1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_parent_dir, img1_name + '.nii.gz')))
        self.img2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_parent_dir, img2_name + '.nii.gz')))

    def update_opacity(self, value): 
        self.img2_plot.set_alpha(value)    
        self.fig.canvas.draw_idle()
    
    def update_slice(self, value):
        slice_idx = int(value)
        self.img1_plot.set_data(self.img1[:, slice_idx, :])
        self.img2_plot.set_data(self.img2[:, slice_idx, :])
        self.fig.canvas.draw_idle()

    def overlay_3d_plot(self) -> None:
        OPACITY = 0.5
        SLICE_NUMBER = self.img1.shape[1] // 2

        # PLOT
        self.fig, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [5, 1, 1]})
        self.img1_plot = ax0.imshow(self.img1[:,SLICE_NUMBER,:], cmap="gray")
        self.img2_plot = ax0.imshow(self.img1[:,SLICE_NUMBER,:], alpha=OPACITY, cmap="gray")

        slider0 = mpwidgets.Slider(ax=ax1, label='opacity', valmin=0, valmax=1, valinit=OPACITY)
        slider0.on_changed(self.update_opacity)

        slider1 = mpwidgets.Slider(ax=ax2, label='slice', valmin=0, valmax=self.img1.shape[1]-1, valinit=SLICE_NUMBER)
        slider1.on_changed(self.update_slice)

        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "index",
        type=int,
        help="The index of data; 0-60:A, 61-120:B, 121:180:C"
    )

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Path to the parent folder containing data",
        required=False,
        default='../Task1/brain/'
    )

    parser.add_argument(
        "-i",
        "--image1",
        type=str,
        help="name of the first image to show; no suffix",
        required=False,
        default='mr'
    )

    parser.add_argument(
        "-j",
        "--image2",
        type=str,
        help="name of the second image to show; no suffix",
        required=False,
        default='ct'
    )

    args = parser.parse_args()
    DATA_PARENT_DIR = args.folder
    iterable_dir_list = sorted(os.listdir(DATA_PARENT_DIR), key=lambda x: x)
    dataset_parent_dir = [os.path.join(DATA_PARENT_DIR, mri_path) for mri_path in iterable_dir_list]

    vis = Visualization(dataset_parent_dir[args.index], img1_name=args.image1, img2_name=args.image2)
    vis.overlay_3d_plot()




