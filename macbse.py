#!/usr/bin/env python3

import argparse


import numpy as np
import nibabel as nib
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.nn import MSELoss
from monai.data import Dataset, DataLoader, partition_dataset

from monai.transforms import (
    Compose,
    HistogramNormalizeD,
    ScaleIntensityd,
    LoadImaged,
    ToTensord,
    LoadImage,
    ToTensor,
    EnsureChannelFirstD,
    EnsureChannelFirstd,
    Resized,
    Resize,
)
from monai.utils import set_determinism
from glob import glob
import random
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre


VOXSIZE = 64 #128

def macbse(
    input_filename, output_filename, model_filename, mask_filename, device="cpu"
):

    # Specify spatial_dims and strides for 3D data
    spatial_dims = 3
    strides = (1, 1, 1, 1)

    channels = (16, 64, 64, 128, 256);#(2, 8, 8, 16, 32) #(16, 64, 64, 128, 256) #(2, 8, 8, 16, 32)
    #  #tuple(map(int, last_subdirectory.split('_')[1][1:-1].split(',')))

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,  # Adjust based on your data
        out_channels=1,  # Adjust based on your data
        channels=channels,  # (16, 64, 64, 128, 256),#(2,8,8,16,32),#(16, 64, 64, 128, 256),
        strides=strides,
    ).to(device)

    keys = ["image"]

    test_transforms = Compose(
        [
            LoadImaged(keys, image_only=True),
            EnsureChannelFirstd(keys),
            ScaleIntensityd(keys="image", minv=0.0, maxv=255.0),
            # the Unet has instance normalization, so this scaling won't make any difference.
            # But we keep it for now in case we choose to change the network.
            Resized(
                keys,
                spatial_size=(VOXSIZE, VOXSIZE, VOXSIZE),
                mode="trilinear",
            ),
        ]
    )

    model.eval()

    # Load the test image (adjust the path to your validation image)
    test_image_path = input_filename
    test_bse_image_path = output_filename
    model_file = model_filename

    model.load_state_dict(
        torch.load(model_file, map_location=torch.device(device))
    )

    test_dict = [{"image": test_image_path}]
    # test_dict = [{"image": image, "mask": mask} for image, mask in zip(image_files, mask_files)]
    # Apply transformations to the validation image
    test_image = test_transforms(test_dict)[0]["image"].to(device)

    # Apply the trained model to estimate the mask
    with torch.no_grad():
        print(test_image.max(), test_image.min())
        estimated_mask = model(test_image[None,])

    # Convert the estimated mask to a Numpy array
    estimated_mask = estimated_mask.squeeze().cpu().numpy()

    # Load the original validation image without resizing (for displaying the corrected image)
    original_test_image = nib.load(test_image_path).get_fdata().squeeze()

    orig_shape = original_test_image.shape

    estimated_mask = Resize(mode="trilinear", spatial_size=orig_shape)(
        estimated_mask[None,]
    )[0]

    # Apply the estimated mask to remove the skull from the original image
    bse_image = torch.tensor(original_test_image) * (estimated_mask > 0.5)

    input_nifti = nib.load(test_image_path)
    input_dtype = input_nifti.get_data_dtype()
    bse_image = bse_image.numpy().astype(np.single)

    # Create a new NIfTI image with the result data
    result_nifti = nib.Nifti1Image(bse_image, input_nifti.affine)

    # Save the result as a new NIfTI image
    nib.save(result_nifti, output_filename)

    # Save the mask
    if mask_filename is not None:
        mask_nifti = nib.Nifti1Image(
            255*(estimated_mask>0.5).numpy().astype(np.uint8), input_nifti.affine
        )
        nib.save(mask_nifti, mask_filename)


def main():
    parser = argparse.ArgumentParser(
        description="Command-line tool for processing input and output filenames."
    )
    parser.add_argument(
        "-i", "--input", help="Input filename (MRI image filename)", required=True
    )
    parser.add_argument(
        "-m", "--model", help="Model file (Trained model .pth file)", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (Skull extracted image filename)",
        required=True,
    )
    parser.add_argument("--mask", help="Brain Mask filename", required=False)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output
    model_filename = args.model
    mask_filename = args.mask

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = "cpu"

    # Use the selected device for computation
    device = torch.device(args.device)

    # Define the UNet model and optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    macbse(input_filename, output_filename, model_filename, mask_filename, device)


if __name__ == "__main__":
    main()
