import argparse
import os
from shutil import copyfile

import imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CVUSA with polar-transformed satellite images.")
    parser.add_argument(
        "--data_dir",
        default=os.path.join("data", "CVUSA"),
        help="Path to the CVUSA root directory.",
    )
    return parser.parse_args()


def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    fx1 = (ix1 - rx)[..., na] * signal_00 + (rx - ix0)[..., na] * signal_10
    fx2 = (ix1 - rx)[..., na] * signal_01 + (rx - ix0)[..., na] * signal_11

    return (iy1 - ry)[..., na] * fx1 + (ry - iy0)[..., na] * fx2


def apply_aerial_polar_transform(src_path, dst_path, imgname):
    source_size = 750
    height = 112
    width = 616

    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i)

    y = source_size / 2.0 - source_size / 2.0 / height * (height - 1 - ii) * np.sin(2 * np.pi * jj / width)
    x = source_size / 2.0 + source_size / 2.0 / height * (height - 1 - ii) * np.cos(2 * np.pi * jj / width)

    signal = imageio.imread(src_path)
    image = sample_bilinear(signal, x, y)
    imageio.imsave(os.path.join(dst_path, imgname), image)


def prepare_split(data_dir, split_name, output_name):
    split_file = os.path.join(data_dir, "splits", f"{split_name}-19zl.csv")
    output_dir = os.path.join(data_dir, output_name)
    street_dir = os.path.join(output_dir, "street")
    satellite_dir = os.path.join(output_dir, "satellite")

    os.makedirs(street_dir, exist_ok=True)
    os.makedirs(satellite_dir, exist_ok=True)

    with open(split_file) as fp:
        for line in fp:
            filename = line.strip().split(",")
            if len(filename) < 2:
                continue

            satellite_src = os.path.join(data_dir, filename[0])
            satellite_dst = os.path.join(satellite_dir, os.path.basename(filename[0][:-4]))
            os.makedirs(satellite_dst, exist_ok=True)
            apply_aerial_polar_transform(satellite_src, satellite_dst, os.path.basename(filename[0]))

            street_src = os.path.join(data_dir, filename[1])
            street_dst = os.path.join(street_dir, os.path.basename(filename[1][:-4]))
            os.makedirs(street_dst, exist_ok=True)
            copyfile(street_src, os.path.join(street_dst, os.path.basename(filename[1])))


def main():
    args = parse_args()
    prepare_split(args.data_dir, "train", "train_pt")
    prepare_split(args.data_dir, "val", "val_pt")


if __name__ == "__main__":
    main()
