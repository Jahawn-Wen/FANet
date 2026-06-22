import argparse
import os
from shutil import copyfile


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the CVUSA dataset layout.")
    parser.add_argument(
        "--data_dir",
        default=os.path.join("data", "CVUSA"),
        help="Path to the CVUSA root directory.",
    )
    return parser.parse_args()


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
            copyfile(satellite_src, os.path.join(satellite_dst, os.path.basename(filename[0])))

            street_src = os.path.join(data_dir, filename[1])
            street_dst = os.path.join(street_dir, os.path.basename(filename[1][:-4]))
            os.makedirs(street_dst, exist_ok=True)
            copyfile(street_src, os.path.join(street_dst, os.path.basename(filename[1])))


def main():
    args = parse_args()
    prepare_split(args.data_dir, "train", "train")
    prepare_split(args.data_dir, "val", "val")


if __name__ == "__main__":
    main()
