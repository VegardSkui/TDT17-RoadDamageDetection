import pathlib
import random
import xml.etree.ElementTree as ET
from typing import List

import tqdm
from PIL import Image


INPUT_DIR = pathlib.Path("./original/Norway/train")
OUTPUT_DIR = pathlib.Path("./derived/train-nitti-ll-1824")

RATIO = 9 / 10

NAMES = ["D00", "D10", "D20", "D40"]


def main():
    if OUTPUT_DIR.exists():
        print("ERROR: Output directory already exists")
        quit(1)

    # Ensure reproducible splits
    random.seed(42)

    annotations = {}
    for xml_path in tqdm.tqdm(INPUT_DIR.joinpath("annotations/xmls").glob("*.xml"), desc="process annotations"):
        tree = ET.parse(xml_path)
        annotation = tree.getroot()

        size = tree.find("size")
        # image_width = float(size.findtext("width"))
        image_height = float(size.findtext("height"))

        boxes = []
        for object in annotation.iter("object"):
            # Skip objects of types were not looking at
            name = object.find("name").text
            if name not in NAMES:
                continue

            # Extract the parameters of the bounding box and crop them to be
            # within the 1824x1824 box in the lower left corner of the image
            # The y-coordinates are adjusted to be relative to the cropped image
            # (we can assume the image to be taller than 1824px)
            bndbox = object.find("bndbox")
            xmin = min(float(bndbox.findtext("xmin")), 1824)
            ymin = max(float(bndbox.findtext("ymin")) - (image_height - 1824), 0)
            xmax = min(float(bndbox.findtext("xmax")), 1824)
            ymax = max(float(bndbox.findtext("ymax")) - (image_height - 1824), 0)

            # Skip the box if the remaining area is zero
            if (xmax - xmin) * (ymax - ymin) == 0:
                continue

            x_center = (xmax + xmin) / 2 / 1824
            y_center = (ymax + ymin) / 2 / 1824
            width = (xmax - xmin) / 1824
            height = (ymax - ymin) / 1824
            boxes.append([NAMES.index(name), x_center, y_center, width, height])

        # Keep the name and annotations if there is at least one object
        name = xml_path.name[:-4]
        if len(boxes) > 0:
            annotations[name] = boxes

    # Get all image names in order and shuffle (must be ordered to ensure
    # reproducible shuffling)
    names = sorted(annotations.keys())
    random.shuffle(names)

    # Create output directories
    OUTPUT_DIR.joinpath("images/train").mkdir(parents=True)
    OUTPUT_DIR.joinpath("images/val").mkdir()
    OUTPUT_DIR.joinpath("labels/train").mkdir(parents=True)
    OUTPUT_DIR.joinpath("labels/val").mkdir()

    train_len = int(len(names) * RATIO)
    for name in tqdm.tqdm(names[:train_len], desc="train"):
        process_image(name, split="train")
        write_annotations(annotations[name], OUTPUT_DIR.joinpath(f"labels/train/{name}.txt"))
    for name in tqdm.tqdm(names[train_len:], desc="val"):
        process_image(name, split="val")
        write_annotations(annotations[name], OUTPUT_DIR.joinpath(f"labels/val/{name}.txt"))


def process_image(name: str, split: str) -> None:
    image = Image.open(INPUT_DIR.joinpath(f"images/{name}.jpg"))

    # Crop the image
    height = image.height
    image = image.crop((0, height - 1824, 1824, height))

    image.save(OUTPUT_DIR.joinpath(f"images/{split}/{name}.jpg"))


def write_annotations(boxes: List[List[float]], path: pathlib.Path) -> None:
    with open(path, "w") as fp:
        for box in boxes:
            fp.write(" ".join(str(item) for item in box) + "\n")


if __name__ == "__main__":
    main()
