import pathlib

import tqdm
from PIL import Image


OUTPUT_DIR = pathlib.Path("./derived/test-ll-1856")


def main():
    if OUTPUT_DIR.exists():
        print("ERROR: Output directory already exists")
        quit(1)
    OUTPUT_DIR.mkdir()

    # Crop test images to lower left 1856x1856 square
    for path in tqdm.tqdm(list(pathlib.Path("./original/Norway/test/images").glob("*.jpg"))):
        image = Image.open(path)
        height = image.height
        image = image.crop((0, height - 1856, 1856, height))
        image.save(OUTPUT_DIR.joinpath(path.name))


if __name__ == "__main__":
    main()
