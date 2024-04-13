import os
import cv2
import argparse
import numpy as np


def visualize(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of input images
    input_images = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith((".jpg", ".png"))
    ]

    # Iterate over the input images
    for input_image in input_images:
        # Read the input image
        image = cv2.imread(input_image)

        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(input_image))[0]

        # Create a canvas for visualization
        canvas = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)

        # Place the input image on the left side of the canvas
        canvas[:, : image.shape[1], :] = image

        # Read the corresponding colorized image
        colorized_path = os.path.join(output_dir, f"{filename}.png")
        if os.path.exists(colorized_path):
            colorized_image = cv2.imread(colorized_path)
            canvas[:, image.shape[1] :, :] = colorized_image

        # Display the canvas
        cv2.imshow("Colorized Image", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize colorized images")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the input image directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output image directory",
    )
    args = parser.parse_args()

    visualize(args.input_dir, args.output_dir)
