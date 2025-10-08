import cv2
import numpy as np
from pathlib import Path


def remove_background(
    image_path: str,
    saturation_threshold: int = 40,
    open_kernel_size: int = 3,
    close_kernel_size: int = 7,
    output_path: str = None,
):
    """Removes a grayscale background using HSV color space masking.

    This method is highly effective at separating a colorful foreground from a
    grayscale (white, gray, black) background, even with shadows and gradients.

    Args:
        image_path (str): Path to the input image or a directory of images.
        saturation_threshold (int, optional): Max saturation for a pixel to be
            considered background. Lower is stricter. Defaults to 40.
        open_kernel_size (int, optional): Kernel size for morphological opening
            to remove noise around the object. Defaults to 3.
        close_kernel_size (int, optional): Kernel size for morphological closing
            to fill holes within the object. Defaults to 7.
        output_path (str, optional): Path to save the output file or directory.
            Defaults to None.
    """

    if Path(image_path).is_dir():
        IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
        image_files = [
            p
            for p in Path(image_path).glob("*.*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXT
            and not any("no_bg" in suf for suf in p.suffixes)
        ]
    else:
        image_files = [Path(image_path)]

    for image_file in image_files:
        try:
            img_bgr = cv2.imread(str(image_file))
            if img_bgr is None:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"Error: The file at {image_file} was not found.")
            continue  # Use continue to proceed with other files in a batch

        # --- Start of Integrated HSV Masking Logic ---

        # Convert the image from BGR to HSV color space
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Define the range for the grayscale background based on saturation
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([179, saturation_threshold, 255])

        # Create a mask that selects the background, then invert it for the foreground
        background_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        foreground_mask = cv2.bitwise_not(background_mask)

        # --- Two-Stage Cleanup ---
        # 1. Opening: Remove small noise around the frame.
        if open_kernel_size > 0:
            open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_OPEN, open_kernel
            )

        # 2. Closing: Fill small holes within the frame.
        if close_kernel_size > 0:
            close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
            foreground_mask = cv2.morphologyEx(
                foreground_mask, cv2.MORPH_CLOSE, close_kernel
            )

        alpha_channel = foreground_mask

        # --- End of Integrated Logic ---

        final_img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        final_img_bgra[:, :, 3] = alpha_channel

        # Determine the final output path
        current_output_path = None
        default_output_path = Path(image_file).with_suffix(".no_bg.png")
        if output_path is None:
            current_output_path = default_output_path
        else:
            output_path_p = Path(output_path)
            if output_path_p.is_dir():
                current_output_path = output_path_p / default_output_path.name
            else:  # Assumes output_path is a file path
                if len(image_files) > 1:
                    # For batches, save relative to the specified output file's directory
                    current_output_path = (
                        output_path_p.parent / default_output_path.name
                    )
                else:
                    # For a single file, use the specified output path directly
                    current_output_path = output_path_p

        try:
            current_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(current_output_path), final_img_bgra)
            print(f"Image saved to {current_output_path}")
        except Exception as error:
            print(f"Error saving file to {current_output_path}: {error}")


# if __name__ == "__main__":
#     # Example usage for a single, difficult image:
#     remove_background(
#         image_path="../../assets/frames",
#         saturation_threshold=50,
#         open_kernel_size=5,
#         close_kernel_size=16,
#     )

#     # Example usage for a directory of images:
#     remove_background(
#         image_path="../../assets/frames/",
#         output_path="../../assets/frames/processed/"
#     )
