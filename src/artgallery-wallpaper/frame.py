import cv2
import numpy as np
from pathlib import Path


def remove_background_opencv(
    image_path: str,
    tolerance: int = 20,
    cleanup_kernel_size: int = 5,
    output_path: str = None,
):
    """Remove the background from an image using OpenCV.

    Args:
        image_path (str): Path to the input image(s).
        tolerance (int, optional): Tolerance level for background removal. Defaults to 20.
        cleanup_kernel_size (int, optional): Kernel size for morphological cleanup. Defaults to 5.
        output_path (str, optional): Path to save the output image. Defaults to None.

    Raises:
        FileNotFoundError: If the input image is not found.
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
            return

        img_fill = img_bgr.copy()
        height, width, _ = img_fill.shape
        mask = np.zeros((height + 2, width + 2), np.uint8)
        fill_color_bgr = (255, 0, 255)
        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

        for x, y in corners:
            if not np.array_equal(img_fill[y, x], fill_color_bgr):
                cv2.floodFill(
                    img_fill,
                    mask,
                    (x, y),
                    fill_color_bgr,
                    loDiff=(tolerance,) * 3,
                    upDiff=(tolerance,) * 3,
                )

        background_mask = np.all(img_fill == fill_color_bgr, axis=2)
        alpha_channel = np.where(background_mask, 0, 255).astype(np.uint8)

        # --- NEW: Cleanup step with Morphological Opening ---
        if cleanup_kernel_size > 0:
            kernel = np.ones((cleanup_kernel_size, cleanup_kernel_size), np.uint8)
            alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_OPEN, kernel)
        # ---------------------------------------------------

        final_img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        final_img_bgra[:, :, 3] = alpha_channel

        default_output_path = Path(image_file).with_suffix(".no_bg.png")
        if output_path is None:
            output_path = default_output_path
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / default_output_path.name
            elif output_path.is_file() and len(image_files) > 1:
                output_path = output_path.parent / default_output_path.name

            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as error:
                print(f"Error creating directories for {output_path}: {error}")
                print("Falling back to the default output path.")
                output_path = default_output_path

        cv2.imwrite(str(output_path), final_img_bgra)
        print(f"Image saved to {output_path}")
