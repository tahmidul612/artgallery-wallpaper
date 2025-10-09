"""
Background removal utilities using HSV color space masking.

This module provides functionality to remove grayscale backgrounds from images,
which is particularly useful for isolating colorful artwork or frame elements
from white, gray, or black backgrounds.
"""

from pathlib import Path
from typing import List
from typing_extensions import Annotated
import cv2
import numpy as np
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print

app = typer.Typer(help="Background removal utilities using HSV color space masking.")


@app.command()
def remove_background(
    image_path: Annotated[
        str, typer.Option(help="Path to the input image or directory", prompt=True)
    ],
    saturation_threshold: Annotated[
        int,
        typer.Option(help="Max saturation for a pixel to be considered background"),
    ] = 40,
    open_kernel_size: Annotated[
        int,
        typer.Option(help="Kernel size for morphological opening to remove noise"),
    ] = 3,
    close_kernel_size: Annotated[
        int,
        typer.Option(help="Kernel size for morphological closing to fill holes"),
    ] = 7,
    output_path: Annotated[
        str, typer.Option(help="Path to save the output file or directory")
    ] = None,
    batch: Annotated[
        bool, typer.Option(help="Whether to process images in batch mode")
    ] = False,
) -> List[Path]:
    """Removes a grayscale background using HSV color space masking.

    This method is highly effective at separating a colorful foreground from a
    grayscale (white, gray, black) background, even with shadows and gradients.

    Args:
        image_path: Path to the input image or a directory of images
        saturation_threshold: Max saturation for a pixel to be considered background
        open_kernel_size: Kernel size for morphological opening to remove noise
        close_kernel_size: Kernel size for morphological closing to fill holes
        output_path: Path to save the output file or directory
        batch: Whether to process directory as batch

    Returns:
        List of paths to processed images
    """
    image_path = Path(image_path)
    processed_files = []

    if image_path.is_dir() or batch:
        IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

        if image_path.is_dir():
            search_path = image_path
        else:
            search_path = image_path.parent

        image_files = [
            p
            for p in search_path.glob("*.*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXT
            and not any("no_bg" in suf for suf in p.suffixes)
        ]

        if not image_files:
            print(f"[yellow]No valid image files found in {search_path}[/yellow]")
            return []

        print(f"[cyan]Found {len(image_files)} images to process[/cyan]")
    else:
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image file/folder not found: {image_path}")
            image_files = [image_path]
        except FileNotFoundError as e:
            print(f"[red]{e}[/red]")
            return []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(image_files)} image(s)...", total=len(image_files)
        )

        for idx, image_file in enumerate(image_files, 1):
            try:
                progress.update(
                    task,
                    description=f"Processing [{idx}/{len(image_files)}]: {image_file.name}",
                )

                img_bgr = cv2.imread(str(image_file))
                if img_bgr is None:
                    print(f"[red]Error: Could not read image {image_file}[/red]")
                    continue

                # Convert BGR to HSV
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                # Create mask for grayscale background
                lower_bound = np.array([0, 0, 0])
                upper_bound = np.array([179, saturation_threshold, 255])

                background_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
                foreground_mask = cv2.bitwise_not(background_mask)

                # Morphological operations for cleanup
                if open_kernel_size > 0:
                    open_kernel = np.ones(
                        (open_kernel_size, open_kernel_size), np.uint8
                    )
                    foreground_mask = cv2.morphologyEx(
                        foreground_mask, cv2.MORPH_OPEN, open_kernel
                    )

                if close_kernel_size > 0:
                    close_kernel = np.ones(
                        (close_kernel_size, close_kernel_size), np.uint8
                    )
                    foreground_mask = cv2.morphologyEx(
                        foreground_mask, cv2.MORPH_CLOSE, close_kernel
                    )

                # Apply alpha channel
                final_img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
                final_img_bgra[:, :, 3] = foreground_mask

                # Determine output path
                default_output_name = image_file.stem + ".no_bg.png"

                if output_path is None:
                    current_output_path = image_file.parent / default_output_name
                else:
                    output_path_p = Path(output_path)
                    if output_path_p.is_dir():
                        current_output_path = output_path_p / default_output_name
                    else:
                        if len(image_files) > 1:
                            current_output_path = (
                                output_path_p.parent / default_output_name
                            )
                        else:
                            current_output_path = output_path_p

                # Save the result
                current_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(current_output_path), final_img_bgra)
                processed_files.append(current_output_path)

                progress.advance(task)

            except Exception as error:
                print(f"[red]Error processing {image_file}: {error}[/red]")
                continue

    if processed_files:
        print(
            f"[green]✓ Successfully processed {len(processed_files)} image(s)[/green]"
        )
        for f in processed_files:
            print(f"  • {f}")

    return processed_files


if __name__ == "__main__":
    app()
