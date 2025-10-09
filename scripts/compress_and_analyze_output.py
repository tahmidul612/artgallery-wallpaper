#!/usr/bin/env python3
"""Compress output image and analyze what's being generated."""

from PIL import Image
import numpy as np
from pathlib import Path

# Find most recent frame output
output_files = sorted(Path(".").glob("frame_output_*.png"))
if not output_files:
    print("No frame output files found!")
    exit(1)

latest = output_files[-1]
print(f"Analyzing: {latest}")

# Load and analyze
img = Image.open(latest)
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")

# Check if there's any variation in the image (not just mirrored corners)
arr = np.array(img.convert("RGB"))
h, w = arr.shape[:2]

# Compare quadrants
top_left = arr[: h // 2, : w // 2]
top_right = arr[: h // 2, w // 2 :]
bottom_left = arr[h // 2 :, : w // 2]
bottom_right = arr[h // 2 :, w // 2 :]

# Check if top_right is just mirrored top_left
top_right_flipped = np.flip(top_right, axis=1)
if np.allclose(top_left, top_right_flipped, atol=5):
    print("✓ Top quadrants are mirrored (expected)")
else:
    print("✗ Top quadrants are NOT perfectly mirrored")

# Check transparency - see if there are gaps
if img.mode == "RGBA":
    alpha = np.array(img)[:, :, 3]
    transparent_pixels = np.sum(alpha < 128)
    total_pixels = alpha.size
    print(
        f"Transparent pixels: {transparent_pixels}/{total_pixels} ({100 * transparent_pixels / total_pixels:.1f}%)"
    )

    # Check if center has content
    center_alpha = alpha[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_transparent = np.sum(center_alpha < 128)
    print(
        f"Center region transparent: {100 * center_transparent / center_alpha.size:.1f}%"
    )

# Compress for viewing
max_size = 1024
if max(img.size) > max_size:
    scale = max_size / max(img.size)
    new_size = (int(img.width * scale), int(img.height * scale))
    compressed = img.resize(new_size, Image.Resampling.LANCZOS)
else:
    compressed = img

# Save with significant compression
output_path = Path("compressed_frame_analysis.jpg")
compressed.convert("RGB").save(output_path, "JPEG", quality=50)
print(
    f"\nCompressed image saved to: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)"
)

# Show corner size vs frame
print(f"\nFrame dimensions: {w}x{h}")
print("If you see mostly transparent areas, the corner image is very small")
print("and the AI should be filling in the rest.")
