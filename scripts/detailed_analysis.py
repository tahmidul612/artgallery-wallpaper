#!/usr/bin/env python3
"""Detailed analysis of what's being generated."""

from PIL import Image
import numpy as np
from pathlib import Path

latest = Path("frame_output_42.png")
img = Image.open(latest).convert("RGBA")
arr = np.array(img)

h, w = arr.shape[:2]
print(f"Image size: {w}x{h}")

# Check the quadrant that was generated
top_left_quadrant = arr[: h // 2, : w // 2]

# Sample some points to see what was generated
print("\nSampling pixel values in top-left quadrant:")
print(f"Corner (50, 50): {arr[50, 50]}")
print(f"Top edge middle (50, 200): {arr[50, 200]}")
print(f"Left edge middle (200, 50): {arr[200, 50]}")
print(f"Quadrant center (200, 200): {arr[200, 200]}")

# Check if there's variation in the generated area
# The corner should be ornate (varied), the generated edges should have some pattern
center_region = arr[150:250, 150:250]  # 100x100 region in quadrant center
std_dev = np.std(center_region[:, :, :3], axis=(0, 1))
print(f"\nColor variation in quadrant center (RGB std dev): {std_dev}")
print("(Low values = mostly uniform color, high values = detailed texture)")

# Check what percentage is "white-ish" (>240)
white_pixels = np.sum(np.all(arr[:, :, :3] > 240, axis=2))
total_pixels = w * h
print(
    f"\nWhite-ish pixels: {white_pixels}/{total_pixels} ({100 * white_pixels / total_pixels:.1f}%)"
)

# Visual check: save the top-left quadrant only
quadrant_img = Image.fromarray(top_left_quadrant)
quadrant_img.save("debug_quadrant.png")
print("\nSaved top-left quadrant to debug_quadrant.png for inspection")
