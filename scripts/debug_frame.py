#!/usr/bin/env python3
"""Debug script to get full traceback from frame generation."""

import sys
import traceback

sys.path.insert(0, "/home/aryan/Documents/repos/artgallery-wallpaper/src")


# Import the frame module
import importlib.util

spec = importlib.util.spec_from_file_location(
    "frame_module",
    "/home/aryan/Documents/repos/artgallery-wallpaper/src/artgallery-wallpaper/frame.py",
)
frame_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_module)

# Run the generation with full error reporting
try:
    frame_module.generate_frame_sd15(
        corner_path="/home/aryan/Documents/repos/artgallery-wallpaper/assets/frames/corners/Cassetta_frame_MET_86K_FRTS5R7.no_bg.png",
        width=1000,
        height=1000,
        device="auto",
        low_memory=True,
        generation_mode="full",
    )
except Exception:
    print("\n" + "=" * 80)
    print("FULL TRACEBACK:")
    print("=" * 80)
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)
