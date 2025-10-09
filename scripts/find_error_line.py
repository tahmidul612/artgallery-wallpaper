#!/usr/bin/env python3
"""Script to find the exact line where top_mask error occurs."""

import sys
import traceback

# Patch the frame module to add detailed error reporting
sys.path.insert(0, "/home/aryan/Documents/repos/artgallery-wallpaper/src")

# Run with detailed traceback
if __name__ == "__main__":
    try:
        # Import after path is set
        import importlib.util

        # Load the module
        spec = importlib.util.spec_from_file_location(
            "frame",
            "/home/aryan/Documents/repos/artgallery-wallpaper/src/artgallery-wallpaper/frame.py",
        )
        frame = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frame)

        # Try to generate
        frame.generate_frame_sd15(
            corner_path="/home/aryan/Documents/repos/artgallery-wallpaper/assets/frames/corners/Cassetta_frame_MET_86K_FRTS5R7.no_bg.png",
            width=1000,
            height=1000,
            device="auto",
            low_memory=True,
            generation_mode="full",
        )
        print("\n✅ SUCCESS! Frame generated without errors.")
    except UnboundLocalError as e:
        print(f"\n❌ UnboundLocalError: {e}")
        print("\n" + "=" * 80)
        print("DETAILED TRACEBACK:")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)

        # Print locals and line info
        import sys

        tb = sys.exc_info()[2]
        frame_obj = tb.tb_frame
        print(f"\nError occurred in: {frame_obj.f_code.co_filename}")
        print(f"Function: {frame_obj.f_code.co_name}")
        print(f"Line: {tb.tb_lineno}")
        print("\nLocal variables at error:")
        for key, value in list(frame_obj.f_locals.items())[:20]:  # First 20 vars
            print(f"  {key}: {type(value).__name__}")
    except Exception as e:
        print(f"\n❌ Other error: {type(e).__name__}: {e}")
        traceback.print_exc()
