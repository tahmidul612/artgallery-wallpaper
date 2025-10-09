#!/usr/bin/env python3
"""
CUDA Out of Memory Fix Script
Helps diagnose and fix GPU memory issues for Stable Diffusion generation.
"""

import torch
import subprocess
import gc
import sys
from pathlib import Path
import warnings


def clear_gpu_memory():
    """Clear GPU memory cache."""
    print("🧹 Clearing GPU memory...")

    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

        # Clear any remaining tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except Exception as e:
                print(f"⚠ Error clearing tensor: {e}")
                pass

        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
    else:
        print("⚠ No CUDA GPU available")


def check_gpu_status():
    """Check current GPU status."""
    print("\n📊 GPU Status:")

    if not torch.cuda.is_available():
        print("  No CUDA GPU detected")
        return False

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total_memory - reserved

        print(f"\n  GPU {i}: {props.name}")
        print(f"  Total VRAM: {total_memory:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Free: {free:.1f} GB")

        # Recommendations
        if free < 4:
            print("  ⚠️  Low free memory! Recommended to use --low-memory flag")
        elif free < 8:
            print("  ⚠️  Moderate free memory. May need --low-memory for large images")
        else:
            print("  ✅ Good amount of free memory")

    return True


def kill_gpu_processes():
    """Kill processes using GPU (with user confirmation)."""
    try:
        # Try nvidia-smi to find processes
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            print("\n🔍 Found processes using GPU:")
            print(result.stdout)

            response = input("Kill these processes? (y/n): ").strip().lower()
            if response == "y":
                pids = [
                    line.split(",")[0].strip()
                    for line in result.stdout.strip().split("\n")
                ]
                for pid in pids:
                    try:
                        if sys.platform == "win32":
                            subprocess.run(["taskkill", "/F", "/PID", pid], check=False)
                        else:
                            subprocess.run(["kill", "-9", pid], check=False)
                    except Exception as e:
                        print(f"⚠ Error killing process {pid}: {e}")
                        pass
                print("✓ Processes killed")
                return True
        else:
            print("✓ No other processes using GPU")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found - can't check for GPU processes")

    return False


def test_memory_usage():
    """Test actual memory usage with a small model."""
    print("\n🧪 Testing memory requirements...")

    try:
        # Clear memory first
        torch.cuda.empty_cache()
        initial = torch.cuda.memory_reserved(0) / 1024**3

        print("  Loading test model...")
        from diffusers import DiffusionPipeline

        # Load a tiny model for testing
        pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")

        after = torch.cuda.memory_reserved(0) / 1024**3
        used = after - initial

        print(f"  ✓ Test model uses ~{used:.1f} GB")
        print("  Full SD 1.5 will need ~4-5 GB minimum")

        # Clean up
        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ⚠ Could not test: {e}")


def show_recommendations():
    """Show recommendations based on GPU status."""
    print("\n💡 Recommendations for your corner image:\n")

    if not torch.cuda.is_available():
        print("Since you don't have CUDA, use CPU mode:")
        print("  python frame.py frame corner.png --device cpu")
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3

    print(f"Based on your {props.name} with {total_gb:.1f}GB VRAM:\n")

    if total_gb < 6:
        print("🔴 Limited VRAM - Use maximum optimization:")
        print(
            "  python frame.py frame corner.png --low-memory --width 512 --height 512"
        )
        print("\nOr use CPU (slow but reliable):")
        print("  python frame.py frame corner.png --device cpu")

    elif total_gb < 8:
        print("🟡 Moderate VRAM - Use low memory mode:")
        print(
            "  python frame.py frame corner.png --low-memory --width 768 --height 768"
        )

    elif total_gb < 12:
        print("🟡 Good VRAM - Should work with optimization:")
        print("  python frame.py frame corner.png --low-memory")
        print("\nOr try lower resolution without low-memory:")
        print("  python frame.py frame corner.png --width 768 --height 768")

    else:
        print("🟢 Excellent VRAM - Try these in order:")
        print("  1. python frame.py frame corner.png")
        print("  2. python frame.py frame corner.png --low-memory  (if #1 fails)")
        print(
            "  3. python frame.py frame corner.png --width 768 --height 768  (if #2 fails)"
        )

    print("\n🔧 Additional options:")
    print("  --steps 20         # Fewer steps = less memory")
    print("  --corner-ratio 0.3 # Smaller corner = less memory")
    print("  --seed 42          # Set seed for reproducibility")


def main():
    """Main function."""
    print("=" * 50)
    print("🔧 CUDA Out of Memory Fix Tool")
    print("=" * 50)

    # Check if specific corner file was passed
    corner_file = None
    if len(sys.argv) > 1:
        corner_file = sys.argv[1]
        if Path(corner_file).exists():
            print(f"✓ Corner file: {corner_file}")

    # Clear GPU memory
    clear_gpu_memory()

    # Check GPU status
    has_gpu = check_gpu_status()

    if has_gpu:
        # Ask about killing processes
        print("\n" + "=" * 50)
        response = input("Kill GPU processes? (y/n): ").strip().lower()
        if response == "y":
            if kill_gpu_processes():
                clear_gpu_memory()
                check_gpu_status()

        # Test memory if requested
        print("\n" + "=" * 50)
        response = input("Test memory usage? (y/n): ").strip().lower()
        if response == "y":
            test_memory_usage()

    # Show recommendations
    print("\n" + "=" * 50)
    show_recommendations()

    # Generate command if corner file provided
    if corner_file:
        print("\n" + "=" * 50)
        print("📋 Ready-to-use command for your file:\n")

        if has_gpu:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1024**3

            if total_gb < 8:
                print(
                    f"python frame.py frame {corner_file} --low-memory --width 768 --height 768"
                )
            else:
                print(f"python frame.py frame {corner_file} --low-memory")
        else:
            print(f"python frame.py frame {corner_file} --device cpu")

    print("\n" + "=" * 50)
    print("✨ Good luck with your frame generation!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
