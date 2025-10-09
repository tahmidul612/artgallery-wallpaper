"""
Artgallery Wallpaper - Frame generation and background removal utilities.
Compatible with diffusers>=0.25.0 and transformers>=4.36.0

Features:
- Automatic corner image scaling to fit frame dimensions
- Background removal using HSV masking
- Memory-efficient SD 1.5 generation with ControlNet
- Batch processing support
"""

from pathlib import Path
from typing import Optional, Tuple, List
from typing_extensions import Annotated
from enum import Enum
import sys
import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
import warnings

# Try importing diffusers with version check
try:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        UniPCMultistepScheduler,
    )

    # from diffusers.utils import load_image
    import diffusers

    # Check diffusers version for compatibility
    diffusers_version = diffusers.__version__
    print(f"[dim]Using diffusers version: {diffusers_version}[/dim]")

except ImportError:
    print("[red]Error: diffusers library not installed[/red]")
    print("[yellow]Install with: pip install 'diffusers[torch]>=0.35.0'[/yellow]")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled.*")

console = Console()
app = typer.Typer(
    name="artgallery-wallpaper",
    help="Generate framed wallpapers from artwork",
    add_completion=False,
)


class Command(str, Enum):
    """Available commands"""

    frame = "frame"
    remove_bg = "remove-bg"
    both = "both"


def check_cuda_availability() -> str:
    """Check if CUDA is available and return appropriate device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[green]✓[/green] CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("[green]✓[/green] MPS (Apple Silicon) available. Using MPS.")
        return "mps"
    else:
        print("[yellow]⚠[/yellow] No GPU available. Using CPU (will be slower).")
        return "cpu"


def create_mask_and_control(
    canvas: Image.Image, mask_coords: Tuple[int, int, int, int]
) -> Tuple[Image.Image, Image.Image]:
    """Creates a mask and control image from a canvas and coordinates.

    Args:
        canvas: The base canvas image
        mask_coords: Tuple of (x1, y1, x2, y2) coordinates for the mask

    Returns:
        Tuple of (mask_image, control_image)
    """
    mask = Image.new("L", canvas.size, 0)
    mask_np = np.array(mask)
    x1, y1, x2, y2 = mask_coords

    # Validate coordinates
    x1, x2 = max(0, x1), min(canvas.width, x2)
    y1, y2 = max(0, y1), min(canvas.height, y2)

    mask_np[y1:y2, x1:x2] = 255
    mask = Image.fromarray(mask_np)

    # Keep full canvas visible for ControlNet structural guidance
    # Composite on neutral gray background for better AI generation
    if canvas.mode == "RGBA":
        control_bg = Image.new("RGB", canvas.size, (128, 128, 128))
        control_bg.paste(canvas, (0, 0), canvas)
        control_image = control_bg
    else:
        control_image = canvas.convert("RGB")

    return mask, control_image


def generate_frame_sd15(
    corner_path: str,
    width: int = 1000,
    height: int = 1000,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    auto_scale: bool = True,
    corner_size_ratio: float = 0.5,
    low_memory: bool = False,
    generation_mode: str = "full",  # "full" or "edges"
) -> Path:
    """Generates a frame using the SD 1.5 model with ControlNet inpainting.

    Args:
        corner_path: Path to the frame corner image
        width: Width of the frame
        height: Height of the frame
        prompt: Custom prompt for generation
        negative_prompt: Custom negative prompt
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility (None for random)
        output_path: Path to save the output
        device: Device to use (cuda/cpu/mps/auto)
        auto_scale: Whether to automatically scale corner image if needed
        corner_size_ratio: Maximum ratio of corner to frame size (0.0-1.0, default 0.5)
        low_memory: Enable aggressive memory optimization (slower but uses less VRAM)
        generation_mode: "full" to generate entire quadrant, "edges" for edge-only generation

    Returns:
        Path to the generated frame
    """
    # Set memory optimization environment variables
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        if low_memory:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            torch.cuda.empty_cache()
            gc.collect()
    # Check device
    if device is None or device == "auto":
        device = check_cuda_availability()

    # Validate input
    corner_path = Path(corner_path)
    if not corner_path.exists():
        raise FileNotFoundError(f"Corner image not found: {corner_path}")

    # Set up prompts with defaults - emphasize pattern continuation
    if prompt is None:
        prompt = (
            "ornate gilded baroque picture frame, intricate wood carvings continuing seamlessly, "
            "detailed golden ornamental patterns extending from corner, museum quality antique frame, "
            "elaborate decorative molding, carved floral motifs, symmetrical ornate design, "
            "aged patina, high detail photography, professional lighting"
        )
    if negative_prompt is None:
        negative_prompt = (
            "painting, portrait, artwork, picture content, modern frame, minimalist, plain, simple, "
            "blurry, low-resolution, text, signature, watermark, smooth, flat, "
            "gray background, empty space, incomplete pattern"
        )

    print(f"[cyan]Loading models for device: {device}[/cyan]")

    try:
        # Configure dtype and device settings
        if device == "cuda":
            dtype = torch.float16
            variant = "fp16"
            # Check available VRAM
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory = (
                torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            ) / 1024**3
            print(
                f"[dim]GPU Memory: {gpu_memory:.1f}GB total, ~{free_memory:.1f}GB free[/dim]"
            )

            # Adjust for low memory
            if low_memory or gpu_memory < 12:
                print(
                    f"[yellow]Using low memory mode for GPU with {gpu_memory:.1f}GB VRAM[/yellow]"
                )
                low_memory = True
        elif device == "mps":
            dtype = torch.float16
            variant = "fp16"
        else:
            dtype = torch.float32
            variant = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Load ControlNet
            task = progress.add_task("Loading ControlNet model...", total=None)

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint",
                torch_dtype=dtype,
                variant=variant,
                use_safetensors=True,
                low_cpu_mem_usage=True,  # Always use this
                local_files_only=False,
            )

            # Load Pipeline
            progress.update(task, description="Loading Stable Diffusion pipeline...")

            # Load with appropriate settings to avoid memory issues
            pipeline_kwargs = {
                "controlnet": controlnet,
                "torch_dtype": dtype,
                "safety_checker": None,
                "requires_safety_checker": False,
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "local_files_only": False,
            }

            if variant:
                pipeline_kwargs["variant"] = variant

            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", **pipeline_kwargs
            )

            # Set scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            # Memory optimization based on device and settings
            if device == "cuda":
                if low_memory:
                    # Use sequential CPU offload for low memory mode
                    print(
                        "[dim]Using sequential CPU offload for memory optimization[/dim]"
                    )
                    pipe.enable_sequential_cpu_offload()

                    # Additional memory optimizations
                    pipe.enable_attention_slicing(1)  # Maximum memory savings
                    pipe.enable_vae_slicing()
                    pipe.vae.enable_tiling()  # Enable VAE tiling for large images
                else:
                    # Standard GPU mode with some optimizations
                    pipe = pipe.to(device)

                    # Try xformers first
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        print("[dim]Using xformers memory efficient attention[/dim]")
                    except Exception:
                        # Fall back to attention slicing
                        pipe.enable_attention_slicing()
                        print(
                            "[dim]Using attention slicing for memory optimization[/dim]"
                        )

                    # Enable VAE slicing for large images
                    if width > 768 or height > 768:
                        pipe.enable_vae_slicing()
                        pipe.vae.enable_tiling()
                        print(
                            "[dim]Using VAE slicing and tiling for large images[/dim]"
                        )

            elif device == "mps":
                pipe = pipe.to(device)
                pipe.enable_attention_slicing()

            else:  # CPU
                pipe = pipe.to(device)
                pipe.enable_attention_slicing()

            progress.update(task, description="Models loaded successfully!")

    except Exception as e:
        print(f"[red]Error loading models: {e}[/red]")
        print(
            "[yellow]Tip: Try running with --device cpu if GPU issues persist[/yellow]"
        )
        raise

    # Prepare Canvas
    print(f"[cyan]Preparing canvas ({width}x{height})...[/cyan]")
    corner_img = Image.open(corner_path).convert("RGBA")
    original_corner_w, original_corner_h = corner_img.size

    # Calculate maximum allowed dimensions for corner based on ratio
    corner_size_ratio = min(
        max(corner_size_ratio, 0.1), 1.0
    )  # Clamp between 0.1 and 1.0
    max_corner_width = int(width * corner_size_ratio)
    max_corner_height = int(height * corner_size_ratio)

    # Check if corner needs scaling
    corner_w, corner_h = original_corner_w, original_corner_h

    if auto_scale:
        if corner_w > max_corner_width or corner_h > max_corner_height:
            # Corner is too large, need to downscale
            # Calculate scale factor to fit within bounds while maintaining aspect ratio
            scale_factor = min(
                max_corner_width / corner_w, max_corner_height / corner_h
            )

            new_width = int(corner_w * scale_factor)
            new_height = int(corner_h * scale_factor)

            print(
                f"[yellow]Corner image ({corner_w}x{corner_h}) exceeds {corner_size_ratio:.0%} of frame size.[/yellow]"
            )
            print(f"[cyan]Downscaling to ({new_width}x{new_height})...[/cyan]")

            # Use high-quality Lanczos resampling for downscaling
            corner_img = corner_img.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            corner_w, corner_h = new_width, new_height

        elif corner_w < max_corner_width * 0.2 or corner_h < max_corner_height * 0.2:
            # Corner is too small (less than 20% of max size), consider upscaling
            target_size_ratio = 0.4  # Target 40% of max size
            scale_factor = min(
                (max_corner_width * target_size_ratio) / corner_w,
                (max_corner_height * target_size_ratio) / corner_h,
                3.0,  # Don't upscale more than 3x to avoid quality loss
            )

            if scale_factor > 1.1:  # Only upscale if significant (>10% increase)
                new_width = int(corner_w * scale_factor)
                new_height = int(corner_h * scale_factor)

                print(
                    f"[yellow]Corner image ({corner_w}x{corner_h}) is small for frame.[/yellow]"
                )
                print(f"[cyan]Upscaling to ({new_width}x{new_height})...[/cyan]")

                # Use cubic interpolation for upscaling
                corner_img = corner_img.resize(
                    (new_width, new_height), Image.Resampling.BICUBIC
                )
                corner_w, corner_h = new_width, new_height
        else:
            print(
                f"[dim]Corner size ({corner_w}x{corner_h}) is appropriate for frame[/dim]"
            )
    else:
        # Manual mode - just validate
        if corner_w > max_corner_width or corner_h > max_corner_height:
            raise ValueError(
                f"Corner image ({corner_w}x{corner_h}) exceeds maximum size "
                f"({max_corner_width}x{max_corner_height}) for frame. "
                f"Use --auto-scale to automatically resize."
            )
        print(f"[dim]Using corner size as-is: ({corner_w}x{corner_h})[/dim]")

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(corner_img, (0, 0), corner_img if corner_img.mode == "RGBA" else None)

    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[cyan]Using seed: {seed}[/cyan]")
    else:
        import random

        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[cyan]Generated random seed: {seed}[/cyan]")

    # Generation approach based on mode
    if generation_mode == "full":
        # Generate the entire top-left quadrant (better for avoiding gaps)
        print("[yellow]Generating full top-left quadrant...[/yellow]")

        # Create mask for everything except the corner
        # Add small overlap and ensure divisible by 8 for SD model
        quadrant_width = ((width // 2 + 4 + 7) // 8) * 8
        quadrant_height = ((height // 2 + 4 + 7) // 8) * 8

        # Create a quadrant canvas
        quadrant_canvas = Image.new(
            "RGBA", (quadrant_width, quadrant_height), (0, 0, 0, 0)
        )
        quadrant_canvas.paste(
            corner_img, (0, 0), corner_img if corner_img.mode == "RGBA" else None
        )

        # Mask everything except the corner
        quadrant_mask = Image.new("L", (quadrant_width, quadrant_height), 255)
        mask_draw = ImageDraw.Draw(quadrant_mask)
        mask_draw.rectangle([(0, 0), (corner_w, corner_h)], fill=0)

        # Create control image - composite on neutral gray to help AI understand frame context
        control_bg = Image.new(
            "RGB", (quadrant_width, quadrant_height), (128, 128, 128)
        )
        control_bg.paste(
            quadrant_canvas,
            (0, 0),
            quadrant_canvas if quadrant_canvas.mode == "RGBA" else None,
        )
        quadrant_control = control_bg

        # Adjust for memory if needed
        gen_q_width = quadrant_width
        gen_q_height = quadrant_height
        if low_memory and (quadrant_width > 512 or quadrant_height > 512):
            scale = min(512 / quadrant_width, 512 / quadrant_height)
            gen_q_width = int(quadrant_width * scale)
            gen_q_height = int(quadrant_height * scale)
            gen_q_width = (gen_q_width // 8) * 8
            gen_q_height = (gen_q_height // 8) * 8

            quadrant_canvas = quadrant_canvas.resize(
                (gen_q_width, gen_q_height), Image.Resampling.LANCZOS
            )
            quadrant_mask = quadrant_mask.resize(
                (gen_q_width, gen_q_height), Image.Resampling.NEAREST
            )
            quadrant_control = quadrant_control.resize(
                (gen_q_width, gen_q_height), Image.Resampling.LANCZOS
            )

        try:
            if device == "cuda":
                torch.cuda.empty_cache()

            # Create image with neutral gray background for inpainting (helps AI generate textures)
            image_bg = Image.new("RGB", (gen_q_width, gen_q_height), (128, 128, 128))
            if gen_q_width == quadrant_width and gen_q_height == quadrant_height:
                image_bg.paste(
                    quadrant_canvas,
                    (0, 0),
                    quadrant_canvas if quadrant_canvas.mode == "RGBA" else None,
                )
            else:
                # Canvas was already resized above
                image_bg.paste(
                    quadrant_canvas,
                    (0, 0),
                    quadrant_canvas if quadrant_canvas.mode == "RGBA" else None,
                )

            with torch.no_grad():
                if device == "cuda" and low_memory:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=image_bg,
                            mask_image=quadrant_mask,
                            control_image=quadrant_control,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            guidance_scale=12.0,  # Higher guidance for better pattern following
                            controlnet_conditioning_scale=0.5,  # Lower to allow creative generation
                            control_guidance_end=0.5,  # Stop ControlNet guidance halfway through
                            strength=0.99,  # Maximum creativity in masked area
                            width=gen_q_width,
                            height=gen_q_height,
                        ).images[0]
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image_bg,
                        mask_image=quadrant_mask,
                        control_image=quadrant_control,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=12.0,  # Higher guidance for better pattern following
                        controlnet_conditioning_scale=0.5,  # Lower to allow creative generation
                        control_guidance_end=0.5,  # Stop ControlNet guidance halfway through
                        strength=0.99,  # Maximum creativity in masked area
                        width=gen_q_width,
                        height=gen_q_height,
                    ).images[0]

            # Upscale if needed
            if gen_q_width != quadrant_width or gen_q_height != quadrant_height:
                result = result.resize(
                    (quadrant_width, quadrant_height), Image.Resampling.LANCZOS
                )

            # Convert to RGBA and ensure corner is preserved
            result = result.convert("RGBA")
            result.paste(
                corner_img, (0, 0), corner_img if corner_img.mode == "RGBA" else None
            )

            # Set canvas to the generated quadrant for assembly
            canvas = result

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("[red]CUDA out of memory! Try using --low-memory flag[/red]")
            raise
        except Exception as e:
            print(f"[red]Error generating quadrant: {e}[/red]")
            raise

    else:  # edges mode (original approach with fixes)
        # Generate Top Edge
        print("[yellow]Generating top edge...[/yellow]")
        # Generate full width to avoid gaps
        top_mask_coords = (corner_w, 0, width, corner_h)
        top_mask, top_control = create_mask_and_control(canvas, top_mask_coords)

        # Adjust generation size for memory constraints if needed
        gen_width = width
        gen_height = height
        if low_memory and (width > 1024 or height > 1024):
            # Generate at lower resolution then upscale
            scale = min(1024 / width, 1024 / height)
            gen_width = int(width * scale)
            gen_height = int(height * scale)
            gen_width = (gen_width // 8) * 8  # Ensure divisible by 8
            gen_height = (gen_height // 8) * 8
            print(f"[dim]Generating at {gen_width}x{gen_height} to save memory[/dim]")

            # Resize inputs for generation
            gen_canvas = canvas.resize(
                (gen_width, gen_height), Image.Resampling.LANCZOS
            )
            gen_top_mask = top_mask.resize(
                (gen_width, gen_height), Image.Resampling.NEAREST
            )
            gen_top_control = top_control.resize(
                (gen_width, gen_height), Image.Resampling.LANCZOS
            )
        else:
            gen_canvas = canvas
            gen_top_mask = top_mask
            gen_top_control = top_control

        try:
            # Clear cache before generation
            if device == "cuda":
                torch.cuda.empty_cache()

            # Create image with neutral gray background for inpainting
            if gen_canvas.mode == "RGBA":
                gen_image_bg = Image.new(
                    "RGB", (gen_width, gen_height), (128, 128, 128)
                )
                gen_image_bg.paste(gen_canvas, (0, 0), gen_canvas)
            else:
                gen_image_bg = gen_canvas.convert("RGB")

            with torch.no_grad():
                if device == "cuda" and low_memory:
                    # Use autocast for additional memory savings
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result_top = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=gen_image_bg,
                            mask_image=gen_top_mask,
                            control_image=gen_top_control,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            guidance_scale=12.0,
                            controlnet_conditioning_scale=0.5,
                            control_guidance_end=0.5,
                            strength=0.99,
                            width=gen_width,
                            height=gen_height,
                        ).images[0]
                else:
                    result_top = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=gen_image_bg,
                        mask_image=gen_top_mask,
                        control_image=gen_top_control,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=12.0,
                        controlnet_conditioning_scale=0.5,
                        control_guidance_end=0.5,
                        strength=0.99,
                        width=gen_width,
                        height=gen_height,
                    ).images[0]

            # Upscale if we generated at lower resolution
            if gen_width != width or gen_height != height:
                result_top = result_top.resize(
                    (width, height), Image.Resampling.LANCZOS
                )

            # Update canvas with top edge
            result_top_rgba = result_top.convert("RGBA")
            canvas.paste(result_top_rgba, (0, 0), mask=top_mask)

            # Aggressive memory cleanup
            del result_top, result_top_rgba
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("[red]CUDA out of memory! Try using --low-memory flag[/red]")
            print(
                "[yellow]Tip: You can also try --width 768 --height 768 for lower resolution[/yellow]"
            )
            raise
        except Exception as e:
            print(f"[red]Error generating top edge: {e}[/red]")
            raise

        # Generate Left Edge
        print("[yellow]Generating left edge...[/yellow]")
        # Fix: Ensure left edge covers the full height including the bottom corner
        left_mask_coords = (
            0,
            corner_h,
            corner_w,
            height,
        )  # Changed from height - corner_h to height
        left_mask, left_control = create_mask_and_control(canvas, left_mask_coords)

        # Use same resolution as before
        if gen_width != width or gen_height != height:
            gen_canvas = canvas.resize(
                (gen_width, gen_height), Image.Resampling.LANCZOS
            )
            gen_left_mask = left_mask.resize(
                (gen_width, gen_height), Image.Resampling.NEAREST
            )
            gen_left_control = left_control.resize(
                (gen_width, gen_height), Image.Resampling.LANCZOS
            )
        else:
            gen_canvas = canvas
            gen_left_mask = left_mask
            gen_left_control = left_control

        try:
            # Clear cache before generation
            if device == "cuda":
                torch.cuda.empty_cache()

            # Create image with neutral gray background for inpainting
            if gen_canvas.mode == "RGBA":
                gen_image_bg_left = Image.new(
                    "RGB", (gen_width, gen_height), (128, 128, 128)
                )
                gen_image_bg_left.paste(gen_canvas, (0, 0), gen_canvas)
            else:
                gen_image_bg_left = gen_canvas.convert("RGB")

            with torch.no_grad():
                if device == "cuda" and low_memory:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result_left = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=gen_image_bg_left,
                            mask_image=gen_left_mask,
                            control_image=gen_left_control,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            guidance_scale=12.0,
                            control_guidance_end=0.5,
                            controlnet_conditioning_scale=0.5,
                            strength=0.99,
                            width=gen_width,
                            height=gen_height,
                        ).images[0]
                else:
                    result_left = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=gen_image_bg_left,
                        mask_image=gen_left_mask,
                        control_image=gen_left_control,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=12.0,
                        control_guidance_end=0.5,
                        controlnet_conditioning_scale=0.5,
                        strength=0.99,
                        width=gen_width,
                        height=gen_height,
                    ).images[0]

            # Upscale if needed
            if gen_width != width or gen_height != height:
                result_left = result_left.resize(
                    (width, height), Image.Resampling.LANCZOS
                )

            # Update canvas with left edge
            result_left_rgba = result_left.convert("RGBA")
            canvas.paste(result_left_rgba, (0, 0), mask=left_mask)

            # Aggressive memory cleanup
            del result_left, result_left_rgba
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("[red]CUDA out of memory! Try using --low-memory flag[/red]")
            print(
                "[yellow]Tip: You can also try --width 768 --height 768 for lower resolution[/yellow]"
            )
            raise
        except Exception as e:
            print(f"[red]Error generating left edge: {e}[/red]")
            raise

    # Clean up pipeline to free memory
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # Assemble Full Frame with proper blending
    print("[cyan]Assembling full frame with symmetry...[/cyan]")

    if generation_mode == "full":
        # We generated a full quadrant, now mirror it
        quadrant = canvas  # The generated quadrant

        # Ensure we're working with the right size
        min(quadrant.width, width // 2 + 4)
        min(quadrant.height, height // 2 + 4)

        # Create final frame
        final_frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Crop to exact half dimensions for clean mirroring
        top_left = quadrant.crop((0, 0, width // 2, height // 2))

        # Create other quadrants by mirroring
        top_right = ImageOps.mirror(top_left)
        bottom_left = ImageOps.flip(top_left)
        bottom_right = ImageOps.mirror(bottom_left)

        # Paste with 1 pixel overlap to avoid gaps
        final_frame.paste(
            top_left, (0, 0), top_left if top_left.mode == "RGBA" else None
        )
        final_frame.paste(
            top_right, (width // 2, 0), top_right if top_right.mode == "RGBA" else None
        )
        final_frame.paste(
            bottom_left,
            (0, height // 2),
            bottom_left if bottom_left.mode == "RGBA" else None,
        )
        final_frame.paste(
            bottom_right,
            (width // 2, height // 2),
            bottom_right if bottom_right.mode == "RGBA" else None,
        )

    else:
        # Original edge-based assembly
        # Extract quadrants from the canvas with generated edges
        top_left = canvas.crop((0, 0, width // 2, height // 2))

        # Mirror to create other quadrants
        top_right = ImageOps.mirror(top_left)
        bottom_left = ImageOps.flip(top_left)
        bottom_right = ImageOps.mirror(bottom_left)

        # Create final frame
        final_frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Paste quadrants
        final_frame.paste(
            top_left, (0, 0), top_left if top_left.mode == "RGBA" else None
        )
        final_frame.paste(
            top_right, (width // 2, 0), top_right if top_right.mode == "RGBA" else None
        )
        final_frame.paste(
            bottom_left,
            (0, height // 2),
            bottom_left if bottom_left.mode == "RGBA" else None,
        )
        final_frame.paste(
            bottom_right,
            (width // 2, height // 2),
            bottom_right if bottom_right.mode == "RGBA" else None,
        )

    # Optional: Blend seams to make them less visible
    # Apply a subtle blur at the center seams
    if width >= 256 and height >= 256:
        try:
            # Create a copy for blending
            blended_frame = final_frame.copy()

            # Get the center strips
            v_strip_width = 4
            h_strip_height = 4

            # Extract vertical center strip
            v_strip = final_frame.crop(
                (
                    width // 2 - v_strip_width // 2,
                    0,
                    width // 2 + v_strip_width // 2,
                    height,
                )
            )
            v_strip = v_strip.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Extract horizontal center strip
            h_strip = final_frame.crop(
                (
                    0,
                    height // 2 - h_strip_height // 2,
                    width,
                    height // 2 + h_strip_height // 2,
                )
            )
            h_strip = h_strip.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Paste blurred strips back
            blended_frame.paste(v_strip, (width // 2 - v_strip_width // 2, 0), v_strip)
            blended_frame.paste(
                h_strip, (0, height // 2 - h_strip_height // 2), h_strip
            )

            final_frame = blended_frame
            print("[dim]Applied seam blending[/dim]")
        except Exception as e:
            print(f"[dim]Seam blending skipped: {e}[/dim]")

    # Make interior transparent (where gray background is still visible)
    print("[cyan]Making interior transparent...[/cyan]")
    try:
        # Convert to numpy for processing
        frame_array = np.array(final_frame)

        # Find pixels that are close to gray (128, 128, 128) - these are un-generated areas
        # Calculate distance from gray for each pixel
        gray_target = np.array([128, 128, 128])
        rgb = frame_array[:, :, :3].astype(float)
        diff = np.abs(rgb - gray_target)
        dist_from_gray = np.mean(diff, axis=2)  # Average distance across RGB channels

        # Pixels very close to gray (threshold: 30) should be made transparent
        # This represents areas the AI didn't fill with frame details
        is_interior = dist_from_gray < 30

        # Set alpha to 0 for interior pixels
        frame_array[:, :, 3] = np.where(is_interior, 0, frame_array[:, :, 3])

        # Convert back to image
        final_frame = Image.fromarray(frame_array)
        print("[dim]Interior made transparent[/dim]")
    except Exception as e:
        print(f"[yellow]Warning: Could not make interior transparent: {e}[/yellow]")
        print("[dim]Continuing with full frame...[/dim]")

    # Save output
    if output_path is None:
        output_path = Path(f"frame_output_{seed}.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame.save(output_path, "PNG", optimize=True)

    print(f"[green]✓[/green] Frame saved to: {output_path}")
    print(f"[dim]Seed used: {seed} (save this to reproduce the same frame)[/dim]")

    return output_path


def remove_background(
    image_path: str,
    saturation_threshold: int = 40,
    open_kernel_size: int = 3,
    close_kernel_size: int = 7,
    output_path: Optional[str] = None,
    batch: bool = False,
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
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_files = [image_path]

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


@app.command("frame")
def frame_command(
    corner_path: Annotated[str, typer.Argument(help="Path to the frame corner image")],
    width: Annotated[
        int, typer.Option("--width", "-w", help="Width of the frame")
    ] = 1000,
    height: Annotated[
        int, typer.Option("--height", "-h", help="Height of the frame")
    ] = 1000,
    prompt: Annotated[
        Optional[str], typer.Option("--prompt", "-p", help="Custom generation prompt")
    ] = None,
    negative_prompt: Annotated[
        Optional[str], typer.Option("--negative", "-n", help="Custom negative prompt")
    ] = None,
    steps: Annotated[
        int, typer.Option("--steps", "-s", help="Number of inference steps")
    ] = 30,
    seed: Annotated[
        Optional[int], typer.Option("--seed", help="Random seed (None for random)")
    ] = None,
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output path")
    ] = None,
    device: Annotated[
        str, typer.Option("--device", "-d", help="Device (cuda/cpu/mps/auto)")
    ] = "auto",
    auto_scale: Annotated[
        bool,
        typer.Option(
            "--auto-scale/--no-auto-scale", help="Auto-scale corner if needed"
        ),
    ] = True,
    corner_ratio: Annotated[
        float, typer.Option("--corner-ratio", help="Max corner size ratio (0.1-1.0)")
    ] = 0.5,
    low_memory: Annotated[
        bool,
        typer.Option(
            "--low-memory", "-l", help="Enable aggressive memory optimization"
        ),
    ] = False,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Generation mode: 'full' or 'edges'")
    ] = "full",
):
    """Generate a decorative frame using Stable Diffusion 1.5 with ControlNet.

    Generation modes:
    - 'full': Generate entire quadrant then mirror (better for avoiding gaps/seams)
    - 'edges': Generate top and left edges separately (original method)

    Examples:
      frame corner.png                          # Basic usage (full quadrant mode)
      frame corner.png --mode edges             # Use edge-based generation
      frame corner.png --low-memory             # For GPUs with <12GB VRAM
      frame corner.png --width 768 --height 768 # Smaller size for less VRAM
    """
    try:
        generate_frame_sd15(
            corner_path=corner_path,
            width=width,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            seed=seed,
            output_path=output,
            device=device,
            auto_scale=auto_scale,
            corner_size_ratio=corner_ratio,
            low_memory=low_memory,
            generation_mode=mode,
        )
    except Exception as e:
        console.print(f"[red]Frame generation failed: {e}[/red]")
        if "CUDA out of memory" in str(e):
            console.print(
                "\n[yellow]💡 Suggestions to fix CUDA out of memory:[/yellow]"
            )
            console.print("1. Try with --low-memory flag")
            console.print("2. Reduce size: --width 768 --height 768")
            console.print("3. Use CPU instead: --device cpu")
            console.print("4. Close other GPU applications")
        raise typer.Exit(code=1)


@app.command("remove-bg")
def remove_bg_command(
    image_path: Annotated[str, typer.Argument(help="Path to image or directory")],
    saturation: Annotated[
        int, typer.Option("--saturation", "-s", help="Saturation threshold")
    ] = 40,
    open_kernel: Annotated[int, typer.Option("--open", help="Opening kernel size")] = 3,
    close_kernel: Annotated[
        int, typer.Option("--close", help="Closing kernel size")
    ] = 7,
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output path")
    ] = None,
    batch: Annotated[
        bool, typer.Option("--batch", "-b", help="Process directory as batch")
    ] = False,
):
    """Remove grayscale background from images using HSV masking."""
    try:
        processed = remove_background(
            image_path=image_path,
            saturation_threshold=saturation,
            open_kernel_size=open_kernel,
            close_kernel_size=close_kernel,
            output_path=output,
            batch=batch,
        )

        if not processed:
            console.print("[yellow]No images were processed[/yellow]")

    except Exception as e:
        console.print(f"[red]Background removal failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("both")
def both_command(
    corner_path: Annotated[str, typer.Argument(help="Path to the frame corner image")],
    width: Annotated[int, typer.Option("--width", "-w")] = 1000,
    height: Annotated[int, typer.Option("--height", "-h")] = 1000,
    remove_corner_bg: Annotated[
        bool, typer.Option("--remove-bg", help="Remove corner background first")
    ] = False,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    output: Annotated[Optional[str], typer.Option("--output", "-o")] = None,
    device: Annotated[str, typer.Option("--device", "-d")] = "auto",
    auto_scale: Annotated[bool, typer.Option("--auto-scale/--no-auto-scale")] = True,
    corner_ratio: Annotated[float, typer.Option("--corner-ratio")] = 0.5,
    low_memory: Annotated[bool, typer.Option("--low-memory", "-l")] = False,
    mode: Annotated[str, typer.Option("--mode", "-m")] = "full",
):
    """Generate frame, optionally removing background from corner first."""
    try:
        corner_to_use = Path(corner_path)

        if remove_corner_bg:
            console.print("[cyan]Removing background from corner image...[/cyan]")
            processed = remove_background(
                image_path=str(corner_path),
                output_path=str(Path(corner_path).with_suffix(".no_bg.png")),
            )
            if processed:
                corner_to_use = processed[0]

        console.print("[cyan]Generating frame...[/cyan]")
        generate_frame_sd15(
            corner_path=str(corner_to_use),
            width=width,
            height=height,
            seed=seed,
            output_path=output,
            device=device,
            auto_scale=auto_scale,
            corner_size_ratio=corner_ratio,
            low_memory=low_memory,
            generation_mode=mode,
        )

    except Exception as e:
        console.print(f"[red]Process failed: {e}[/red]")
        if "CUDA out of memory" in str(e):
            console.print(
                "\n[yellow]💡 Try with --low-memory flag or smaller dimensions[/yellow]"
            )
        raise typer.Exit(code=1)


@app.command("resize")
def resize_command(
    image_path: Annotated[str, typer.Argument(help="Path to image to resize")],
    target_width: Annotated[
        Optional[int], typer.Option("--width", "-w", help="Target width")
    ] = None,
    target_height: Annotated[
        Optional[int], typer.Option("--height", "-h", help="Target height")
    ] = None,
    max_size: Annotated[
        Optional[int], typer.Option("--max", "-m", help="Maximum dimension")
    ] = None,
    scale: Annotated[
        Optional[float], typer.Option("--scale", "-s", help="Scale factor")
    ] = None,
    output: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Output path")
    ] = None,
    quality: Annotated[
        int, typer.Option("--quality", "-q", help="JPEG quality (1-100)")
    ] = 95,
):
    """Resize an image with various options.

    Examples:
      resize image.png --width 500 --height 500  # Exact size
      resize image.png --max 1000                 # Fit within 1000x1000
      resize image.png --scale 0.5                # Scale to 50%
    """
    try:
        # Open image
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        console.print(f"[cyan]Original size: {orig_w}x{orig_h}[/cyan]")

        # Determine target size
        if scale is not None:
            # Scale by factor
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        elif max_size is not None:
            # Fit within max_size while maintaining aspect ratio
            ratio = min(max_size / orig_w, max_size / orig_h)
            if ratio < 1:
                new_w = int(orig_w * ratio)
                new_h = int(orig_h * ratio)
            else:
                new_w, new_h = orig_w, orig_h
        elif target_width is not None or target_height is not None:
            # Resize to specific dimensions
            if target_width and target_height:
                # Both specified
                new_w, new_h = target_width, target_height
            elif target_width:
                # Only width specified, maintain aspect ratio
                ratio = target_width / orig_w
                new_w = target_width
                new_h = int(orig_h * ratio)
            else:
                # Only height specified, maintain aspect ratio
                ratio = target_height / orig_h
                new_h = target_height
                new_w = int(orig_w * ratio)
        else:
            console.print(
                "[red]Please specify target size (--width, --height, --max, or --scale)[/red]"
            )
            raise typer.Exit(code=1)

        # Determine resampling method
        if new_w < orig_w or new_h < orig_h:
            # Downscaling
            resample = Image.Resampling.LANCZOS
            console.print(f"[yellow]Downscaling to {new_w}x{new_h}...[/yellow]")
        else:
            # Upscaling
            resample = Image.Resampling.BICUBIC
            console.print(f"[yellow]Upscaling to {new_w}x{new_h}...[/yellow]")

        # Resize
        resized = img.resize((new_w, new_h), resample)

        # Determine output path
        if output is None:
            p = Path(image_path)
            output = str(p.parent / f"{p.stem}_resized{p.suffix}")

        # Save with appropriate format
        output_path = Path(output)
        if output_path.suffix.lower() in [".jpg", ".jpeg"]:
            resized.save(output, quality=quality, optimize=True)
        else:
            resized.save(output)

        console.print(f"[green]✓ Saved to: {output} ({new_w}x{new_h})[/green]")

        # Show size reduction if applicable
        orig_size = os.path.getsize(image_path)
        new_size = os.path.getsize(output)
        if new_size < orig_size:
            reduction = (1 - new_size / orig_size) * 100
            console.print(f"[dim]File size reduced by {reduction:.1f}%[/dim]")

    except Exception as e:
        console.print(f"[red]Resize failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("gpu-check")
def gpu_check_command(
    test_load: Annotated[
        bool, typer.Option("--test-load", help="Test loading the models")
    ] = False,
):
    """Check GPU memory and capabilities.

    Shows available VRAM and provides recommendations for optimal settings.
    Use --test-load to actually load the models and check memory usage.
    """

    console.print("[bold cyan]GPU Memory Check[/bold cyan]\n")

    if not torch.cuda.is_available():
        console.print("[yellow]No CUDA GPU detected.[/yellow]")
        console.print("You can still use --device cpu but it will be slower.\n")
        return

    # Get GPU info
    gpu_id = 0
    gpu_name = torch.cuda.get_device_name(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    free = total_memory - reserved

    console.print(f"GPU: [green]{gpu_name}[/green]")
    console.print(f"Total VRAM: [cyan]{total_memory:.1f} GB[/cyan]")
    console.print(f"Currently allocated: [yellow]{allocated:.1f} GB[/yellow]")
    console.print(f"Currently reserved: [yellow]{reserved:.1f} GB[/yellow]")
    console.print(f"Available: [green]{free:.1f} GB[/green]\n")

    # Recommendations based on VRAM
    console.print("[bold]Recommendations based on your GPU:[/bold]")

    if total_memory >= 16:
        console.print("✅ Excellent! You have plenty of VRAM.")
        console.print("• Can generate at high resolutions (2048x2048+)")
        console.print("• No need for --low-memory flag")
    elif total_memory >= 12:
        console.print("✅ Good! You have sufficient VRAM for most tasks.")
        console.print("• Can generate at 1536x1536 without issues")
        console.print("• Use --low-memory for 2048x2048 or larger")
    elif total_memory >= 8:
        console.print("⚠️  Moderate VRAM - some optimizations needed.")
        console.print("• Recommended max: 1024x1024 without --low-memory")
        console.print("• Use --low-memory flag for better stability")
        console.print("• Or use --width 768 --height 768 for faster generation")
    elif total_memory >= 6:
        console.print("⚠️  Limited VRAM - significant optimizations needed.")
        console.print("• Always use --low-memory flag")
        console.print("• Recommended max: 768x768")
        console.print("• Consider --width 512 --height 512 for stability")
    else:
        console.print("❌ Very limited VRAM.")
        console.print("• Use --device cpu instead (will be slow)")
        console.print("• Or use --low-memory with --width 512 --height 512")

    # Test model loading if requested
    if test_load:
        console.print("\n[cyan]Testing model loading...[/cyan]")
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetInpaintPipeline,
            )

            torch.cuda.empty_cache()
            initial_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved(0)
            ) / 1024**3

            console.print("Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint",
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            console.print("Loading Stable Diffusion pipeline...")
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()

            after_load = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved(0)
            ) / 1024**3
            used = initial_free - after_load

            console.print("\n[green]✓ Models loaded successfully![/green]")
            console.print(f"Memory used by models: ~{used:.1f} GB")
            console.print(f"Memory remaining: {after_load:.1f} GB")

            # Clean up
            del pipe, controlnet
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            console.print(f"[red]Error loading models: {e}[/red]")

    console.print("\n[dim]Tip: Close other GPU applications to free up VRAM[/dim]")


@app.command("info")
def info_command():
    """Show system and dependency information."""
    import platform

    console.print("[bold cyan]System Information[/bold cyan]")
    console.print(f"Python: {sys.version}")
    console.print(f"Platform: {platform.platform()}")

    console.print("\n[bold cyan]PyTorch Information[/bold cyan]")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA version: {torch.version.cuda}")
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Memory info
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        console.print(f"GPU Memory: {mem_allocated:.1f}GB / {mem_total:.1f}GB")
    console.print(f"MPS available: {torch.backends.mps.is_available()}")

    console.print("\n[bold cyan]Package Versions[/bold cyan]")
    packages = {
        "diffusers": None,
        "transformers": None,
        "accelerate": None,
        "opencv-python": "cv2",
        "safetensors": None,
        "xformers": None,
    }

    for pkg_name, import_name in packages.items():
        try:
            import importlib

            mod_name = import_name or pkg_name.replace("-", "_")
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            console.print(f"{pkg_name}: {version}")
        except ImportError:
            console.print(f"{pkg_name}: [red]Not installed[/red]")

    console.print("\n[bold cyan]Available Commands[/bold cyan]")
    console.print("• frame      - Generate a decorative frame from a corner image")
    console.print("• remove-bg  - Remove grayscale background from images")
    console.print("• resize     - Resize images with various options")
    console.print("• both       - Combine background removal and frame generation")
    console.print("• gpu-check  - Check GPU memory and get recommendations")
    console.print("• info       - Show this information")

    console.print("\n[dim]Tip: Use --help with any command for more options[/dim]")
    console.print(
        "[dim]Having CUDA OOM errors? Try: frame corner.png --low-memory[/dim]"
    )
    console.print(
        "[dim]Seeing gaps or seams? Use default --mode full (not --mode edges)[/dim]"
    )


def main():
    """Entry point for the artgallery-wallpaper application."""
    app()


if __name__ == "__main__":
    main()
