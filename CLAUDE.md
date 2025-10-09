# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application that generates framed wallpapers from artwork using AI image generation. It takes frame corner images and generates complete decorative frames using Stable Diffusion 1.5 with ControlNet inpainting, then can composite artwork into those frames for desktop wallpapers.

## Commands

### Development Setup

```bash
# Install dependencies (uses uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows

# Run the main CLI
python -m artgallery-wallpaper.frame <command>
# Or if installed in development mode
artgallery-wallpaper <command>
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Linting and Formatting

```bash
# Format code with ruff
ruff format .

# Check and fix linting issues
ruff check --fix .
```

### Testing Commands

```bash
# Test frame generation (basic)
python -m artgallery-wallpaper.frame frame assets/frames/corners/<corner-image.png>

# Test with low memory mode (for GPUs <12GB VRAM)
python -m artgallery-wallpaper.frame frame assets/frames/corners/<corner-image.png> --low-memory

# Test background removal
python -m artgallery-wallpaper.frame remove-bg assets/art/<image.jpg>

# Check GPU status and get recommendations
python -m artgallery-wallpaper.frame gpu-check

# Diagnose CUDA memory issues
python scripts/cuda_memory_fix.py
```

## Architecture

### Core Module: `src/artgallery-wallpaper/frame.py`

This is the main module containing all functionality. It implements:

1. **Frame Generation Pipeline** (`generate_frame_sd15`):

   - Uses Stable Diffusion 1.5 + ControlNet inpainting to generate frames
   - Takes a corner image and generates a complete symmetrical frame
   - Two generation modes:
     - `"full"` (default): Generates entire top-left quadrant then mirrors it (avoids seams)
     - `"edges"`: Generates top and left edges separately (legacy method)
   - Automatically handles corner image scaling based on frame dimensions
   - Memory optimization strategies for different GPU VRAM levels
   - Supports CUDA, MPS (Apple Silicon), and CPU backends

1. **Background Removal** (`remove_background`):

   - Uses HSV color space masking to remove grayscale backgrounds
   - Effective for separating colorful artwork from white/gray/black backgrounds
   - Supports batch processing of directories
   - Morphological operations (opening/closing) for cleanup

1. **CLI Commands** (using Typer):

   - `frame`: Generate decorative frame from corner image
   - `remove-bg`: Remove background from images
   - `both`: Chain background removal + frame generation
   - `resize`: Resize images with quality preservation
   - `gpu-check`: Check GPU memory and get optimization recommendations
   - `info`: Show system/dependency information

### Memory Optimization Strategy

The code implements sophisticated memory management for CUDA GPUs:

- **High VRAM (≥16GB)**: Full resolution, no special optimizations
- **Medium VRAM (8-12GB)**: Attention slicing, VAE slicing/tiling
- **Low VRAM (6-8GB)**: `--low-memory` mode with sequential CPU offload, autocast, reduced resolution generation with upscaling
- **Very Low VRAM (\<6GB)**: Force smaller dimensions (512x512 or 768x768)

Key memory techniques:

- Sequential CPU offload (`enable_sequential_cpu_offload`)
- Attention slicing (`enable_attention_slicing`)
- VAE slicing and tiling for large images
- xformers memory efficient attention when available
- Aggressive cache clearing between operations
- Generation at reduced resolution with Lanczos upscaling

### Frame Assembly Process

1. **Quadrant Generation** (full mode):

   - Generate top-left quadrant (width/2 × height/2) with corner preserved
   - Mirror horizontally for top-right
   - Mirror vertically for bottom-left and bottom-right
   - Optional seam blending with Gaussian blur at center lines

1. **Edge Generation** (edges mode - legacy):

   - Generate top edge (preserving corner)
   - Generate left edge (covering full height)
   - Extract and mirror quadrant from combined result

### Utility Script: `scripts/cuda_memory_fix.py`

Interactive diagnostic tool for CUDA memory issues:

- Clears GPU memory cache
- Reports current VRAM usage
- Identifies GPU processes
- Provides command recommendations based on available VRAM
- Can kill GPU processes (with confirmation)
- Tests memory usage with small models

## Key Technical Details

### Dependencies Compatibility

Critical version constraints in `pyproject.toml`:

- PyTorch 2.1.x (not 2.2+) for stability
- NumPy \<2.0 (many ML packages incompatible with 2.0+)
- diffusers 0.25.0 with transformers 4.36.2 (tested compatible set)
- Python 3.9-3.11 (not 3.12+, PyTorch compatibility)

### Model Downloads

First run downloads ~5GB of models from HuggingFace:

- `lllyasviel/control_v11p_sd15_inpaint` (ControlNet)
- `runwayml/stable-diffusion-v1-5` (base model)

Models are cached in `~/.cache/huggingface/`

### Image Processing Details

- All images converted to RGBA for transparency support
- Corner scaling uses Lanczos for downscaling, Bicubic for upscaling
- Default corner size ratio: 50% of frame dimension (configurable)
- Seam blending applies 4-pixel Gaussian blur at center lines
- Output always saved as PNG with optimization

### Generation Parameters

Default prompts optimized for ornate picture frames:

- **Positive**: "ornate gilded wood frame, intricate baroque carvings, detailed texture, high-resolution photography, aged patina, museum quality, antique frame"
- **Negative**: "painting, portrait, picture inside frame, modern, minimalist, blurry, low-resolution, text, signature, watermark, content, image"

## Common Patterns

### Adding New Commands

1. Define command function with `@app.command()` decorator
1. Use `Annotated[type, typer.Option(...)]` for parameters
1. Wrap main logic in try-except for graceful error handling
1. Use `rich.console` for formatted output
1. Return appropriate exit code on error

### Error Handling

- CUDA OOM errors provide specific suggestions (--low-memory, smaller dimensions)
- File not found errors include full path
- Model loading errors suggest CPU fallback
- All errors use rich console formatting for readability

### Adding New Frame Features

When modifying frame generation:

- Always preserve corner image in final output
- Maintain symmetry by operating on quadrants
- Clear CUDA cache before/after major operations
- Wrap generation in `torch.no_grad()` context
- Test with both `--mode full` and `--mode edges`

## Development Notes

- Use `uv` for dependency management (not pip)
- Pre-commit hooks enforce ruff formatting and linting
- Line length: 88 characters (Black-compatible)
- Python 3.9 target version for compatibility
- Conventional commits enforced via commitizen
- All file paths should use `pathlib.Path`
