#!/usr/bin/env python3
"""Test what parameters the pipeline actually accepts."""

import inspect
from diffusers import StableDiffusionControlNetInpaintPipeline

# Check the __call__ method signature
sig = inspect.signature(StableDiffusionControlNetInpaintPipeline.__call__)
print("StableDiffusionControlNetInpaintPipeline.__call__ parameters:")
print("=" * 80)
for param_name, param in sig.parameters.items():
    if param_name not in ["self", "args", "kwargs"]:
        default = (
            param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        )
        print(f"  {param_name}: {default}")

print("\n" + "=" * 80)
print("\nKey findings:")
if "strength" in sig.parameters:
    print("✓ 'strength' parameter IS supported")
else:
    print("✗ 'strength' parameter NOT supported - will be ignored!")

if "mask_image" in sig.parameters:
    print("✓ 'mask_image' parameter IS supported")
else:
    print("✗ 'mask_image' parameter NOT supported!")
