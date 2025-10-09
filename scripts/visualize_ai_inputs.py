#!/usr/bin/env python3
"""Visualize what inputs the AI is receiving."""

import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")

# Simulate what the AI sees
corner_img = Image.open(
    "assets/frames/corners/Cassetta_frame_MET_86K_FRTS5R7.no_bg.png"
)
print(f"Original corner: {corner_img.size}")

# Scale it like the code does
corner_w, corner_h = corner_img.size
max_corner_width = int(800 * 0.5)  # 400
max_corner_height = int(800 * 0.5)  # 400

scale_factor = min(max_corner_width / corner_w, max_corner_height / corner_h)
new_width = int(corner_w * scale_factor)
new_height = int(corner_h * scale_factor)
corner_img = corner_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
print(f"Scaled corner: {corner_img.size}")

# Quadrant size
quadrant_width = ((800 // 2 + 4 + 7) // 8) * 8
quadrant_height = ((800 // 2 + 4 + 7) // 8) * 8
print(f"Quadrant size: {quadrant_width}x{quadrant_height}")

# Create what AI sees
quadrant_canvas = Image.new("RGBA", (quadrant_width, quadrant_height), (0, 0, 0, 0))
quadrant_canvas.paste(corner_img, (0, 0), corner_img)

# Show with gray background (what AI receives)
visual = Image.new("RGB", (quadrant_width, quadrant_height), (128, 128, 128))
visual.paste(quadrant_canvas, (0, 0), quadrant_canvas)

# Draw mask overlay
mask = Image.new("L", (quadrant_width, quadrant_height), 255)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rectangle([(0, 0), (new_width, new_height)], fill=0)

# Create visualization with mask shown in red overlay
viz = visual.copy()
viz_arr = Image.new("RGBA", viz.size, (0, 0, 0, 0))
viz_rgba = Image.new("RGBA", viz.size)
viz_rgba.paste(viz, (0, 0))

# Add red overlay where mask is 255 (to be generated)
mask_arr = np.array(mask)
viz_data = np.array(viz_rgba)
viz_data[:, :, 0] = np.where(
    mask_arr > 127, np.minimum(viz_data[:, :, 0] + 100, 255), viz_data[:, :, 0]
)
viz_data[:, :, 3] = 255
viz_rgba = Image.fromarray(viz_data)

viz_rgba.save("ai_input_visualization.png")
visual.save("ai_input_actual.png")

print("\nSaved:")
print("  ai_input_actual.png - What the AI actually sees")
print("  ai_input_visualization.png - Red = area AI must generate")
print(
    f"\nProblem: The AI has very little context (just {new_width}x{new_height} corner)"
)
print(f"         and must fill {quadrant_width}x{quadrant_height} - that's a huge gap!")
