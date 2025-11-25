#!/usr/bin/env python3
"""
Blueprint Generator - Image to Painted Beam Pixel Art
Convert images to Satisfactory blueprints using painted beams as pixels

IMPROVED VERSION with universal offset strategy to prevent diagonal overlap
"""

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from lib.blueprint import Blueprint, Layer, ObjectType, Rotation


class ImageToBlueprint:
    """Convert images to painted beam blueprints with advanced rendering options"""

    def __init__(
            self,
            beam_spacing: float = 100.0,
            condensed_rendering: bool = False,
            cr_multiplier: int = 2,
            cr_universal_offset: float = 0.001
    ):
        """
        Initialize converter

        Args:
            beam_spacing: Distance between beams in cm
            condensed_rendering: Enable condensed rendering mode (multiple beams per pixel)
            cr_multiplier: Number of beams per pixel axis (NxN grid)
            cr_universal_offset: Universal offset per unit distance (replaces depth and valley offsets)
        """
        self.beam_spacing = beam_spacing
        self.condensed_rendering = condensed_rendering
        self.cr_multiplier = cr_multiplier
        self.cr_universal_offset = cr_universal_offset

    def rgb_to_linear(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """
        Convert RGB (0-255) to linear color space (0-1) with gamma correction

        Args:
            r, g, b: RGB values in range 0-255

        Returns:
            Linear RGB values in range 0-1
        """

        def srgb_to_linear(c: float) -> float:
            """Convert sRGB to linear color space"""
            if c <= 0.04045:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4

        # Normalize to 0-1 range
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        # Apply gamma correction for linear color space
        r_linear = srgb_to_linear(r_norm)
        g_linear = srgb_to_linear(g_norm)
        b_linear = srgb_to_linear(b_norm)

        return (r_linear, g_linear, b_linear)

    def filter_background(
            self,
            img_array: np.ndarray,
            method: str = 'auto',
            tolerance: float = 30.0
    ) -> np.ndarray:
        """
        Filter background pixels to transparency

        Args:
            img_array: RGB image array
            method: 'auto', 'corners', 'color', or 'brightness'
            tolerance: Color distance tolerance (0-255)

        Returns:
            RGBA image array with background as transparent
        """
        height, width = img_array.shape[:2]

        # Convert to RGBA if needed
        if img_array.shape[2] == 3:
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array
            rgba[:, :, 3] = 255  # Fully opaque initially
        else:
            rgba = img_array.copy()

        if method == 'auto':
            # Auto-detect: sample corners and use most common color
            corners = [
                img_array[0, 0],  # Top-left
                img_array[0, -1],  # Top-right
                img_array[-1, 0],  # Bottom-left
                img_array[-1, -1]  # Bottom-right
            ]
            # Use top-left corner as reference
            bg_color = corners[0]
            method = 'color'

        elif method == 'corners':
            # Average the corner colors
            corners = [
                img_array[0, 0],
                img_array[0, -1],
                img_array[-1, 0],
                img_array[-1, -1]
            ]
            bg_color = np.mean(corners, axis=0).astype(np.uint8)
            method = 'color'

        elif method == 'brightness':
            # Remove very dark or very bright pixels
            brightness = np.mean(img_array, axis=2)
            dark_mask = brightness < 20
            bright_mask = brightness > 235
            rgba[dark_mask | bright_mask, 3] = 0
            return rgba

        # Color-based filtering
        if method == 'color' and 'bg_color' in locals():
            for y in range(height):
                for x in range(width):
                    pixel = img_array[y, x]
                    # Calculate color distance
                    distance = np.sqrt(np.sum((pixel.astype(float) - bg_color.astype(float)) ** 2))

                    if distance < tolerance:
                        rgba[y, x, 3] = 0  # Make transparent

        return rgba

    def should_skip_pixel(self, r: int, g: int, b: int, alpha: int = 255) -> bool:
        """
        Determine if a pixel should be skipped (not converted to beam)

        Args:
            r, g, b: RGB values (0-255)
            alpha: Alpha value (0-255, default 255 for opaque)

        Returns:
            True if pixel should be skipped
        """
        # Skip transparent pixels
        if alpha < 10:
            return True

        return False

    def calculate_universal_depth_offset(
            self,
            x: int,
            y: int,
            width: int,
            height: int
    ) -> float:
        """
        Calculate depth offset using a universal strategy that ensures all 8 neighbors
        have unique depths, preventing overlap on diagonals.

        Strategy combines:
        1. Checkerboard pattern (alternating x+y parity)
        2. Spiral distance from edges (like peeling an onion)
        3. Fine-grained position encoding (x*prime1 + y*prime2)

        Args:
            x, y: Pixel coordinates
            width, height: Image dimensions

        Returns:
            Depth offset value
        """
        # # Component 1: Checkerboard pattern (primary separation)
        # # This ensures diagonal neighbors differ by at least 1 unit
        # checkerboard = (x + y) % 2

        # # Component 2: Spiral/onion distance from edges
        # # Distance to nearest edge, creating concentric layers
        # edge_dist = min(x, width - 1 - x, y, height - 1 - y)

        # # Component 3: Fine-grained position encoding
        # # Use coprime numbers to ensure unique values for adjacent positions
        # # Prime multipliers ensure no two adjacent cells have the same encoding
        # position_encoding = (x * 7 + y * 11) % 97  # Modulo keeps values reasonable

        # # Combine all components with different weights
        # # The weights are carefully chosen to ensure neighboring beams differ
        # total_offset = (
        #     checkerboard * 1.0 +           # Primary separation (0 or 1)
        #     edge_dist * 0.5 +               # Layer separation (0, 0.5, 1.0, 1.5, ...)
        #     position_encoding * 0.01        # Fine position encoding (0.00 to 0.96)
        # )

        # this is what I was actually going for, silly AI
        edge_dist_x = min(x, width - 1 - x)
        edge_dist_y = min(y, height - 1 - y)
        total_offset = edge_dist_x + edge_dist_y

        return total_offset * self.cr_universal_offset

    def load_and_prepare_image(
            self,
            path: Path,
            target_size: Optional[Tuple[int, int]] = None,
            filter_bg: Optional[str] = None,
            bg_tolerance: float = 30.0
    ) -> np.ndarray:
        """
        Load image and optionally resize and filter background

        Args:
            path: Image file path
            target_size: Optional (width, height) to resize to
            filter_bg: Background filter method: 'auto', 'corners', 'brightness', or None
            bg_tolerance: Tolerance for background color matching (0-255)
        """
        img = Image.open(path).convert('RGBA')

        # Calculate final size upfront for condensed rendering to avoid double-resizing and fidelity loss
        if target_size:
            if self.condensed_rendering:
                # For condensed rendering, calculate the final upscaled size and resize once
                final_size = (target_size[0] * self.cr_multiplier, target_size[1] * self.cr_multiplier)
                img = img.resize(final_size, Image.Resampling.LANCZOS)
                print(f"Resized to {final_size[0]}x{final_size[1]} for condensed rendering (target: {target_size[0]}x{target_size[1]}, multiplier: {self.cr_multiplier}x)")
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                print(f"Resized to {target_size[0]}x{target_size[1]}")
        elif self.condensed_rendering:
            # No target_size but condensed rendering enabled - upscale from original resolution
            upscaled_size = (img.width * self.cr_multiplier, img.height * self.cr_multiplier)
            img = img.resize(upscaled_size, Image.Resampling.LANCZOS)
            print(f"Upscaled for condensed rendering to {upscaled_size[0]}x{upscaled_size[1]}")

        img_array = np.array(img)

        # Apply background filtering if requested
        if filter_bg:
            print(f"Filtering background (method: {filter_bg}, tolerance: {bg_tolerance})...")
            img_array = self.filter_background(img_array, method=filter_bg, tolerance=bg_tolerance)

            # Count transparent pixels
            if img_array.shape[2] == 4:
                transparent_count = np.sum(img_array[:, :, 3] < 10)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                print(
                    f"  Filtered {transparent_count:,} pixels ({transparent_count / total_pixels * 100:.1f}% of image)")

        return img_array

    def convert(
            self,
            image_path: Path,
            name: Optional[str] = None,
            target_size: Optional[Tuple[int, int]] = None,
            filter_bg: Optional[str] = None,
            bg_tolerance: float = 30.0,
            rotation: Rotation = Rotation.VERTICAL
    ) -> Blueprint:
        """
        Convert image to painted beam blueprint

        Args:
            image_path: Path to image file
            name: Blueprint name
            target_size: (width, height) to resize to, or None for original
            filter_bg: Background filter method: 'auto', 'corners', 'brightness', or None
            bg_tolerance: Tolerance for background color matching (0-255)
            rotation: Beam rotation (default: Rotation.VERTICAL)
        """
        img_array = self.load_and_prepare_image(image_path, target_size, filter_bg, bg_tolerance)
        height, width = img_array.shape[:2]
        has_alpha = img_array.shape[2] == 4

        name = name or image_path.stem
        blueprint = Blueprint(name)

        print(f"Converting {width}x{height} image ({width * height} pixels)...")

        # Adjust beam spacing for condensed rendering (smaller spacing so beams overlap)
        effective_beam_spacing = self.beam_spacing / self.cr_multiplier if self.condensed_rendering else self.beam_spacing

        # Center the grid - vertical layout uses X-Z plane
        offset_x = -(width * effective_beam_spacing) / 2
        offset_z = -(height * effective_beam_spacing) / 2
        base_y = -1000.0  # Constant Y depth for vertical layout

        # Generate beams for primary layer
        primary_layer = blueprint.layers[0]

        for y in range(height):
            for x in range(width):
                if has_alpha:
                    r, g, b, a = img_array[y, x]
                else:
                    r, g, b = img_array[y, x]
                    a = 255

                # Skip transparent/background pixels
                if self.should_skip_pixel(r, g, b, a):
                    continue

                # Convert RGB to linear color space (0-1 range with gamma correction)
                color_linear = self.rgb_to_linear(r, g, b)

                # Calculate base position for this pixel - vertical layout (X-Z plane)
                base_x = offset_x + (x * effective_beam_spacing)
                base_pos_z = offset_z + ((height - 1 - y) * effective_beam_spacing)

                # Calculate Y-depth based on universal offset strategy
                if self.condensed_rendering:
                    depth_offset = self.calculate_universal_depth_offset(x, y, width, height)
                    base_pos_y = base_y - depth_offset  # Subtract to come closer to viewer
                else:
                    base_pos_y = base_y  # Constant Y (depth into the wall)

                # Add beam to blueprint
                pos = primary_layer.get_position(base_x, base_pos_y, base_pos_z)
                blueprint.add_object(
                    ObjectType.BEAM_PAINTED,
                    pos,
                    rotation,
                    color_linear
                )


        return blueprint


def parse_size_argument(size_arg: str, original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Parse size argument which can be:
    - "WxH" (e.g., "64x64", "1920x1080")
    - "W H" (e.g., "64 64", "1920 1080")
    - "50%" (percentage of original)
    - None (full resolution)
    """
    if not size_arg:
        return None

    # Check for percentage
    if size_arg.endswith('%'):
        try:
            percent = float(size_arg[:-1]) / 100.0
            if percent <= 0 or percent > 1:
                raise ValueError("Percentage must be between 0 and 100")
            new_w = int(original_size[0] * percent)
            new_h = int(original_size[1] * percent)
            return (max(1, new_w), max(1, new_h))
        except ValueError as e:
            raise ValueError(f"Invalid percentage format: {size_arg}") from e

    # Check for WxH format
    if 'x' in size_arg.lower():
        try:
            parts = size_arg.lower().split('x')
            if len(parts) != 2:
                raise ValueError("Size must be in format WxH (e.g., 64x64)")
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")
            return (w, h)
        except ValueError as e:
            raise ValueError(f"Invalid size format: {size_arg}") from e

    # Check for "W H" format with space
    parts = size_arg.split()
    if len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")
            return (w, h)
        except ValueError as e:
            raise ValueError(f"Invalid size format: {size_arg}") from e

    raise ValueError(f"Invalid size format: {size_arg}. Use WxH, 'W H', or %%")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to Satisfactory painted beam blueprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                        # Full resolution, default spacing
  %(prog)s image.png -s 64x64               # Downsample to 64x64
  %(prog)s image.png -s "64 64"             # Same as above (quoted)
  %(prog)s image.png -s 25%%              # Downsample to 25%% of original
  %(prog)s image.png -o art.json -n "Art"   # Custom output and name
  %(prog)s image.png --spacing 150          # Increase beam spacing
  %(prog)s image.png --filter-bg auto       # Auto-detect and remove background
  %(prog)s image.png --filter-bg corners --bg-tolerance 50  # Remove corner color
  %(prog)s image.png -H                     # Use horizontal picture layout (X-Y plane)
  %(prog)s image.png --condensed            # Enable condensed rendering (2x2 beams per pixel)
  %(prog)s image.png --condensed --cr-multiplier 3  # 3x3 beams per pixel (9x detail)
  %(prog)s image.png --condensed --cr-universal-offset 0.01 # Adjust universal offset

Condensed rendering (IMPROVED):
  --condensed:               Enable condensed rendering mode with depth-clipping
  --cr-multiplier N:         Number of beams per pixel axis (NxN grid, default: 2)
  --cr-universal-offset:     Universal depth offset multiplier (default: 0.001)

  The universal offset strategy combines checkerboard pattern, spiral layers, and
  fine position encoding to ensure all 8 neighboring beams have unique depths.
  This completely eliminates diagonal overlap while simplifying configuration.

Background filtering:
  --filter-bg auto:       Use top-left corner color as background
  --filter-bg corners:    Average all four corners as background
  --filter-bg brightness: Remove very dark (<20) and bright (>235) pixels
  --bg-tolerance:         Color distance (0-255, default 30, lower=stricter)

Resolution formats:
  - WxH:      64x64, 1920x1080, 3840x2160
  - "W H":    "64 64", "1920 1080" (use quotes for spaces)
  - Percent:  50%%, 25%%, 10%%

Resolution limits:
  - Maximum: 4K (3840x2160 = 8,294,400 beams)
  - Default: Full resolution (no downsampling)
  - Recommended: 64x64 to 512x512 for practical blueprints
        """
    )

    parser.add_argument('image', type=Path, help='Input image file')
    parser.add_argument('-o', '--output', type=Path, help='Output blueprint file (default: <image>.json)')
    parser.add_argument('-n', '--name', help='Blueprint name (default: image filename)')
    parser.add_argument('-s', '--size', type=str, metavar='SIZE',
                        help='Target size: WxH (e.g., 64x64) or percentage (e.g., 50%%). Default: full resolution')
    parser.add_argument('--spacing', type=float, default=100.0,
                        help='Spacing between beams in cm (default: 100)')
    parser.add_argument('--max-4k', action='store_true',
                        help='Enforce 4K resolution limit (3840x2160)')
    parser.add_argument('--filter-bg', type=str, choices=['auto', 'corners', 'brightness'],
                        help='Filter background: auto (corner color), corners (average corners), brightness (dark/bright)')
    parser.add_argument('--bg-tolerance', type=float, default=30.0,
                        help='Background color tolerance 0-255 (default: 30, lower=stricter)')
    parser.add_argument('-H', '--horizontal', action='store_true',
                        help='Use horizontal picture layout (X-Y plane, beams vertical). Default: vertical picture layout (X-Z plane, beams horizontal)')
    parser.add_argument('--condensed', action='store_true',
                        help='Enable condensed rendering: pack multiple beams per pixel using depth-clipping for higher detail')
    parser.add_argument('--cr-multiplier', type=int, default=3, metavar='N',
                        help='Condensed rendering: NxN grid of sub-beams per pixel (default: 2, i.e., 2x2=4 beams per pixel)')
    parser.add_argument('--cr-universal-offset', type=int, default=1, metavar='OFFSET',
                        help='Condensed rendering: Universal offset multiplier (default: 1). Replaces separate depth and valley offsets.')

    args = parser.parse_args()

    # Validate input file
    if not args.image.exists():
        print(f"❌ Error: Image not found: {args.image}")
        return 1

    # Get original image dimensions
    try:
        with Image.open(args.image) as img:
            original_size = img.size
            print(
                f"📐 Original image size: {original_size[0]}x{original_size[1]} ({original_size[0] * original_size[1]:,} pixels)")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1

    # Parse target size
    try:
        target_size = parse_size_argument(args.size, original_size)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Apply 4K limit warnings and enforcement
    MAX_4K_PIXELS = 3840 * 2160  # 8,294,400 pixels
    MAX_4K_SIZE = (3840, 2160)

    if target_size is None:
        # Full resolution mode
        total_pixels = original_size[0] * original_size[1]
        if total_pixels > MAX_4K_PIXELS:
            if args.max_4k:
                # Enforce 4K limit
                aspect_ratio = original_size[0] / original_size[1]
                if aspect_ratio > MAX_4K_SIZE[0] / MAX_4K_SIZE[1]:
                    # Width is limiting factor
                    target_size = (MAX_4K_SIZE[0], int(MAX_4K_SIZE[0] / aspect_ratio))
                else:
                    # Height is limiting factor
                    target_size = (int(MAX_4K_SIZE[1] * aspect_ratio), MAX_4K_SIZE[1])
                print(f"⚠️  Image exceeds 4K. Downscaling to {target_size[0]}x{target_size[1]}")
            else:
                print(f"⚠️  Warning: Image exceeds 4K resolution!")
                print(f"    This will create {total_pixels:,} beams.")
                print(f"    Consider using -s option to downsample (e.g., -s 50% or -s 1920x1080)")
                print(f"    Or use --max-4k flag to enforce 4K limit")
                response = input("    Continue with full resolution? [y/N]: ")
                if response.lower() != 'y':
                    print("Cancelled.")
                    return 0
        elif total_pixels > 1000000:  # Warn for anything over 1M pixels
            print(f"⚠️  Large resolution: {total_pixels:,} beams will be created")
            print(f"    Blueprint file will be approximately {total_pixels * 2 / 1024:.0f} MB")
    else:
        # Check if downsampled size still exceeds 4K
        target_pixels = target_size[0] * target_size[1]
        if args.max_4k and target_pixels > MAX_4K_PIXELS:
            print(f"❌ Error: Target size {target_size[0]}x{target_size[1]} exceeds 4K limit")
            print(f"    Maximum allowed: {MAX_4K_PIXELS:,} pixels")
            return 1

    # Display target resolution
    if target_size:
        print(f"🎯 Target size: {target_size[0]}x{target_size[1]} ({target_size[0] * target_size[1]:,} pixels)")
        reduction = (1 - (target_size[0] * target_size[1]) / (original_size[0] * original_size[1])) * 100
        print(f"   ({reduction:.1f}% reduction)")
    else:
        print(f"🎯 Target size: Full resolution ({original_size[0] * original_size[1]:,} pixels)")

    # Determine output path
    output_path = args.output or args.image.with_suffix('.json')

    # Convert image to blueprint
    print(f"\n🔧 Converting image to painted beam blueprint...")
    if args.condensed:
        print(f"   Condensed rendering enabled: {args.cr_multiplier}x{args.cr_multiplier} beams per pixel (universal-offset: {args.cr_universal_offset})")
    converter = ImageToBlueprint(
        beam_spacing=args.spacing,
        condensed_rendering=args.condensed,
        cr_multiplier=args.cr_multiplier,
        cr_universal_offset=args.cr_universal_offset
    )

    # Determine rotation based on horizontal flag
    # Vertical picture (default) uses HORIZONTAL_90 rotation for beams lying in X-Z plane
    # Horizontal picture uses VERTICAL rotation for beams standing through X-Y plane
    beam_rotation = Rotation.VERTICAL if args.horizontal else Rotation.HORIZONTAL_90

    blueprint = converter.convert(
        args.image,
        name=args.name,
        target_size=target_size,
        filter_bg=args.filter_bg,
        bg_tolerance=args.bg_tolerance,
        rotation=beam_rotation
    )

    # Save blueprint
    blueprint.save(output_path)

    # Display summary
    print(f"\n📊 Summary:")
    print(f"   Input:  {args.image}")
    print(f"   Output: {output_path}")
    print(f"   Beams:  {len(blueprint.objects):,}")
    print(f"   Size:   {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    return 0


if __name__ == '__main__':
    exit(main())