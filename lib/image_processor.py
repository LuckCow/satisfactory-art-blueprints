#!/usr/bin/env python3
"""
Image to Blueprint Converter
Convert images to Satisfactory painted beam pixel art
"""

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image

from .blueprint import Blueprint, ObjectType, Rotation, Vector3, Layer


class ImageToBlueprint:
    """Convert images to painted beam pixel art"""

    def __init__(self, beam_spacing: float = 100.0):
        self.beam_spacing = beam_spacing

    def rgb_to_linear(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """
        Convert RGB (0-255) to linear color space (0-1) with gamma correction
        Satisfactory uses linear color space internally
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

        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            print(f"Resized to {target_size[0]}x{target_size[1]}")

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
            base_z: float = 1200.0,
            layers: Optional[List[Layer]] = None,
            filter_bg: Optional[str] = None,
            bg_tolerance: float = 30.0,
            orientation: str = 'horizontal'
    ) -> Blueprint:
        """
        Convert image to painted beam blueprint

        Args:
            image_path: Path to image file
            name: Blueprint name
            target_size: (width, height) to resize to, or None for original
            base_z: Base Z height for beams
            layers: List of dimensional layers (primary layer at index 0)
            filter_bg: Background filter method: 'auto', 'corners', 'brightness', or None
            bg_tolerance: Tolerance for background color matching (0-255)
            orientation: 'horizontal' or 'vertical' - beam orientation
        """
        img_array = self.load_and_prepare_image(image_path, target_size, filter_bg, bg_tolerance)
        height, width = img_array.shape[:2]
        has_alpha = img_array.shape[2] == 4

        name = name or image_path.stem
        blueprint = Blueprint(name)

        # Use provided layers or create default primary layer
        if layers:
            for layer in layers:
                blueprint.add_layer(layer)
        else:
            blueprint.add_layer(Layer("Primary", z_offset=0, density=1.0))

        # Determine beam rotation based on orientation
        if orientation.lower() == 'vertical':
            beam_rotation = Rotation.HORIZONTAL_90
            print(f"Converting {width}x{height} image ({width * height} pixels) with VERTICAL orientation...")
        else:
            beam_rotation = Rotation.VERTICAL
            print(f"Converting {width}x{height} image ({width * height} pixels) with HORIZONTAL orientation...")

        # Center the grid
        offset_x = -(width * self.beam_spacing) / 2
        offset_y = -(height * self.beam_spacing) / 2

        # Generate beams for primary layer
        primary_layer = blueprint.layers[0]
        object_count = 0
        skipped_count = 0

        for y in range(height):
            for x in range(width):
                if has_alpha:
                    r, g, b, a = img_array[y, x]
                else:
                    r, g, b = img_array[y, x]
                    a = 255

                # Skip transparent/background pixels
                if self.should_skip_pixel(r, g, b, a):
                    skipped_count += 1
                    continue

                # Convert RGB to linear color space (0-1 range with gamma correction)
                color_linear = self.rgb_to_linear(r, g, b)

                # Calculate position
                base_pos = Vector3(
                    x=offset_x + (x * self.beam_spacing),
                    y=offset_y + (y * self.beam_spacing),
                    z=base_z
                )

                pos = primary_layer.get_position(base_pos.x, base_pos.y, base_pos.z)

                blueprint.add_object(
                    ObjectType.BEAM_PAINTED,
                    pos,
                    beam_rotation,
                    color_linear
                )
                object_count += 1

        if skipped_count > 0:
            print(f"✓ Created {object_count} painted beams ({skipped_count} pixels skipped)")
        else:
            print(f"✓ Created {object_count} painted beams")

        return blueprint


def parse_size_argument(size_arg: str, original_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
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

    # Check for "W H" format (space-separated)
    if ' ' in size_arg:
        try:
            parts = size_arg.split()
            if len(parts) != 2:
                raise ValueError("Size must be two numbers (e.g., '64 64')")
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must be positive")
            return (w, h)
        except ValueError as e:
            raise ValueError(f"Invalid size format: {size_arg}") from e

    raise ValueError(
        f"Invalid size format: {size_arg}. Use WxH (e.g., 64x64), 'W H' (e.g., '64 64'), or percentage (e.g., 50%)")
