#!/usr/bin/env python3
"""
Blueprint Generator - Image to Painted Beam Pixel Art
Convert images to Satisfactory blueprints using painted beams as pixels
"""

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


class ObjectType(Enum):
    """Satisfactory object types"""
    BEAM_PAINTED = "Beam_Painted"
    FOUNDATION_8X4 = "Foundation_8x4_01"
    PILLAR_MIDDLE = "PillarMiddle"
    WALL_8X4 = "Wall_8x4_01"

    @property
    def path(self) -> str:
        """Get the full Unreal Engine path for this object"""
        paths = {
            self.BEAM_PAINTED: "/Game/FactoryGame/Prototype/Buildable/Beams/Build_Beam_Painted.Build_Beam_Painted_C",
            self.FOUNDATION_8X4: "/Game/FactoryGame/Buildable/Building/Foundation/Build_Foundation_8x4_01.Build_Foundation_8x4_01_C",
            self.PILLAR_MIDDLE: "/Game/FactoryGame/Buildable/Building/Pillars/Build_PillarMiddle.Build_PillarMiddle_C",
            self.WALL_8X4: "/Game/FactoryGame/Buildable/Building/Wall/Build_Wall_8x4_01.Build_Wall_8x4_01_C",
        }
        return paths[self]

    @property
    def recipe_path(self) -> str:
        """Get the recipe path for this object"""
        recipes = {
            self.BEAM_PAINTED: "/Game/FactoryGame/Prototype/Buildable/Beams/Recipe_Beam_Painted.Recipe_Beam_Painted_C",
            self.FOUNDATION_8X4: "/Game/FactoryGame/Recipes/Buildings/Foundations/Recipe_Foundation_8x4_01.Recipe_Foundation_8x4_01_C",
            self.PILLAR_MIDDLE: "/Game/FactoryGame/Recipes/Buildings/Foundations/Recipe_PillarMiddle.Recipe_PillarMiddle_C",
            self.WALL_8X4: "/Game/FactoryGame/Recipes/Buildings/Walls/Recipe_Wall_8x4_01.Recipe_Wall_8x4_01_C",
        }
        return recipes[self]


class ColorSwatch(Enum):
    """
    Satisfactory color swatches (slots 0-255)
    NOTE: This is now deprecated in favor of custom RGB colors (OverrideColorData)
    Custom colors allow exact pixel-perfect color matching instead of predefined swatches
    """
    SLOT_0 = 0  # Custom Slot 1
    SLOT_1 = 1  # Custom Slot 2
    SLOT_2 = 2  # Custom Slot 3
    SLOT_3 = 3  # Custom Slot 4
    SLOT_4 = 4  # Custom Slot 5
    SLOT_5 = 5  # Custom Slot 6
    SLOT_6 = 6  # Custom Slot 7
    SLOT_7 = 7  # Custom Slot 8
    SLOT_8 = 8  # FICSIT Orange
    SLOT_9 = 9  # FICSIT Yellow-Orange
    SLOT_10 = 10  # FICSIT Yellow
    SLOT_11 = 11  # FICSIT Lime
    SLOT_12 = 12  # FICSIT Green
    SLOT_13 = 13  # FICSIT Aqua
    SLOT_14 = 14  # FICSIT Sky Blue
    SLOT_15 = 15  # FICSIT Blue
    SLOT_16 = 16  # FICSIT Dark Blue
    SLOT_17 = 17  # FICSIT Purple
    SLOT_18 = 18  # FICSIT Magenta
    SLOT_19 = 19  # FICSIT Pink
    SLOT_20 = 20  # FICSIT Red

    @property
    def path(self) -> str:
        """Get the Unreal Engine path for this swatch"""
        return f"/Game/FactoryGame/Buildable/-Shared/Customization/Swatches/SwatchDesc_Slot{self.value}.SwatchDesc_Slot{self.value}_C"


class Rotation(Enum):
    """Common rotation presets for painted beams"""
    HORIZONTAL_0 = (0.0, 0.0, 0.0, 1.0)  # Facing forward
    HORIZONTAL_90 = (0.0, 0.0, 0.707107, 0.707107)  # Rotated 90°
    HORIZONTAL_180 = (0.0, 0.0, 1.0, 0.0)  # Facing back
    HORIZONTAL_270 = (0.0, 0.0, -0.707107, 0.707107)  # Rotated 270°
    VERTICAL = (-0.5, 0.5, -0.5, -0.5)  # Standing upright (as in example)

    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        return self.value


@dataclass
class Vector3:
    """3D vector"""
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Quaternion:
    """Quaternion for rotations"""
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_rotation(cls, rotation: Rotation) -> 'Quaternion':
        x, y, z, w = rotation.quaternion
        return cls(x, y, z, w)

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}


@dataclass
class Layer:
    """A dimensional layer in the manifold system"""
    name: str
    z_offset: float = 0.0  # Z position offset
    angle_offset: float = 0.0  # Angular offset in degrees
    density: float = 1.0  # 1.0 = solid, 0.0 = empty

    def get_position(self, base_x: float, base_y: float, base_z: float) -> Vector3:
        """Calculate position with layer offsets applied"""
        # For now, simple Z offset (angular offset can be added later)
        return Vector3(base_x, base_y, base_z + self.z_offset)


class BlueprintObject:
    """A single object in the blueprint"""

    def __init__(
            self,
            object_type: ObjectType,
            position: Vector3,
            rotation: Quaternion,
            color_rgb: Optional[Tuple[float, float, float]] = None,  # RGB values 0-1
            instance_id: int = 0
    ):
        self.object_type = object_type
        self.position = position
        self.rotation = rotation
        self.color_rgb = color_rgb  # Store RGB instead of swatch
        self.instance_id = instance_id

    def to_dict(self) -> Dict:
        """Convert to blueprint JSON format"""
        type_name = self.object_type.value
        instance_name = f"Persistent_Level:PersistentLevel.Build_{type_name}_C_{self.instance_id}"

        obj_dict = {
            "typePath": self.object_type.path,
            "rootObject": "Persistent_Level",
            "instanceName": instance_name,
            "flags": 8,
            "properties": {
                "mBuiltWithRecipe": {
                    "type": "ObjectProperty",
                    "ueType": "ObjectProperty",
                    "name": "mBuiltWithRecipe",
                    "value": {
                        "levelName": "",
                        "pathName": self.object_type.recipe_path
                    }
                }
            },
            "specialProperties": {"type": "EmptySpecialProperties"},
            "trailingData": [],
            "saveCustomVersion": 0,
            "shouldMigrateObjectRefsToPersistent": False,
            "parentEntityName": "",
            "type": "SaveEntity",
            "needTransform": True,
            "wasPlacedInLevel": False,
            "parentObject": {
                "levelName": "Persistent_Level",
                "pathName": "Persistent_Level:PersistentLevel.BuildableSubsystem"
            },
            "transform": {
                "rotation": self.rotation.to_dict(),
                "translation": self.position.to_dict(),
                "scale3d": {"x": 1, "y": 1, "z": 1}
            },
            "components": []
        }

        # Add painted beam specific properties
        if self.object_type == ObjectType.BEAM_PAINTED:
            obj_dict["properties"]["mLength"] = {
                "type": "FloatProperty",
                "ueType": "FloatProperty",
                "name": "mLength",
                "value": 100
            }

            # Set color slot to 255 for custom color
            obj_dict["properties"]["mColorSlot"] = {
                "type": "ByteProperty",
                "ueType": "ByteProperty",
                "name": "mColorSlot",
                "value": {
                    "type": "None",
                    "value": 255
                }
            }

        # Add custom color override if specified
        if self.color_rgb:
            r, g, b = self.color_rgb
            obj_dict["properties"]["mCustomizationData"] = {
                "type": "StructProperty",
                "ueType": "StructProperty",
                "name": "mCustomizationData",
                "value": {
                    "type": "FactoryCustomizationData",
                    "properties": {
                        "SwatchDesc": {
                            "type": "ObjectProperty",
                            "ueType": "ObjectProperty",
                            "name": "SwatchDesc",
                            "value": {
                                "levelName": "",
                                "pathName": "/Game/FactoryGame/Buildable/-Shared/Customization/Swatches/SwatchDesc_Custom.SwatchDesc_Custom_C"
                            }
                        },
                        "OverrideColorData": {
                            "type": "StructProperty",
                            "ueType": "StructProperty",
                            "name": "OverrideColorData",
                            "value": {
                                "type": "FactoryCustomizationColorSlot",
                                "properties": {
                                    "PrimaryColor": {
                                        "type": "StructProperty",
                                        "ueType": "StructProperty",
                                        "name": "PrimaryColor",
                                        "value": {
                                            "r": r,
                                            "g": g,
                                            "b": b,
                                            "a": 1
                                        },
                                        "subtype": "LinearColor"
                                    },
                                    "SecondaryColor": {
                                        "type": "StructProperty",
                                        "ueType": "StructProperty",
                                        "name": "SecondaryColor",
                                        "value": {
                                            "r": r * 0.5,  # Slightly darker for secondary
                                            "g": g * 0.5,
                                            "b": b * 0.5,
                                            "a": 1
                                        },
                                        "subtype": "LinearColor"
                                    },
                                    "PaintFinish": {
                                        "type": "ObjectProperty",
                                        "ueType": "ObjectProperty",
                                        "name": "PaintFinish",
                                        "value": {
                                            "levelName": "",
                                            "pathName": "/Game/FactoryGame/Buildable/-Shared/Customization/PaintFinishes/PaintFinishDesc_Concrete.PaintFinishDesc_Concrete_C"
                                        }
                                    }
                                }
                            },
                            "subtype": "FactoryCustomizationColorSlot"
                        }
                    }
                },
                "subtype": "FactoryCustomizationData"
            }

        return obj_dict


class Blueprint:
    """Multi-layer blueprint with dimensional manifold support"""

    def __init__(self, name: str = "PixelArt"):
        self.name = name
        self.objects: List[BlueprintObject] = []
        self.layers: List[Layer] = []
        self.next_id = 2147483647  # Start from max int32

        # Default beam configuration
        self.beam_spacing = 100.0  # 100cm between beams (tight grid)
        self.default_rotation = Rotation.VERTICAL

    def add_layer(self, layer: Layer):
        """Add a dimensional layer"""
        self.layers.append(layer)

    def add_object(
            self,
            object_type: ObjectType,
            position: Vector3,
            rotation: Optional[Rotation] = None,
            color_rgb: Optional[Tuple[float, float, float]] = None
    ) -> BlueprintObject:
        """Add an object to the blueprint with optional RGB color (0-1 range)"""
        rot = Quaternion.from_rotation(rotation or self.default_rotation)
        obj = BlueprintObject(object_type, position, rot, color_rgb, self.next_id)
        self.objects.append(obj)
        self.next_id -= 1
        return obj

    def calculate_dimensions(self) -> Vector3:
        """Calculate blueprint dimensions"""
        if not self.objects:
            return Vector3(1, 1, 1)

        positions = [obj.position for obj in self.objects]
        max_x = max(abs(p.x) for p in positions)
        max_y = max(abs(p.y) for p in positions)
        max_z = max(abs(p.z) for p in positions)

        return Vector3(
            int(max_x / 100) + 2,
            int(max_y / 100) + 2,
            int(max_z / 100) + 2
        )

    def get_unique_recipes(self) -> List[Dict]:
        """Get list of unique recipes used"""
        recipes = set()
        for obj in self.objects:
            recipes.add(obj.object_type.recipe_path)

        return [
            {"levelName": "", "pathName": recipe}
            for recipe in sorted(recipes)
        ]

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        dimensions = self.calculate_dimensions()

        return {
            "name": self.name,
            "compressionInfo": {
                "chunkHeaderVersion": 572662306,
                "packageFileTag": 2653586369,
                "maxUncompressedChunkContentSize": 131072,
                "compressionAlgorithm": 3
            },
            "header": {
                "headerVersion": 2,
                "saveVersion": 52,
                "buildVersion": 455399,
                "designerDimension": dimensions.to_dict(),
                "recipeReferences": self.get_unique_recipes(),
                "itemCosts": []  # Could be calculated if needed
            },
            "config": {
                "configVersion": 4,
                "description": "Generated pixel art from image",
                "color": {"r": 0.28755027055740356, "g": 0.10702301561832428, "b": 0.5583410263061523, "a": 1},
                "iconID": 393,
                "referencedIconLibrary": "/Game/FactoryGame/-Shared/Blueprint/IconLibrary",
                "iconLibraryType": "IconLibrary",
                "lastEditedBy": []
            },
            "objects": [obj.to_dict() for obj in self.objects]
        }

    def save(self, filepath: Path):
        """Save blueprint to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Blueprint saved: {filepath} ({len(self.objects)} objects)")


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
        img = Image.open(path).convert('RGB')

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
            rotation: Rotation = Rotation.VERTICAL
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
            rotation: Beam rotation (default: Rotation.VERTICAL)
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

        print(f"Converting {width}x{height} image ({width * height} pixels)...")

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
                    rotation,
                    color_linear
                )
                object_count += 1

        if skipped_count > 0:
            print(f"✓ Created {object_count} painted beams ({skipped_count} pixels skipped)")
        else:
            print(f"✓ Created {object_count} painted beams")

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


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to Satisfactory painted beam pixel art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                        # Full resolution (default, up to 4K)
  %(prog)s image.png -s 64x64               # Downsample to 64x64
  %(prog)s image.png -s "128 128"           # Downsample to 128x128 (space-separated)
  %(prog)s image.png -s 1920x1080           # Downsample to 1080p
  %(prog)s image.png -s 50%%              # Downsample to 50%% of original
  %(prog)s image.png -s 25%%              # Downsample to 25%% of original
  %(prog)s image.png -o art.json -n "Art"   # Custom output and name
  %(prog)s image.png --spacing 150          # Increase beam spacing
  %(prog)s image.png --filter-bg auto       # Auto-detect and remove background
  %(prog)s image.png --filter-bg corners --bg-tolerance 50  # Remove corner color
  %(prog)s image.png --orientation vertical # Vertical orientation (beams rotated 90° in Z)
  %(prog)s image.png --orientation horizontal  # Horizontal orientation (default)

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
    parser.add_argument('--base-z', type=float, default=1200.0,
                        help='Base Z height for beams (default: 1200)')
    parser.add_argument('--max-4k', action='store_true',
                        help='Enforce 4K resolution limit (3840x2160)')
    parser.add_argument('--filter-bg', type=str, choices=['auto', 'corners', 'brightness'],
                        help='Filter background: auto (corner color), corners (average corners), brightness (dark/bright)')
    parser.add_argument('--bg-tolerance', type=float, default=30.0,
                        help='Background color tolerance 0-255 (default: 30, lower=stricter)')
    parser.add_argument('-H', '--horizontal', action='store_true',
                        help='Use horizontal beam orientation (default: vertical)')
    parser.add_argument('--orientation', type=str, choices=['horizontal', 'vertical'], default='horizontal',
                        help='Beam orientation: horizontal (default) or vertical (rotated 90° in Z-axis)')

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
    converter = ImageToBlueprint(beam_spacing=args.spacing)

    # Determine rotation based on horizontal flag
    beam_rotation = Rotation.HORIZONTAL_0 if args.horizontal else Rotation.VERTICAL

    blueprint = converter.convert(
        args.image,
        name=args.name,
        target_size=target_size,
        base_z=args.base_z,
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
