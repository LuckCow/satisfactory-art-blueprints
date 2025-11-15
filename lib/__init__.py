"""
Satisfactory Art Blueprints Library
Core modules for converting images and 3D models to Satisfactory blueprints
"""

from .blueprint import (
    ObjectType,
    ColorSwatch,
    Rotation,
    Vector3,
    Quaternion,
    Layer,
    BlueprintObject,
    Blueprint
)

from .image_processor import ImageToBlueprint, parse_size_argument
from .model_voxelizer import ModelVoxelizer

__all__ = [
    # Blueprint core
    'ObjectType',
    'ColorSwatch',
    'Rotation',
    'Vector3',
    'Quaternion',
    'Layer',
    'BlueprintObject',
    'Blueprint',
    # Image processing
    'ImageToBlueprint',
    'parse_size_argument',
    # 3D model voxelization
    'ModelVoxelizer',
]

__version__ = '2.0.0'
