#!/usr/bin/env python3
"""
Shared Blueprint Classes for Satisfactory
Core data structures for creating blueprint JSON files
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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
    """Container for blueprint data"""

    def __init__(self, name: str = "Generated Blueprint"):
        self.name = name
        self.objects: List[BlueprintObject] = []
        self.layers: List[Layer] = [Layer("Primary", z_offset=0, density=1.0)]
        self._object_counter = 0

    def add_layer(self, layer: Layer):
        """Add a dimensional layer"""
        self.layers.append(layer)

    def add_object(
            self,
            object_type: ObjectType,
            position: Vector3,
            rotation: Rotation,
            color_rgb: Optional[Tuple[float, float, float]] = None
    ):
        """Add an object to the blueprint"""
        obj = BlueprintObject(
            object_type=object_type,
            position=position,
            rotation=Quaternion.from_rotation(rotation),
            color_rgb=color_rgb,
            instance_id=self._object_counter
        )
        self.objects.append(obj)
        self._object_counter += 1

    def to_dict(self) -> Dict:
        """Convert to blueprint JSON format"""
        return {
            "header": {
                "saveHeaderVersion": 13,
                "saveVersion": 46,
                "buildVersion": 365306,
                "mapName": "Persistent_Level",
                "mapOptions": "",
                "sessionName": self.name,
                "playDurationSeconds": 0,
                "saveDateTime": 0,
                "sessionVisibility": 0,
                "editorObjectVersion": 0,
                "modMetadata": "",
                "isModdedSave": False,
                "saveIdentifier": ""
            },
            "objects": [obj.to_dict() for obj in self.objects]
        }

    def save(self, path: Path):
        """Save blueprint to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Saved blueprint: {path}")