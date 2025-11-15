#!/usr/bin/env python3
"""
3D Model Voxelizer - Surface Voxelization to Painted Beams
Converts 3D models to Satisfactory blueprints using painted beams as voxels
"""

import argparse
from pathlib import Path


class ObjectType(Enum):
    """Satisfactory object types"""
    BEAM_PAINTED = "Beam_Painted"

    @property
    def path(self) -> str:
        """Get the full Unreal Engine path for this object"""
        return "/Game/FactoryGame/Prototype/Buildable/Beams/Build_Beam_Painted.Build_Beam_Painted_C"

    @property
    def recipe_path(self) -> str:
        """Get the recipe path for this object"""
        return "/Game/FactoryGame/Prototype/Buildable/Beams/Recipe_Beam_Painted.Recipe_Beam_Painted_C"


class Rotation(Enum):
    """Common rotation presets for painted beams"""
    HORIZONTAL_0 = (0.0, 0.0, 0.0, 1.0)  # Facing forward
    HORIZONTAL_90 = (0.0, 0.0, 0.707107, 0.707107)  # Rotated 90°
    HORIZONTAL_180 = (0.0, 0.0, 1.0, 0.0)  # Facing back
    HORIZONTAL_270 = (0.0, 0.0, -0.707107, 0.707107)  # Rotated 270°
    VERTICAL = (-0.5, 0.5, -0.5, -0.5)  # Standing upright

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


class BlueprintObject:
    """A single object in the blueprint"""

    def __init__(
            self,
            object_type: ObjectType,
            position: Vector3,
            rotation: Quaternion,
            color_rgb: Optional[Tuple[float, float, float]] = None,
            instance_id: int = 0
    ):
        self.object_type = object_type
        self.position = position
        self.rotation = rotation
        self.color_rgb = color_rgb
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
                },
                "mLength": {
                    "type": "FloatProperty",
                    "ueType": "FloatProperty",
                    "name": "mLength",
                    "value": 100
                },
                "mColorSlot": {
                    "type": "ByteProperty",
                    "ueType": "ByteProperty",
                    "name": "mColorSlot",
                    "value": {
                        "type": "None",
                        "value": 255
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
                                            "r": r * 0.5,
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
    """3D voxel blueprint"""

    def __init__(self, name: str = "VoxelArt"):
        self.name = name
        self.objects: List[BlueprintObject] = []
        self.next_id = 2147483647  # Start from max int32
        self.default_rotation = Rotation.VERTICAL

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
        """Calculate blueprint dimensions in foundation units (800cm)"""
        if not self.objects:
            return Vector3(4, 4, 4)

        positions = [obj.position for obj in self.objects]
        min_x = min(p.x for p in positions)
        max_x = max(p.x for p in positions)
        min_y = min(p.y for p in positions)
        max_y = max(p.y for p in positions)
        min_z = min(p.z for p in positions)
        max_z = max(p.z for p in positions)

        # Calculate span in each dimension
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z

        # Convert to foundation units (800cm per foundation)
        foundation_size = 800.0
        dim_x = max(4, int(np.ceil(span_x / foundation_size)) + 1)
        dim_y = max(4, int(np.ceil(span_y / foundation_size)) + 1)
        dim_z = max(4, int(np.ceil(span_z / foundation_size)) + 1)

        return Vector3(dim_x, dim_y, dim_z)

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
                "itemCosts": []
            },
            "config": {
                "configVersion": 4,
                "description": f"Voxelized 3D model - {len(self.objects)} beams",
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


class ModelVoxelizer:
    """Convert 3D models to voxelized painted beam blueprints"""

    def __init__(self, voxel_size: float = 100.0):
        """
        Initialize voxelizer

        Args:
            voxel_size: Size of each voxel in cm (default: 100cm = 1m)
        """
        self.voxel_size = voxel_size

    def surface_voxelize(self, mesh: trimesh.Trimesh, voxel_size: float) -> np.ndarray:
        """
        Surface-only voxelization for hollow structures.
        Samples points on the mesh surface and snaps to voxel grid.

        Args:
            mesh: Trimesh mesh object
            voxel_size: Size of voxels in world units

        Returns:
            Array of voxel positions (Nx3)
        """
        # Calculate number of samples based on surface area
        surface_area = mesh.area
        point_density = surface_area / (voxel_size ** 2)
        num_samples = int(point_density * 4)  # Oversample for coverage
        num_samples = max(1000, num_samples)  # Minimum 1000 samples

        print(f"  Sampling {num_samples:,} points on surface...")

        # Sample points uniformly on the surface
        points, _ = trimesh.sample.sample_surface(mesh, count=num_samples)

        # Snap points to voxel grid
        voxel_coords = np.round(points / voxel_size).astype(int)

        # Remove duplicate voxels
        unique_voxels = np.unique(voxel_coords, axis=0)

        # Convert back to world coordinates (centered on voxels)
        return unique_voxels * voxel_size

    def sample_vertex_colors(
            self,
            mesh: trimesh.Trimesh,
            voxel_positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Sample vertex colors from mesh at voxel positions.

        Args:
            mesh: Trimesh mesh with vertex colors
            voxel_positions: Voxel positions (Nx3)

        Returns:
            RGB colors (Nx3) in 0-1 range, or None if no colors
        """
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            return None

        # Find closest vertices to voxel positions
        closest_points, distances, triangle_ids = mesh.nearest.on_surface(voxel_positions)

        # Get vertex colors (RGBA)
        vertex_colors = mesh.visual.vertex_colors

        # Sample colors at closest points
        colors = []
        for tid in triangle_ids:
            # Get triangle vertices
            tri_verts = mesh.faces[tid]
            # Average the vertex colors of the triangle
            tri_colors = vertex_colors[tri_verts]
            avg_color = np.mean(tri_colors, axis=0)[:3]  # RGB only
            colors.append(avg_color / 255.0)  # Normalize to 0-1

        return np.array(colors)

    def load_and_prepare_mesh(
            self,
            model_path: Path,
            max_dimension: float = 10000.0
    ) -> trimesh.Trimesh:
        """
        Load 3D model and prepare for voxelization.

        Args:
            model_path: Path to 3D model file
            max_dimension: Maximum size in cm (default: 100m)

        Returns:
            Prepared trimesh object
        """
        print(f"Loading model: {model_path}")
        mesh = trimesh.load(model_path, force='mesh')

        # Handle multi-mesh files
        if isinstance(mesh, trimesh.Scene):
            print("  Converting scene to single mesh...")
            mesh = mesh.dump(concatenate=True)

        # Center the mesh
        mesh.apply_translation(-mesh.centroid)

        # Get current bounds
        bounds = mesh.bounds
        current_size = bounds[1] - bounds[0]
        current_max = np.max(current_size)

        print(f"  Original size: {current_size[0]:.1f} x {current_size[1]:.1f} x {current_size[2]:.1f} units")

        # Scale to fit within max_dimension
        if current_max > max_dimension:
            scale_factor = max_dimension / current_max
            mesh.apply_scale(scale_factor)
            new_size = mesh.bounds[1] - mesh.bounds[0]
            print(
                f"  Scaled to: {new_size[0]:.1f} x {new_size[1]:.1f} x {new_size[2]:.1f} cm (factor: {scale_factor:.3f})")
        else:
            print(f"  Size in cm: {current_size[0]:.1f} x {current_size[1]:.1f} x {current_size[2]:.1f}")

        # Calculate mesh statistics
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces: {len(mesh.faces):,}")
        print(f"  Surface area: {mesh.area:.1f} cm²")

        return mesh

    def convert(
            self,
            model_path: Path,
            name: Optional[str] = None,
            max_dimension: float = 10000.0,
            default_color: Optional[Tuple[float, float, float]] = None,
            use_vertex_colors: bool = True,
            rotation: Rotation = Rotation.VERTICAL
    ) -> Blueprint:
        """
        Convert 3D model to voxelized blueprint.

        Args:
            model_path: Path to 3D model file
            name: Blueprint name (defaults to filename)
            max_dimension: Maximum model size in cm
            default_color: Default RGB color (0-1) if no vertex colors
            use_vertex_colors: Whether to sample colors from mesh
            rotation: Beam rotation (default: Rotation.VERTICAL)

        Returns:
            Blueprint object
        """
        # Load and prepare mesh
        mesh = self.load_and_prepare_mesh(model_path, max_dimension)

        # Voxelize surface
        print(f"\nVoxelizing with {self.voxel_size}cm voxels...")
        voxel_positions = self.surface_voxelize(mesh, self.voxel_size)

        print(f"✓ Generated {len(voxel_positions):,} voxels")

        # Sample colors if available
        colors = None
        if use_vertex_colors:
            colors = self.sample_vertex_colors(mesh, voxel_positions)
            if colors is not None:
                print(f"✓ Sampled vertex colors")

        # Create blueprint
        name = name or model_path.stem
        blueprint = Blueprint(name)

        print(f"\nCreating blueprint objects...")

        # Add voxels as painted beams
        for i, pos in enumerate(voxel_positions):
            # Get color for this voxel
            if colors is not None:
                color = tuple(colors[i])
            elif default_color is not None:
                color = default_color
            else:
                color = None

            # Create beam at voxel position
            blueprint.add_object(
                ObjectType.BEAM_PAINTED,
                Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
                rotation,
                color
            )

        print(f"✓ Created {len(blueprint.objects):,} painted beams")

        return blueprint
from lib.model_voxelizer import ModelVoxelizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D models to Satisfactory voxel blueprints using painted beams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.stl                        # Convert with default settings (100cm voxels)
  %(prog)s model.obj -s 50                  # Use 50cm voxels (higher detail)
  %(prog)s model.stl -s 200                 # Use 200cm voxels (lower detail)
  %(prog)s model.glb -o art.json -n "Art"   # Custom output and name
  %(prog)s model.ply --max-size 5000        # Limit model to 50m
  %(prog)s model.stl --color 1 0.5 0        # Use orange color (RGB 0-1)

Supported formats:
  STL, OBJ, PLY, GLTF/GLB, FBX, DAE (Collada), 3DS, and more

Voxel sizes:
  - 50cm: High detail (more beams)
  - 100cm: Default (1m voxels)
  - 200cm: Low detail (fewer beams)

Tips:
  - Start with larger voxel sizes (200cm) for testing
  - Models are centered at origin automatically
  - Surface voxelization creates hollow structures (efficient)
  - Vertex colors are sampled if available in the model
        """
    )

    parser.add_argument('model', type=Path, help='Input 3D model file')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file (default: <model>.json)')
    parser.add_argument('-n', '--name', help='Blueprint name (default: model filename)')
    parser.add_argument('-s', '--voxel-size', type=float, default=100.0,
                        help='Voxel size in cm (default: 100)')
    parser.add_argument('--max-size', type=float, default=10000.0,
                        help='Maximum model dimension in cm (default: 10000 = 100m)')
    parser.add_argument('--color', type=float, nargs=3, metavar=('R', 'G', 'B'),
                        help='Default RGB color in 0-1 range (e.g., 1 0.5 0 for orange)')
    parser.add_argument('--no-vertex-colors', action='store_true',
                        help='Ignore vertex colors from model')
    parser.add_argument('-H', '--horizontal', action='store_true',
                        help='Use horizontal beam orientation (default: vertical)')

    args = parser.parse_args()

    # Validate input
    if not args.model.exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return 1

    # Validate color
    default_color = None
    if args.color:
        r, g, b = args.color
        if not (0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1):
            print(f"❌ Error: Color values must be between 0 and 1")
            return 1
        default_color = (r, g, b)

    # Determine output path
    output_path = args.output or args.model.with_suffix('.json')

    print("=" * 60)
    print("3D Model Voxelizer - Surface Voxelization")
    print("=" * 60)
    print()

    try:
        # Convert model to blueprint
        voxelizer = ModelVoxelizer(voxel_size=args.voxel_size)

        # Determine rotation based on horizontal flag
        beam_rotation = Rotation.HORIZONTAL_0 if args.horizontal else Rotation.VERTICAL

        blueprint = voxelizer.convert(
            args.model,
            name=args.name,
            max_dimension=args.max_size,
            default_color=default_color,
            use_vertex_colors=not args.no_vertex_colors,
            rotation=beam_rotation
        )

        # Save blueprint
        print(f"\nSaving blueprint...")
        blueprint.save(output_path)

        # Display summary
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print()
        print("=" * 60)
        print("📊 Summary:")
        print("=" * 60)
        print(f"   Input:      {args.model}")
        print(f"   Output:     {output_path}")
        print(f"   Voxels:     {len(blueprint.objects):,} beams")
        print(f"   Voxel size: {args.voxel_size} cm")
        print(f"   File size:  {file_size_mb:.2f} MB")
        print("=" * 60)
        print()
        print("✓ Done! Import the JSON file using the blueprint encoding script.")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
