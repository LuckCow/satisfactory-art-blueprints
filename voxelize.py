#!/usr/bin/env python3
"""
3D Model Voxelizer - Surface Voxelization to Painted Beams
Converts 3D models to Satisfactory blueprints using painted beams as voxels
"""

import argparse
from pathlib import Path
from lib.model_voxelizer import ModelVoxelizer
from lib.blueprint import Rotation


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D models to Satisfactory voxel blueprints using painted beams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.stl                        # Convert with default settings (1m voxels)
  %(prog)s model.obj -s 50                  # Scale max dimension to 50m
  %(prog)s model.stl -s 100                 # Scale max dimension to 100m
  %(prog)s model.glb -o art.json -n "Art"   # Custom output and name
  %(prog)s model.stl --color 1 0.5 0        # Use orange color (RGB 0-1)

Supported formats:
  STL, OBJ, PLY, GLTF/GLB, FBX, DAE (Collada), 3DS, and more

Scaling:
  Each voxel is a 1m cube (painted beam)
  -s parameter sets the maximum dimension in meters
  Model is scaled proportionally to fit within this size
  Example: -s 50 scales a 500x400x300 model to 50x40x30m

Tips:
  - Models are centered at origin automatically
  - Surface voxelization creates hollow structures (efficient)
  - Vertex colors are sampled if available in the model
        """
    )

    parser.add_argument('model', type=Path, help='Input 3D model file')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file (default: <model>.json)')
    parser.add_argument('-n', '--name', help='Blueprint name (default: model filename)')
    parser.add_argument('-s', '--size', type=float, default=100.0,
                        help='Maximum model dimension in meters (default: 100m). Model is scaled proportionally.')
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
        # Each voxel is 1m (100cm internally for Satisfactory units)
        voxelizer = ModelVoxelizer(voxel_size=1.0)

        # Determine rotation based on horizontal flag
        beam_rotation = Rotation.HORIZONTAL_0 if args.horizontal else Rotation.VERTICAL

        blueprint = voxelizer.convert(
            args.model,
            name=args.name,
            max_dimension=args.size,
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
        print(f"   Voxel size: 1.0m (each beam is a 1m cube)")
        print(f"   Max size:   {args.size}m")
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
