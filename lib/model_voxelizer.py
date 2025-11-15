#!/usr/bin/env python3
"""
3D Model Voxelizer
Convert 3D models to voxelized painted beam blueprints
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import trimesh

from .blueprint import Blueprint, ObjectType, Rotation, Vector3


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
            use_vertex_colors: bool = True
    ) -> Blueprint:
        """
        Convert 3D model to voxelized blueprint.

        Args:
            model_path: Path to 3D model file
            name: Blueprint name (defaults to filename)
            max_dimension: Maximum model size in cm
            default_color: Default RGB color (0-1) if no vertex colors
            use_vertex_colors: Whether to sample colors from mesh

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
                Rotation.VERTICAL,
                color
            )

        print(f"✓ Created {len(blueprint.objects):,} painted beams")

        return blueprint
