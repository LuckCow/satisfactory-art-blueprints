#!/usr/bin/env python3
"""
Blueprint Generator - Image to Painted Beam Pixel Art
Convert images to Satisfactory blueprints using painted beams as pixels
"""

import argparse
from pathlib import Path
from PIL import Image

from lib.image_processor import ImageToBlueprint, parse_size_argument


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
    blueprint = converter.convert(
        args.image,
        name=args.name,
        target_size=target_size,
        base_z=args.base_z,
        filter_bg=args.filter_bg,
        bg_tolerance=args.bg_tolerance,
        orientation=args.orientation
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
