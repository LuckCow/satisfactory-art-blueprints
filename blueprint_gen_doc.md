# CLI Usage Guide - Blueprint Generator v2

## Quick Reference

```bash
# Full resolution (default - up to 4K supported)
python blueprint_gen_v2.py image.png

# Downsample with different formats
python blueprint_gen_v2.py image.png -s 64x64        # WxH format
python blueprint_gen_v2.py image.png -s "128 128"    # Space-separated (needs quotes)
python blueprint_gen_v2.py image.png -s 50%          # Percentage

# Custom output and settings
python blueprint_gen_v2.py image.png -o art.json -n "MyArt" --spacing 150
```

## Resolution Formats

### 1. Full Resolution (Default)
```bash
python blueprint_gen_v2.py photo.jpg
```
- Uses original image dimensions
- Supports up to 4K (3840×2160 = 8.3M pixels)
- No `-s` flag needed

### 2. WxH Format
```bash
python blueprint_gen_v2.py photo.jpg -s 1920x1080
python blueprint_gen_v2.py photo.jpg -s 64x64
python blueprint_gen_v2.py photo.jpg -s 3840x2160
```
- Most common format
- Width × Height separated by lowercase 'x'

### 3. Space-Separated Format
```bash
python blueprint_gen_v2.py photo.jpg -s "128 128"
python blueprint_gen_v2.py photo.jpg -s "1920 1080"
```
- Width and height separated by space
- **Requires quotes** when using in shell

### 4. Percentage Format
```bash
python blueprint_gen_v2.py photo.jpg -s 50%     # Half size
python blueprint_gen_v2.py photo.jpg -s 25%     # Quarter size
python blueprint_gen_v2.py photo.jpg -s 10%     # 10% of original
```
- Relative to original dimensions
- Maintains aspect ratio
- Great for quick downsampling

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output file path | `<image>.json` |
| `-n`, `--name` | Blueprint name | Image filename |
| `-s`, `--size` | Target resolution | Full resolution |
| `--spacing` | Beam spacing in cm | 100 |
| `--base-z` | Base Z height in cm | 1200 |
| `--max-4k` | Enforce 4K limit | Off |

## Examples

### Small Pixel Art
```bash
# Perfect for retro-style art
python blueprint_gen_v2.py sprite.png -s 16x16
python blueprint_gen_v2.py icon.png -s 32x32
```

### Medium Artwork
```bash
# Good balance of detail and size
python blueprint_gen_v2.py artwork.png -s 128x128
python blueprint_gen_v2.py photo.jpg -s 256x256
```

### Large Scenes
```bash
# Detailed landscapes or photos
python blueprint_gen_v2.py landscape.jpg -s 512x512
python blueprint_gen_v2.py panorama.jpg -s 1920x1080
```

### Downsampling Large Images
```bash
# Quick 50% reduction
python blueprint_gen_v2.py 4k_photo.jpg -s 50%

# Aggressive downsampling for huge images
python blueprint_gen_v2.py gigapixel.jpg -s 10%

# Specific target size
python blueprint_gen_v2.py 8k_image.jpg -s 1920x1080
```

### Custom Spacing and Height
```bash
# Tighter grid (smaller beams visually)
python blueprint_gen_v2.py art.png -s 64x64 --spacing 50

# More spaced out (easier to see individual beams)
python blueprint_gen_v2.py art.png -s 64x64 --spacing 200

# Different base height
python blueprint_gen_v2.py art.png -s 64x64 --base-z 1500
```

## Output Information

The CLI provides detailed feedback:

```
📐 Original image size: 1920x1080 (2,073,600 pixels)
🎯 Target size: 64x64 (4,096 pixels)
   (99.8% reduction)

🔧 Converting image to painted beam blueprint...
Resized to 64x64
Converting 64x64 image (4096 pixels)...
✓ Created 4096 painted beams
✓ Blueprint saved: output.json (4096 objects)

📊 Summary:
   Input:  photo.jpg
   Output: output.json
   Beams:  4,096
   Size:   8.37 MB
```

## Resolution Recommendations

| Use Case | Recommended Size | Pixel Count |
|----------|-----------------|-------------|
| Icons/Sprites | 16×16 to 32×32 | 256-1,024 |
| Small Art | 64×64 to 128×128 | 4K-16K |
| Medium Art | 256×256 to 512×512 | 65K-262K |
| Large Scenes | 1024×1024 to 1920×1080 | 1M-2M |
| Maximum | 3840×2160 (4K) | 8.3M |

## File Size Estimates

Approximate blueprint JSON file sizes:
- 16×16 (256 pixels) → ~0.5 MB
- 64×64 (4K pixels) → ~8 MB
- 128×128 (16K pixels) → ~33 MB
- 256×256 (65K pixels) → ~133 MB
- 512×512 (262K pixels) → ~535 MB
- 1920×1080 (2M pixels) → ~4.2 GB

**Note**: Each painted beam creates about 2KB of JSON data.

## 4K Support

The CLI supports images up to 4K resolution:
- **3840×2160 pixels = 8,294,400 painted beams**
- Blueprint file size: ~17 GB
- Processing time: Several minutes
- Memory required: 16GB+ RAM recommended

For 4K images, consider downsampling:
```bash
# 50% reduction → 1920×1080
python blueprint_gen_v2.py 4k.jpg -s 50%

# Or specify exact size
python blueprint_gen_v2.py 4k.jpg -s 1920x1080
```

## Error Handling

### Image Not Found
```bash
❌ Error: Image not found: nonexistent.png
```

### Invalid Resolution Format
```bash
❌ Error: Invalid size format: 64-64. Use WxH (e.g., 64x64), 'W H' (e.g., '64 64'), or percentage (e.g., 50%)
```

### Large Image Warning
For images over 1M pixels without downsampling:
```bash
⚠️  Large resolution: 2,073,600 beams will be created
    Blueprint file will be approximately 4138 MB
```

## Testing

Run the comprehensive test suite:
```bash
python test_cli_resolution.py
```

This creates test images and demonstrates all resolution formats.

## Tips

1. **Start small**: Test with 64×64 before going larger
2. **Use percentages**: Quick way to downsample large images
3. **Check file size**: Large blueprints may be slow to load in-game
4. **Spacing matters**: Default 100cm works for most cases
5. **Full resolution**: Great for small images (<256×256)