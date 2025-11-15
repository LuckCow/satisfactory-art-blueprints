# Z-Offset Feature Test Results

## Test Environment
- **Test Image:** test_image.png (16x16 pixels, 256 total pixels)
- **Feature:** Condensed rendering with z-clipping
- **Default z-offset:** 0.001 cm

## Test Results Summary

### ✅ Test 1: Basic Condensed Rendering (Default Settings)
- **Command:** `--condensed` (2x2 multiplier, 0.001 offset)
- **Expected:** 1,024 beams (4 per pixel)
- **Result:** ✓ PASS - 1,024 beams created, 4.17 MB file

### ✅ Test 2: Different Multiplier Values
| Multiplier | Beams/Pixel | Total Beams | File Size | Status |
|------------|-------------|-------------|-----------|--------|
| 2x2        | 4           | 1,024       | 4.17 MB   | ✓ PASS |
| 3x3        | 9           | 2,304       | 9.41 MB   | ✓ PASS |
| 4x4        | 16          | 4,096       | 16.67 MB  | ✓ PASS |
| 5x5        | 25          | 6,400       | 26.05 MB  | ✓ PASS |

### ✅ Test 3: Different Z-Offset Values
| Z-Offset | Beams | File Size | Status |
|----------|-------|-----------|--------|
| 0.0001   | 1,024 | 4.17 MB   | ✓ PASS |
| 0.001    | 1,024 | 4.17 MB   | ✓ PASS |
| 0.01     | 1,024 | 4.17 MB   | ✓ PASS |
| 0.1      | 1,024 | 4.17 MB   | ✓ PASS |

### ✅ Test 4: Z-Clipping Algorithm Verification

#### 2x2 Grid (No Z-Clipping Expected)
- All 4 beams are corners (edge_dist=0)
- All at same z-height: **1200.0**
- ✓ PASS - Correct behavior for 2x2 grid

#### 3x3 Grid (2 Z-Levels)
- **Level 0** (z=1200.0): 8 beams - corners and edges
- **Level 1** (z=1199.999): 1 beam - center
- Z-offset: 0.001 cm between levels
- ✓ PASS - Correct 2-level clipping

#### 4x4 Grid (2 Z-Levels)
- **Level 0** (z=1200.0): 12 beams - outer ring
- **Level 1** (z=1199.999): 4 beams - inner beams
- ✓ PASS - Correct layering

#### 5x5 Grid (3 Z-Levels)
- **Level 0** (z=1200.000): 16 beams - outermost
- **Level 1** (z=1199.999): 8 beams - middle ring
- **Level 2** (z=1199.998): 1 beam - center
- ✓ PASS - Perfect 3-level clipping!

### ✅ Test 5: Comparison with Standard Rendering

| Mode | Beams | File Size | Beams/Pixel |
|------|-------|-----------|-------------|
| Standard (no condensed) | 256 | 1.04 MB | 1x |
| Condensed 2x2 (default) | 1,024 | 4.17 MB | 4x |
| Condensed 3x3 | 2,304 | 9.41 MB | 9x |
| Condensed 4x4 | 4,096 | 16.67 MB | 16x |
| Condensed 5x5 | 6,400 | 26.05 MB | 25x |

**Observations:**
- ✓ Condensed rendering increases detail by packing multiple beams per pixel
- ✓ File size scales linearly with beam count
- ✓ Z-offset allows beams to clip together in same space
- ✓ Higher multipliers = higher detail but larger files

### ✅ Test 6: Edge Cases

#### 1x1 Multiplier
- **Result:** 256 beams (equivalent to standard rendering)
- **Status:** ✓ PASS - Handles degenerate case correctly

#### 5x5 Multiplier
- **Result:** 6,400 beams with 3 distinct z-levels
- **Status:** ✓ PASS - Handles larger multipliers correctly

## Key Findings

1. **Z-Clipping Algorithm Works Correctly**
   - Outer corners are always highest (edge_dist=0)
   - Inner beams progressively lower based on distance from edges
   - Formula: `edge_dist = min(i, multiplier-1-i, j, multiplier-1-j)`
   - Z-depth: `z_depth = edge_dist * z_offset`

2. **Feature Parameters**
   - `--condensed`: Enable condensed rendering mode
   - `--cr-multiplier N`: Set NxN grid (default: 2)
   - `--cr-z-offset OFFSET`: Set z-offset in cm (default: 0.001)

3. **Performance Characteristics**
   - Beam count = pixels × (multiplier²)
   - File size ≈ beam_count × 4KB
   - Z-levels = ⌈multiplier/2⌉ for square grids

## Conclusion

**All tests PASSED! ✅**

The z-offset feature in condensed rendering mode is working correctly:
- Creates proper multi-layer beam structures
- Z-clipping algorithm produces expected layering
- Edge cases handled gracefully
- File sizes scale predictably

The feature allows users to create higher-detail blueprints in the same physical space by packing multiple beams per pixel with tiny z-offsets for clipping.
