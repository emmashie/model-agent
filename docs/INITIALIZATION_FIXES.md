# Initial Conditions Bug Fixes

## Problem Identified
The initial conditions file had unrealistic values:
- Temperature: up to 305°C (should be ~8-18°C for Monterey Bay)
- Salinity: -760 to 40 PSU (should be ~32-34 PSU, cannot be negative)
- U velocity: up to 131 m/s (should be ~-1 to 1 m/s)
- V velocity: up to -403 m/s (should be ~-1 to 1 m/s)

## Root Causes

### 1. Depth Coordinate Sign Convention Mismatch
**Issue**: ROMS uses negative z-coordinates (below sea surface), while GLORYS uses positive depths (downward from surface). The interpolation was using `-z_rho` which created the wrong sign.

**Fix in `llm_init_agent.py`**:
- Changed from `'z_rho': -z_rho` to `'z_rho': np.abs(z_rho)` (renamed to `z_rho_positive`)
- This ensures both GLORYS and ROMS depths are positive for proper interpolation

### 2. Incorrect Mask Broadcasting
**Issue**: In `initialization.py`, the mask was 2D `(ny, nx)` but being applied to 3D result `(nz, ny, nx)` without proper broadcasting, causing incorrect masking behavior.

**Fix in `model-tools/code/initialization.py`**:
```python
# Before:
result = np.where(mask, result, np.nan)

# After:  
mask_3d = mask[np.newaxis, :, :]  # Add depth dimension
result = np.where(mask_3d, result, np.nan)
```

### 3. Inconsistent Depth Array Usage in Barotropic Calculations
**Issue**: The `z_rho` array was being reused but had the wrong sign for `compute_uvbar` and `compute_w`, which expect positive depths.

**Fix in `llm_init_agent.py`**:
- Store `z_rho_positive` separately from the original `z_rho`
- Use `z_rho_positive` (properly transposed) for all barotropic calculations
- Renamed `z_rho_transposed` to `z_rho_for_uvbar` for clarity

## Files Modified

1. **`llm_init_agent.py`** (lines ~517-620):
   - Fixed depth coordinate sign for interpolation
   - Fixed depth array usage in barotropic velocity calculations

2. **`model-tools/code/initialization.py`** (lines ~307-314):
   - Fixed 2D mask broadcasting to 3D for proper land/sea masking

## Testing Recommendation

After these fixes, regenerate the initial conditions:

```python
from llm_complete_agent import ROMSCompleteSetupAgent

agent = ROMSCompleteSetupAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
)

# Re-run for Monterey Bay
result = agent.execute_workflow(
    "Create ROMS grid for Monterey Bay and initialize for March 24, 2016"
)
```

Expected value ranges after fixes:
- Temperature: ~8-18°C (typical for Monterey Bay)
- Salinity: ~32-34 PSU
- U/V velocities: ~-1 to 1 m/s
- Zeta: ~-0.5 to 0.5 m

## Technical Details

### ROMS Coordinate Conventions
- **h**: Positive values (depth at rho-points)
- **z_rho**: Negative values from compute_z (height below sea surface)
- **sigma coordinates**: Range from -1 (bottom) to 0 (surface)

### GLORYS Coordinate Conventions  
- **depth**: Positive values (depth below surface)
- **thetao, so, uo, vo**: Data indexed by positive depth

### Interpolation Requirements
- Source (GLORYS) depth: Positive, 1D array
- Target (ROMS) depth: Must be positive for interpolation, 3D array `(ny, nx, nz)`
- After interpolation, data has shape `(nz, ny, nx)` matching ROMS convention
