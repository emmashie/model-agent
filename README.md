# ROMS LLM Agents - Complete Model Setup Automation

Intelligent agents for automating ROMS (Regional Ocean Modeling System) model setup using Large Language Models (LLMs) and the model-tools library.

## Overview

Three LLM-powered agents provide a complete workflow from natural language request to ready-to-run ROMS model:

1. **`llm_grid_agent.py`** - Intelligent grid generation with parameter suggestions
2. **`llm_init_agent.py`** - Initial conditions generation
3. **`llm_complete_agent.py`** - Combined workflow (recommended)

## Features

### 🤖 LLM-Powered Natural Language Understanding
- Parse natural language requests for model setup
- Extract region bounds, dates, parameters, and **simulation objectives** automatically
- Understand common location names (e.g., "Chesapeake Bay", "Gulf of Maine", "Gulf Stream")
- **NEW**: Identify simulation goals (e.g., "submesoscale resolving", "coastal dynamics")
- Intelligent fallback to regex parsing when LLM unavailable

### 🧠 Intelligent Parameter Suggestion
- **Context-aware recommendations** based on:
  - Simulation objectives (submesoscale, mesoscale, coastal, etc.)
  - Regional characteristics (depth, dynamics, scales)
  - ROMS S-coordinate vertical stretching knowledge
  - Horizontal resolution requirements for different phenomena
- Suggests complete grid configuration with scientific reasoning
- Users can accept suggestions or customize each parameter
- If goals not specified, interactively asks about simulation objectives

### 📐 Grid Generation
1. Parse region bounds and simulation goals from natural language
2. **Intelligently suggest appropriate grid parameters** or prompt for them:
   - Horizontal resolution (dx, dy in degrees)
   - Vertical levels and stretching (N_layers, theta_s, theta_b, hc)
   - Bathymetry smoothing (initial_smooth_sigma, hmin, rx0_thresh, max_iter, smooth_sigma, buffer_size)
3. Download SRTM15+ bathymetry data
4. Generate ROMS-compliant grid with proper staggering
5. Apply iterative smoothing to meet steepness criteria
6. Create visualization plots

### 🌊 Initial Conditions Generation
1. Parse initialization time from natural language
2. Interactive prompts for initialization parameters:
   - Date/time specification
   - Data source (Copernicus Marine API or NetCDF file)
   - Variable mapping and fill values
3. Load GLORYS ocean data
4. Interpolate to ROMS grid (temperature, salinity, velocities, SSH)
5. Compute derived variables (barotropic velocities, vertical velocity)
6. Create ROMS initial conditions file

### 🔗 Combined Workflow
- Seamless integration of grid generation and initialization
- Single natural language request → complete model setup
- Option to skip initial conditions if only grid needed
- Can use existing grid for initialization

## Quick Start

### Installation

```bash
# Install dependencies
pip install openai==2.6.1 xarray numpy scipy matplotlib cmocean cartopy

# Set API key (or use hardcoded key)
export LLM_API_KEY="your-api-key-here"
```

### Complete Workflow (Recommended)

```python
from llm_complete_agent import ROMSCompleteSetupAgent

# Initialize combined agent
agent = ROMSCompleteSetupAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
)

# Run complete workflow with natural language
result = agent.execute_workflow(
    "Create a ROMS setup for Chesapeake Bay from lat 36.5-39.5, lon -77.5 to -75.5, "
    "initialized for January 1, 2024"
)

# Access generated files
print(f"Grid: {result['files']['grid']}")
print(f"Initial conditions: {result['files']['initial_conditions']}")
```

### Grid Only

```python
from llm_grid_agent import ROMSGridAgent

agent = ROMSGridAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
)

result = agent.execute_workflow("Create a grid for Chesapeake Bay")
print(f"Grid file: {result['grid_file']}")
```

### Initial Conditions Only

```python
from llm_init_agent import ROMSInitAgent

agent = ROMSInitAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
    grid_file="/path/to/roms_grid.nc"
)

result = agent.execute_workflow("Initialize for January 1, 2024")
print(f"Initial conditions: {result['init_file']}")
```

## Example Natural Language Requests

### Complete Setup
- "Create a ROMS setup for the US East Coast from latitude 35 to 42 and longitude -75 to -65, initialized for January 1, 2024"
- "Set up model for Chesapeake Bay, initialize for start of 2024"
- "Generate grid and initial conditions for Gulf of Maine, initialize mid-2023"
- **"I want to set up a submesoscale resolving grid of the Gulf Stream and initialize for July 2023"**
- **"Create a coastal upwelling model for the California coast, initialized for summer 2024"**

### Grid Generation with Simulation Goals
- **"Create a submesoscale-resolving grid of the Gulf Stream"** → Agent suggests ~1km resolution, 50 levels, high surface stretching
- **"Set up a mesoscale model for Chesapeake Bay"** → Agent suggests ~5km resolution, 40 levels, balanced stretching  
- **"Generate a coastal dynamics grid for Puget Sound"** → Agent suggests ~2km resolution, coastal-appropriate vertical structure
- "Create a grid for the US East Coast from latitude 35 to 42 and longitude -75 to -65" → Agent asks about simulation goals

### Grid Generation (Location Only)
- "Create a grid for Chesapeake Bay" → Agent will ask: "What phenomena do you want to resolve?"
- "Generate a grid for the Gulf of Maine region" → Agent prompts for simulation objectives
- "Set up a model for lat 35-42, lon -75 to -65" → Agent inquires about goals before suggesting parameters

### Initial Conditions
- "Initialize for January 1, 2024"
- "Create initial conditions for start of 2024 using API"
- "Initialize model for mid-July 2023"

## Intelligent Parameter Suggestion Workflow

The grid agent now uses domain knowledge to suggest appropriate parameters:

### 1. Simulation Goal Classification
The LLM identifies simulation type from your request:
- **Submesoscale-resolving**: O(1km) resolution, fine vertical structure
- **Mesoscale-resolving**: O(5-10km) resolution, standard vertical structure  
- **Coastal/Shelf**: O(2-5km) resolution, balanced surface/bottom stretching
- **Basin-scale**: O(10-25km) resolution, efficient vertical structure

### 2. ROMS S-Coordinate Knowledge
Parameters suggested based on [ROMS vertical stretching](https://www.myroms.org/wiki/Vertical_S-coordinate):
- **theta_s** (0-10): Surface stretching - higher for surface-focused phenomena
- **theta_b** (0-4): Bottom stretching - higher for bottom boundary layer processes
- **hc** (negative depth): Transition depth between sigma and z-like coordinates

### 3. Regional Characteristics
Considers:
- Expected bathymetry (shallow shelf vs deep ocean)
- Typical dynamics (upwelling, fronts, eddies)
- Steepness requirements for smoothing

### Example Interaction

```
User: "I want to set up a submesoscale resolving grid of the Gulf Stream"

Agent: 
📊 Recommendation: Submesoscale-resolving configuration for Gulf Stream
   - ~1km horizontal resolution to capture O(1-10km) eddies and fronts
   - 50 vertical levels with strong surface stretching (theta_s=7) for 
     mixed layer dynamics
   - Moderate bottom stretching (theta_b=2) for slope currents
   - Critical depth hc=-250m appropriate for shelf-slope region

Suggested Parameters:
  Horizontal Resolution:
    • dx: 0.0100° (~1.1 km)
    • dy: 0.0100° (~1.1 km)
  Vertical Configuration:
    • N_layers:  50
    • theta_s:   7.0 (surface stretching)
    • theta_b:   2.0 (bottom stretching)
    • hc:        -250.0 m (critical depth)
  [... smoothing parameters ...]

Use these suggested parameters? [Y/n]
```

Choose:
- **Y** → Use suggested configuration (recommended for most users)
- **n** → Customize each parameter with interactive prompts

## Interactive Parameter Configuration

After parsing your request, each agent will interactively prompt for parameters:

### Grid Agent Prompts
- **Horizontal resolution**: dx, dy in degrees (e.g., 0.01 = ~1 km)
- **Vertical levels**: N_layers (default: 50)
- **Vertical stretching**: theta_s (default: 5), theta_b (default: 0.5), hc (default: -500 m)
- **Bathymetry smoothing**: initial_smooth_sigma, hmin, rx0_thresh, max_iter, smooth_sigma, buffer_size

### Initial Conditions Agent Prompts
- **Initialization time**: Date/time in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format
- **Data source**: Copernicus Marine API or local NetCDF file
- **NetCDF path**: If using local file
- **Time buffer**: Days before/after init_time for API downloads (default: 1)

## Configuration Details

### Grid Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dx_deg`, `dy_deg` | Horizontal resolution (degrees) | Prompted |
| `N_layers` | Number of vertical levels | 50 |
| `theta_s` | Surface stretching parameter | 5 |
| `theta_b` | Bottom stretching parameter | 0.5 |
| `hc` | Critical depth (m, negative) | -500 |
| `initial_smooth_sigma` | Initial Gaussian smoothing | 10 |
| `hmin` | Minimum depth (m, negative) | -5 |
| `rx0_thresh` | rx0 threshold for smoothing | 0.2 |
| `max_iter` | Maximum smoothing iterations | 10 |
| `smooth_sigma` | Iterative smoothing strength | 6 |
| `buffer_size` | Buffer around steep regions | 5 |

### Initialization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_time` | Initialization date/time | Prompted |
| `use_api` | Use Copernicus Marine API | True |
| `netcdf_path` | Path to NetCDF file | None |
| `time_buffer_days` | Days around init_time | 1 |

## Output Files

### Grid Generation
- `roms_grid.nc` - ROMS grid NetCDF file with:
  - Coordinates (lat/lon for rho, u, v, psi grids)
  - Metrics (pm, pn, f - Coriolis parameter)
  - Bathymetry (h) and masks (mask_rho, mask_u, mask_v, mask_psi)
  - Vertical coordinates (s_rho, s_w, Cs_r, Cs_w)
  - Grid metrics (dndx, dmde, angle)
- `roms_grid_bathymetry.png` - Bathymetry visualization
- `downloaded_bathy.nc` - Downloaded bathymetry data

### Initial Conditions
- `initial_conditions.nc` - ROMS initial conditions file with:
  - Temperature, salinity
  - 3D velocities (u, v, w)
  - Barotropic velocities (ubar, vbar)
  - Sea surface height (zeta)
  - Time coordinates

## Known Regions

The LLM recognizes common ocean regions:
- **US East Coast**: 24°N to 45°N, -81°W to -65°W
- **Gulf of Mexico**: 18°N to 31°N, -98°W to -80°W
- **California Coast**: 32°N to 42°N, -125°W to -117°W
- **Chesapeake Bay**: 36.5°N to 39.5°N, -77.5°W to -75.5°W
- **Gulf of Maine**: 41°N to 45°N, -71°W to -66°W
- **Florida Keys**: 24.5°N to 25.5°N, -82°W to -80°W
- **Cape Cod**: 41°N to 42.5°N, -71°W to -69.5°W

## Command Line Usage

```bash
# Complete workflow
python llm_complete_agent.py

# Grid generation only
python llm_grid_agent.py

# Initial conditions only  
python llm_init_agent.py
```

## API Configuration

Uses OpenAI-compatible API:
- **Base URL**: `https://ai-incubator-api.pnnl.gov`
- **Default Model**: `claude-haiku-4-5-20251001-v1-birthright`
- **API Key**: Set via `LLM_API_KEY` environment variable

To use a different model:

```python
agent = ROMSCompleteSetupAgent(
    model_tools_path="/path/to/model-tools",
    model="gpt-4o-birthright"  # Or any available model
)
```

## Linking Agents

The `ROMSCompleteSetupAgent` demonstrates how to link the grid and initialization agents:

```python
# The combined agent:
# 1. Creates a grid using ROMSGridAgent
# 2. Passes the grid file to ROMSInitAgent  
# 3. Generates initial conditions

agent = ROMSCompleteSetupAgent(model_tools_path="/path/to/model-tools")
result = agent.execute_workflow("Complete setup request")

# Access both results
grid_result = result['grid_result']
init_result = result['init_result']
```

You can also manually link them:

```python
from llm_grid_agent import ROMSGridAgent
from llm_init_agent import ROMSInitAgent

# Step 1: Generate grid
grid_agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
grid_result = grid_agent.execute_workflow("Create grid for Chesapeake Bay")

# Step 2: Generate initial conditions using the grid
init_agent = ROMSInitAgent(
    model_tools_path="/path/to/model-tools",
    grid_file=grid_result['grid_file']
)
init_result = init_agent.execute_workflow("Initialize for January 1, 2024")

print(f"Grid: {grid_result['grid_file']}")
print(f"Initial conditions: {init_result['init_file']}")
```

## Troubleshooting

### LLM Not Available
Agents fall back to basic regex parsing if OpenAI library unavailable or API key invalid. You'll still be prompted for all parameters.

### Copernicus Marine API
For initial conditions via API, configure Copernicus Marine credentials. See `model-tools/CHANGES_COPERNICUS_API.md`.

### Memory Issues
For large domains, consider:
- Reducing resolution
- Using smaller spatial domain
- Running on high-memory node

## Model-Tools Integration

The agents use model-tools library functions:
- **download**: `Downloader` for bathymetry data
- **grid**: `grid_tools` for grid generation, metrics, smoothing
- **initialization**: `init_tools` for data loading and interpolation
- **conversions**: `convert_tools` for derived variables

## Related Files

- `model-tools/` - Core ROMS data processing library
- `model-tools/scripts/grid_generation.py` - Original grid script
- `model-tools/scripts/initialize.py` - Original initialization script
- **Bathymetry**: h (positive depths)
- **Masks**: mask_rho, mask_u, mask_v, mask_psi
- **Vertical**: s_rho, s_w (sigma coordinates)

## Architecture

```
User Prompt
    ↓
[LLM Parser] → Grid Parameters
    ↓
[Download Bathymetry] → SRTM15+ Dataset
    ↓
[Generate Grid]
    ├─ Create staggered grids
    ├─ Compute metrics
    ├─ Interpolate bathymetry
    ├─ Apply smoothing
    └─ Create masks
    ↓
ROMS Grid (roms_grid.nc)
```

## Performance

- **LLM Query**: ~1-2 seconds
- **Bathymetry Download**: ~10-60 seconds
- **Grid Generation**: ~5-30 seconds
- **Total Time**: ~1-2 minutes for standard grids

## Comparison: Old vs New

### Old Agent
- ❌ Hardcoded script modification
- ❌ Limited parsing
- ❌ Subprocess execution
- ❌ Basic regex only

### New Agent  
- ✅ Direct model-tools integration
- ✅ LLM natural language understanding
- ✅ Robust parameter extraction
- ✅ Named region recognition
- ✅ Comprehensive error handling

## Troubleshooting

### LLM Not Available
Set environment variable:
```bash
export ANTHROPIC_API_KEY="your-key"
```
Or pass to constructor:
```python
agent = ROMSGridAgent(model_tools_path="...", api_key="your-key")
```

Agent will fall back to regex parsing if no API key is provided.

### Import Errors
Verify model-tools path and dependencies:
```bash
pip install xarray numpy scipy anthropic
```

## Future Enhancements

- Multiple bathymetry sources
- Interactive parameter refinement
- Grid visualization
- Nested grid support
- Integration with initialization/boundary conditions

## License

Part of the model-tools project.

