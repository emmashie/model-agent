# Model Agent - ROMS Grid Generation with LLM

Code to setup an agentic workflow for setting up ocean models, now with LLM integration for natural language understanding.

## Overview

This repository contains intelligent agents for automating Regional Ocean Modeling System (ROMS) grid generation using Large Language Models (LLMs) and the model-tools library.

## Files

- **`llm-grid-agent.py`** - ✨ New! Full-featured agent with LLM integration
- **`simple-grid-agent.py`** - Original agent (legacy, being updated)

## Features

### 🤖 LLM-Powered Parsing
- Parse natural language requests for grid generation
- Understand common location names (e.g., "Chesapeake Bay", "Gulf of Maine")
- Extract grid parameters: resolution, depth thresholds, smoothing options
- Fallback to regex-based parsing when LLM is unavailable

### 🌊 Automated Workflow
1. **Parse Request**: Extract coordinates and parameters from natural language
2. **Download Bathymetry**: Fetch and subset SRTM15+ bathymetry data
3. **Generate Grid**: Create ROMS-compatible grid with proper staggering
4. **Apply Smoothing**: Iteratively smooth bathymetry to meet steepness criteria (rx0)
5. **Create Masks**: Generate land/sea masks for all grid types (rho, u, v, psi)

### 🔧 Model-Tools Integration
Uses the updated model-tools library functions:
- `download.Downloader` for bathymetry data
- `grid.grid_tools` for grid generation and metrics
- Proper computation of staggered grids (Arakawa C-grid)
- Grid metrics: `pm`, `pn`, `f` (Coriolis parameter)
- Bathymetry smoothing to meet rx0 steepness criteria

## Quick Start

### Installation

```bash
# Set up environment
export ANTHROPIC_API_KEY="your-api-key-here"

# Install dependencies
pip install anthropic xarray numpy scipy
```

### Basic Usage

```python
from llm_grid_agent import ROMSGridAgent

# Option 1: Specify output directory
agent = ROMSGridAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
    output_dir="/path/to/output"  # Your desired output location
)

# Option 2: Will prompt for output directory interactively
agent = ROMSGridAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
)
# The agent will ask: "Enter output directory path: "

# Natural language request
result = agent.execute_workflow(
    "Create a 1 km resolution grid for the US East Coast "
    "from latitude 35 to 42 and longitude -75 to -65"
)
```

### Output Directory Configuration

The agent requires an output directory for NetCDF files:

- **Specified at initialization**: Pass `output_dir="/path/to/output"`
- **Interactive prompt**: Omit `output_dir` and the agent will ask you
- **Directory creation**: If the path doesn't exist, the agent will create it
- **Files generated**: `topo_1min.nc`, `downloaded_bathy.nc`, `roms_grid.nc`

### Example Prompts

```python
# Explicit coordinates
"Create a 1 km resolution grid for latitude 35.0 to 42.0 and longitude -75.0 to -65.0"

# Named regions  
"I need a ROMS grid for Chesapeake Bay with 50 vertical layers"

# Mixed format
"Set up a model for the Gulf of Maine region with 2km resolution"

# Technical specification
"Generate a grid: lat 30-40, lon -80 to -70, 1km resolution, 40 layers, rx0 < 0.15"
```

## Parameters Extracted by LLM

- `lat_min`, `lat_max`: Latitude range (decimal degrees)
- `lon_min`, `lon_max`: Longitude range (decimal degrees)
- `resolution_km`: Grid resolution in kilometers (default: 1.0)
- `N_layers`: Number of vertical layers (default: 50)
- `hmin`: Minimum depth in meters (default: 5)
- `smoothing`: Apply bathymetry smoothing (default: true)
- `rx0_threshold`: Steepness parameter threshold (default: 0.2)

## Known Regions

The LLM recognizes:
- **US East Coast**: 24°N to 45°N, -81°W to -65°W
- **Gulf of Mexico**: 18°N to 31°N, -98°W to -80°W
- **California Coast**: 32°N to 42°N, -125°W to -117°W
- **Chesapeake Bay**: 36.5°N to 39.5°N, -77.5°W to -75.5°W
- **Gulf of Maine**: 41°N to 45°N, -71°W to -66°W
- **Florida Keys**: 24.5°N to 25.5°N, -82°W to -80°W

## Output Files

- `downloaded_bathy.nc`: Subsetted bathymetry data
- `roms_grid.nc`: Complete ROMS grid with all variables

### Grid File Contents
- **Coordinates**: lat/lon for rho, u, v, psi grids
- **Metrics**: pm, pn, f (Coriolis)
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

