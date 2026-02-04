# Intelligent Grid Parameter Suggestion - Enhancement Summary

## Overview

Enhanced the ROMS grid agent with intelligent parameter suggestion capabilities that leverage domain knowledge about ocean modeling, ROMS S-coordinate stretching, and regional characteristics.

## Key Enhancements

### 1. Simulation Goal Extraction
**File**: `llm_grid_agent.py` - `parse_location_with_llm()`

- Now extracts `simulation_goals` in addition to lat/lon bounds
- Identifies keywords like "submesoscale", "coastal", "mesoscale", "upwelling"
- Updated system prompt with examples of different simulation types

### 2. Interactive Goal Prompting  
**File**: `llm_grid_agent.py` - `_prompt_for_simulation_goals()`

New method that asks users about their simulation objectives if not specified:
- What phenomena to resolve (eddies, upwelling, tides, etc.)
- Type of simulation (process study, operational, climate)
- Scale requirements

Provides helpful examples to guide users.

### 3. LLM-Based Parameter Suggestion
**File**: `llm_grid_agent.py` - `_suggest_grid_parameters_with_llm()`

Uses LLM with comprehensive domain knowledge:

**Knowledge Base Includes**:
- **ROMS S-Coordinate Stretching**: 
  - theta_s (0-10): Surface stretching control
  - theta_b (0-4): Bottom stretching control  
  - hc: Critical depth transition
  - References ROMS wiki: https://www.myroms.org/wiki/Vertical_S-coordinate

- **Resolution Guidelines**:
  - Submesoscale: O(1 km) ≈ 0.01°
  - Mesoscale: O(5-10 km) ≈ 0.05-0.1°
  - Regional/shelf: O(2-5 km) ≈ 0.02-0.05°
  - Basin-scale: O(10-25 km) ≈ 0.1-0.25°

- **Vertical Level Recommendations**:
  - Fine surface processes: 50-100 levels
  - Standard ocean: 30-50 levels
  - Coarse/efficient: 20-30 levels

- **Regional Characteristics**:
  - Shallow shelves: More levels, shallower hc, higher theta_b
  - Deep ocean: Fewer levels, deeper hc
  - Coastal upwelling: Balanced stretching
  - Frontal/mesoscale: Higher resolution, surface-focused

- **Bathymetry Smoothing**:
  - Steep bathymetry: Lower rx0_thresh, more iterations
  - Gentle bathymetry: Higher rx0_thresh, fewer iterations

Returns suggestions with scientific reasoning.

### 4. Fallback Suggestion Method
**File**: `llm_grid_agent.py` - `_suggest_grid_parameters_basic()`

Keyword-based heuristics when LLM unavailable:
- Submesoscale: 1km, 50 levels, theta_s=7, hc=-250m
- Coastal: 2km, 40 levels, theta_s=6, theta_b=2, hc=-100m
- Mesoscale: 5km, 40 levels, theta_s=5, theta_b=0.5, hc=-300m
- General: 5km, 40 levels, standard stretching

### 5. Parameter Offering Interface
**File**: `llm_grid_agent.py` - `_offer_suggested_parameters()`

Displays suggested configuration with:
- Scientific reasoning/justification
- All parameter values with units
- Horizontal resolution converted to km
- Clear descriptions of each parameter type

Prompts user: "Use these suggested parameters? [Y/n]"

### 6. Enhanced Workflow Integration
**File**: `llm_grid_agent.py` - `execute_workflow()`

New workflow logic:
1. Parse prompt → extract location + simulation goals
2. If no goals → prompt user interactively
3. If no explicit parameters → suggest intelligent defaults
4. Offer suggestions → user accepts or customizes
5. If user specified some params → only prompt for missing ones
6. Proceed with bathymetry download and grid generation

## Usage Examples

### Example 1: Goals Specified
```python
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")

# User provides simulation type
result = agent.execute_workflow(
    "I want to set up a submesoscale resolving grid of the Gulf Stream"
)

# Agent flow:
# ✓ Parse: Gulf Stream location (25-45°N, -80 to -50°W)
# ✓ Extract: "submesoscale resolving" goal
# 🤖 Suggest: 1km resolution, 50 levels, theta_s=7, hc=-250m
# ❓ Offer: Accept suggestions or customize?
```

### Example 2: Location Only
```python
# User provides only location
result = agent.execute_workflow("Create a grid for Puget Sound")

# Agent flow:
# ✓ Parse: Puget Sound location (47-49°N, -123.5 to -122°W)
# ✗ No goals found
# ❓ Prompt: "What phenomena do you want to resolve?"
# ✓ User: "tidal and estuarine circulation"
# 🤖 Suggest: Appropriate coastal configuration
# ❓ Offer: Accept or customize?
```

### Example 3: Explicit Parameters
```python
# User specifies some parameters
result = agent.execute_workflow(
    "Create a grid for lat 35-42, lon -75 to -65 with 0.01 degree resolution"
)

# Agent flow:
# ✓ Parse: Location + dx_deg=0.01, dy_deg=0.01
# ✓ Skip suggestions (explicit params found)
# ❓ Prompt: Only for missing params (N_layers, theta_s, etc.)
```

## Technical Implementation

### Key Files Modified
1. **llm_grid_agent.py** (~1332 lines, +200 lines added)
   - Enhanced `parse_location_with_llm()` system prompt
   - Added `_prompt_for_simulation_goals()`
   - Added `_suggest_grid_parameters_with_llm()`
   - Added `_suggest_grid_parameters_basic()`
   - Added `_offer_suggested_parameters()`
   - Modified `execute_workflow()` with new logic

2. **README.md**
   - Added "Intelligent Parameter Suggestion" feature section
   - Updated example requests with goal-oriented prompts
   - Added "Intelligent Parameter Suggestion Workflow" section
   - Documented the suggestion process and knowledge base

3. **test_intelligent_grid.py** (new file)
   - Comprehensive test suite demonstrating new features
   - Test cases for different simulation types
   - Tests for location-only prompts
   - Tests for explicit parameter handling
   - Direct testing of suggestion methods

## Benefits

1. **User-Friendly**: Non-expert users get scientifically appropriate defaults
2. **Educational**: Reasoning helps users understand parameter choices
3. **Flexible**: Users can accept suggestions or customize
4. **Context-Aware**: Suggestions adapt to region and simulation type
5. **Robust**: Fallback methods ensure functionality without LLM

## Backward Compatibility

✅ Fully backward compatible:
- Old prompts still work (will ask for goals if needed)
- All existing parameter prompting preserved
- Can still specify explicit parameters in prompt
- Fallback to basic parsing if LLM unavailable

## Future Enhancements

Potential additions:
- Learn from user customizations to improve suggestions
- Add more regional templates (Arctic, tropical, etc.)
- Include computational cost estimates with suggestions
- Suggest different configurations for sensitivity studies
- Integration with initialization agent for end-to-end setup
