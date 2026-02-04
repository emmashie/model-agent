# Quick Reference Guide

Quick reference for the reorganized model-agent repository.

## Directory Structure

```
agents/     - Core agent implementations
examples/   - Example scripts with CLI
tests/      - Test scripts with CLI  
docs/       - Documentation files
```

## Core Agents

- **llm_grid_agent.py** - Grid generation with intelligent parameter suggestions
- **llm_init_agent.py** - Initial conditions generation  
- **llm_boundary_agent.py** - Boundary conditions with automatic ocean boundary detection
- **llm_forcing_agent.py** - Surface forcing from ERA5 data
- **llm_complete_agent.py** - Combined workflow

## Common Commands

### Examples

```bash
# List all examples
python examples/run_examples.py --list

# Run a specific example
python examples/run_examples.py --example grid_explicit

# Interactive mode
python examples/run_examples.py --interactive

# Get help
python examples/run_examples.py --help
```

### Tests

```bash
# List all tests
python tests/run_tests.py --list

# Run a specific test
python tests/run_tests.py --test basic_parsing

# Run a test group
python tests/run_tests.py --test basic

# Run all tests
python tests/run_tests.py --all

# Get help
python tests/run_tests.py --help
```

### Agents

```bash
# Run agents directly
python agents/llm_grid_agent.py
python agents/llm_init_agent.py
python agents/llm_boundary_agent.py \
    --model-tools-path /path/to/model-tools \
    --grid-file /path/to/roms_grid.nc
python agents/llm_forcing_agent.py \
    --model-tools-path /path/to/model-tools \
    --grid-file /path/to/roms_grid.nc
python agents/llm_complete_agent.py
```

## Import Statements

```python
# Import agents
from agents.llm_grid_agent import ROMSGridAgent
from agents.llm_init_agent import ROMSInitAgent
from agents.llm_boundary_agent import ROMSBoundaryAgent
from agents.llm_forcing_agent import ROMSSurfaceForcingAgent
from agents.llm_complete_agent import ROMSCompleteSetupAgent

# Use agents
grid_agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
init_agent = ROMSInitAgent(model_tools_path="/path/to/model-tools", 
                           grid_file="/path/to/grid.nc")
boundary_agent = ROMSBoundaryAgent(model_tools_path="/path/to/model-tools",
                                   grid_file="/path/to/grid.nc")
forcing_agent = ROMSSurfaceForcingAgent(model_tools_path="/path/to/model-tools",
                                        grid_file="/path/to/grid.nc")
complete_agent = ROMSCompleteSetupAgent(model_tools_path="/path/to/model-tools")
```

## Available Examples

**Grid Examples** (5):
- `grid_explicit` - Explicit coordinates
- `grid_named` - Named region
- `grid_custom` - Custom parameters
- `grid_compact` - Compact format
- `grid_no_llm` - Without LLM

**Intelligent Examples** (5):
- `intelligent_submesoscale` - Submesoscale-resolving
- `intelligent_coastal` - Coastal dynamics
- `intelligent_location` - Location only
- `intelligent_explicit` - Explicit parameters
- `intelligent_direct` - Direct suggestions

**Complete Workflow** (2):
- `complete_full` - Full setup
- `complete_grid` - Grid only

**Output Demos** (3):
- `demo_output_specified` - Specified output dir
- `demo_output_interactive` - Interactive prompt
- `demo_output_patterns` - Usage patterns

## Available Tests

**Basic Tests** (3):
- `basic_parsing` - Coordinate parsing
- `basic_integration` - Model-tools integration
- `basic_regex` - Regex fallback

**Intelligent Tests** (5):
- `intelligent_submesoscale` - Submesoscale suggestion
- `intelligent_coastal` - Coastal suggestion
- `intelligent_location` - Location only
- `intelligent_explicit` - Explicit params
- `intelligent_direct` - Direct suggestions

**Test Groups**:
- `basic` - All basic tests
- `intelligent` - All intelligent tests
- `all` - All tests

## Documentation Files

```
docs/CHANGES.md                               - Changelog
docs/INITIALIZATION_FIXES.md                  - Init fixes
docs/INTELLIGENT_SUGGESTION_ENHANCEMENT.md    - Parameter suggestions
docs/OUTPUT_CONFIG.md                         - Output configuration
docs/BEFORE_AFTER.md                          - Visual comparison
```

## Quick Start

```python
# 1. Import an agent
from agents.llm_complete_agent import ROMSCompleteSetupAgent

# 2. Initialize
agent = ROMSCompleteSetupAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
)

# 3. Run workflow
result = agent.execute_workflow(
    "Create a ROMS setup for Chesapeake Bay, initialized for Jan 1, 2024"
)

# 4. Access results
print(f"Grid: {result['files']['grid']}")
print(f"Initial conditions: {result['files']['initial_conditions']}")
```

## File Locations

| Old Location | New Location |
|--------------|--------------|
| `llm_grid_agent.py` | `agents/llm_grid_agent.py` |
| `llm_init_agent.py` | `agents/llm_init_agent.py` |
| `llm_complete_agent.py` | `agents/llm_complete_agent.py` |
| `examples.py` | `examples/run_examples.py` (consolidated) |
| `test_agent.py` | `tests/run_tests.py` (consolidated) |
| `CHANGES.md` | `docs/CHANGES.md` |

## Getting Help

- **Examples**: `python examples/run_examples.py --help`
- **Tests**: `python tests/run_tests.py --help`
- **README**: [README.md](../README.md)
- **Migration**: [REORGANIZATION_GUIDE.md](../REORGANIZATION_GUIDE.md)
- **Summary**: [REORGANIZATION_SUMMARY.md](../REORGANIZATION_SUMMARY.md)
