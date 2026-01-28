# Output Directory Configuration - Quick Reference

## Summary of Changes

The ROMS Grid Agent now requires an output directory to be specified, either:
1. **At initialization** - Pass `output_dir` parameter
2. **Interactively** - Omit the parameter and the agent will prompt you

## Where Output Files Go

### Before (Old Behavior)
```python
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
# Files saved to: /path/to/model-agent/ (agent script directory)
```

### After (New Behavior)
```python
# Option 1: Specify directory
agent = ROMSGridAgent(
    model_tools_path="/path/to/model-tools",
    output_dir="/path/to/desired/output"
)
# Files saved to: /path/to/desired/output/

# Option 2: Interactive prompt
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
# Prompts: "Enter output directory path: "
# You enter: /path/to/output
# Files saved to: /path/to/output/
```

## Files Generated

All files are saved to `output_dir`:

1. **`topo_1min.nc`** - Full SRTM15+ bathymetry dataset (cached, reused)
2. **`downloaded_bathy.nc`** - Subsetted bathymetry for your region
3. **`roms_grid.nc`** - Final ROMS grid file

## Code Locations

### Initialization
File: `llm-grid-agent.py`, lines ~40-55
```python
def __init__(self, model_tools_path: str, api_key: Optional[str] = None, 
             output_dir: Optional[str] = None):
    # If output_dir is None, prompt user
    if output_dir is None:
        output_dir = self._prompt_for_output_dir()
    
    # Validate and create directory
    self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir, exist_ok=True)
```

### Interactive Prompt
File: `llm-grid-agent.py`, lines ~58-103
```python
def _prompt_for_output_dir(self) -> str:
    """Prompt user for output directory path."""
    print("\nWhere should the output NetCDF files be saved?")
    output_dir = input("\nEnter output directory path: ").strip()
    # Validates and returns path
```

### File Saving
**Bathymetry**: `llm-grid-agent.py`, line ~200
```python
output_path = os.path.join(self.output_dir, output_file)
```

**Grid**: `llm-grid-agent.py`, line ~370
```python
output_path = os.path.join(self.output_dir, output_file)
```

## Common Usage Patterns

### Pattern 1: Model-Tools Output Directory (Recommended)
```python
agent = ROMSGridAgent(
    model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
    output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
)
```

### Pattern 2: Current Directory
```python
agent = ROMSGridAgent(
    model_tools_path="/path/to/model-tools",
    output_dir="."  # or "./" or os.getcwd()
)
```

### Pattern 3: Custom Project Directory
```python
agent = ROMSGridAgent(
    model_tools_path="/path/to/model-tools",
    output_dir="/scratch/username/my_project/grids"
)
```

### Pattern 4: Home Directory
```python
agent = ROMSGridAgent(
    model_tools_path="/path/to/model-tools",
    output_dir="~/roms_output"  # Expands to /home/username/roms_output
)
```

### Pattern 5: Interactive (Will Prompt)
```python
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
# Displays prompt:
# ============================================================
# Output Directory Configuration
# ============================================================
# 
# Where should the output NetCDF files be saved?
# Examples:
#   - Current directory: . or ./
#   - Model-tools output: /global/cfs/cdirs/m4304/enuss/model-tools/output
#   - Custom path: /path/to/your/output
# 
# Enter output directory path: _
```

## Features

✅ **Path expansion**: Supports `~` for home directory  
✅ **Relative paths**: Converts to absolute paths automatically  
✅ **Directory creation**: Creates directory if it doesn't exist  
✅ **Validation**: Checks if path is valid and writable  
✅ **Confirmation**: Asks before creating new directories  
✅ **Error handling**: Graceful handling of invalid paths  
✅ **Keyboard interrupt**: Ctrl+C defaults to current directory  

## Testing

Run the demo to see how it works:
```bash
python demo_output_config.py
```

Run tests with explicit output directory:
```bash
python test_agent.py
```

## Migration Guide

If you're updating existing code:

**Old code:**
```python
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
# Files saved to model-agent directory
```

**New code (specify directory):**
```python
agent = ROMSGridAgent(
    model_tools_path="/path/to/model-tools",
    output_dir="/path/to/model-tools/output"
)
# Files saved to specified directory
```

**New code (interactive):**
```python
agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
# Will prompt for directory
```

## Environment Variables

You can also set a default output directory via environment variable:

```bash
export ROMS_OUTPUT_DIR="/path/to/default/output"
```

Then modify the code to check for this:
```python
output_dir = output_dir or os.getenv('ROMS_OUTPUT_DIR')
```

## Best Practices

1. **Use model-tools/output** for consistency with other tools
2. **Use absolute paths** for clarity and reliability
3. **Check disk space** before generating large grids
4. **Keep bathymetry cached** - `topo_1min.nc` is reused across runs
5. **Organize by project** - Create subdirectories for different regions

## Examples

See `examples.py` for complete usage examples showing different output directory configurations.
