#!/usr/bin/env python3
"""
Example usage of the ROMS Grid Agent with LLM integration.

This script demonstrates various ways to use the agent to generate
ROMS grids from natural language prompts.
"""

import os
import sys

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

from llm_grid_agent import ROMSGridAgent


def example_1_explicit_coordinates():
    """Example 1: Explicit coordinate specification"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Explicit Coordinates")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output/agent_test"
    )
    
    result = agent.execute_workflow(
        "Create a 1 km resolution ROMS grid from latitude 45.0 to 50.0 "
        "and longitude -127.0 to -122.0 with 50 vertical layers"
    )
    
    if result.get('success'):
        print("\n✅ Grid generated successfully!")
        print(f"   Output: {result['grid_file']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_2_named_region():
    """Example 2: Named region with default parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Named Region")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    result = agent.execute_workflow(
        "I need a ROMS grid for Chesapeake Bay"
    )
    
    if result.get('success'):
        print("\n✅ Grid generated successfully!")
        print(f"   Bounds: {result['parameters']['lat_min']:.1f}°N to "
              f"{result['parameters']['lat_max']:.1f}°N, "
              f"{result['parameters']['lon_min']:.1f}°W to "
              f"{result['parameters']['lon_max']:.1f}°W")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_3_custom_parameters():
    """Example 3: Custom resolution and smoothing parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Parameters")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    result = agent.execute_workflow(
        "Generate a ROMS grid for the Gulf of Maine with 2 km resolution, "
        "60 vertical layers, and bathymetry smoothing to rx0 < 0.15"
    )
    
    if result.get('success'):
        print("\n✅ Grid generated successfully!")
        print(f"   Resolution: {result['parameters']['resolution_km']} km")
        print(f"   Vertical layers: {result['parameters']['N_layers']}")
        print(f"   rx0 threshold: {result['parameters']['rx0_threshold']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_4_compact_format():
    """Example 4: Compact technical specification"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Compact Format")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="."  # Current directory
    )
    
    result = agent.execute_workflow(
        "lat: 30-35, lon: -80 to -75, 1km, 40 layers"
    )
    
    if result.get('success'):
        print("\n✅ Grid generated successfully!")
        print(f"   Grid file: {result['grid_file']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_5_without_llm():
    """Example 5: Using fallback parsing (no LLM)"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Without LLM (Fallback Parsing)")
    print("="*70)
    
    # Temporarily remove API key to demonstrate fallback
    old_key = os.environ.get('ANTHROPIC_API_KEY')
    if old_key:
        del os.environ['ANTHROPIC_API_KEY']
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    result = agent.execute_workflow(
        "latitude: 36 to 39, longitude: -77 to -75"
    )
    
    # Restore API key
    if old_key:
        os.environ['ANTHROPIC_API_KEY'] = old_key
    
    if result.get('success'):
        print("\n✅ Grid generated successfully (using regex parsing)!")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "ROMS Grid Agent - Example Usage" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    examples = [
        ("Explicit Coordinates", example_1_explicit_coordinates),
        ("Named Region", example_2_named_region),
        ("Custom Parameters", example_3_custom_parameters),
        ("Compact Format", example_4_compact_format),
        ("Without LLM", example_5_without_llm),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nTo run a specific example, uncomment it in the code.")
    print("For now, running Example 1...\n")
    
    # Run one example (change this to run different examples)
    # Note: Will prompt for output directory if not provided
    example_1_explicit_coordinates()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - downloaded_bathy.nc (bathymetry data)")
    print("  - roms_grid.nc (ROMS grid file)")
    print("\nYou can inspect these files with:")
    print("  ncdump -h roms_grid.nc")
    print("  or use xarray in Python:")
    print("  import xarray as xr")
    print("  ds = xr.open_dataset('roms_grid.nc')")
    print("  print(ds)")


if __name__ == "__main__":
    main()
