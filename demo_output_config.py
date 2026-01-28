#!/usr/bin/env python3
"""
Demo script showing the interactive output directory prompt.

This demonstrates what happens when you don't provide an output_dir parameter.
"""

import os
import sys
import importlib.util

# Load the agent module
spec = importlib.util.spec_from_file_location(
    "llm_grid_agent", 
    os.path.join(os.path.dirname(__file__), "llm_grid_agent.py")
)
llm_grid_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_grid_agent)
ROMSGridAgent = llm_grid_agent.ROMSGridAgent


def demo_with_output_dir():
    """Demo 1: Specifying output directory at initialization"""
    print("\n" + "="*70)
    print("DEMO 1: Specifying Output Directory")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    print("\n✓ Agent initialized with specified output directory")
    print(f"  Files will be saved to: {agent.output_dir}")


def demo_interactive():
    """Demo 2: Interactive prompt for output directory"""
    print("\n" + "="*70)
    print("DEMO 2: Interactive Prompt (Will Ask for Directory)")
    print("="*70)
    print("\nWhen you run without output_dir parameter:")
    print("  agent = ROMSGridAgent(model_tools_path='...')")
    print("\nThe agent will interactively prompt you:")
    print("  Enter output directory path: ")
    print("\nYou can enter:")
    print("  - Current directory: . or ./")
    print("  - Absolute path: /full/path/to/output")
    print("  - Relative path: ../output")
    print("  - Home directory: ~/my_roms_output")
    print("\nThe agent will create the directory if it doesn't exist.")


def demo_usage_patterns():
    """Demo 3: Common usage patterns"""
    print("\n" + "="*70)
    print("DEMO 3: Common Usage Patterns")
    print("="*70)
    
    print("\n# Pattern 1: Use model-tools output directory")
    print("agent = ROMSGridAgent(")
    print("    model_tools_path='/path/to/model-tools',")
    print("    output_dir='/path/to/model-tools/output'")
    print(")")
    
    print("\n# Pattern 2: Use current directory")
    print("agent = ROMSGridAgent(")
    print("    model_tools_path='/path/to/model-tools',")
    print("    output_dir='.'")
    print(")")
    
    print("\n# Pattern 3: Use custom project directory")
    print("agent = ROMSGridAgent(")
    print("    model_tools_path='/path/to/model-tools',")
    print("    output_dir='/scratch/username/roms_grids'")
    print(")")
    
    print("\n# Pattern 4: Interactive (will prompt)")
    print("agent = ROMSGridAgent(")
    print("    model_tools_path='/path/to/model-tools'")
    print(")")
    print("# Agent will ask: 'Enter output directory path: '")


def main():
    """Run all demos"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "Output Directory Configuration Demo" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    demo_with_output_dir()
    demo_interactive()
    demo_usage_patterns()
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\n✓ Output directory can be specified at initialization")
    print("✓ If not specified, agent will prompt interactively")
    print("✓ Directory is created automatically if it doesn't exist")
    print("✓ All NetCDF files are saved to the specified directory")
    print("\nFiles generated:")
    print("  - topo_1min.nc (full bathymetry dataset)")
    print("  - downloaded_bathy.nc (subsetted bathymetry)")
    print("  - roms_grid.nc (final ROMS grid file)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
