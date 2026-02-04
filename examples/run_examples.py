#!/usr/bin/env python3
"""
Unified Examples Script for ROMS LLM Agents

This script consolidates all example workflows into a single, organized interface.
Users can run specific examples via command-line arguments or interactively.

Available Examples:
  1. Grid Agent Examples
     - Explicit coordinates
     - Named region
     - Custom parameters
     - Compact format
     - Without LLM (fallback parsing)
  
  2. Intelligent Grid Examples
     - Submesoscale-resolving grid
     - Coastal dynamics grid
     - Location-only (prompts for goals)
     - Explicit parameters
     - Direct parameter suggestions
  
  3. Complete Workflow Examples
     - Full setup from natural language
     - Grid only workflow
     - Using existing grid
  
  4. Output Configuration Demo
     - Specified output directory
     - Interactive prompt
     - Common usage patterns

Usage:
  python run_examples.py --example grid_explicit
  python run_examples.py --example intelligent_submesoscale
  python run_examples.py --list
  python run_examples.py --interactive
"""

import os
import sys
import argparse

# Add parent directory to path
try:
    # When running as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # When running in interactive environment (Jupyter, IPython, etc.)
    sys.path.insert(0, os.path.dirname(os.getcwd()))

from agents.llm_grid_agent import ROMSGridAgent
from agents.llm_complete_agent import ROMSCompleteSetupAgent


# ==============================================================================
# GRID AGENT EXAMPLES
# ==============================================================================

def example_grid_explicit():
    """Grid Example 1: Explicit coordinate specification"""
    print("\n" + "="*70)
    print("EXAMPLE: Explicit Coordinates")
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


def example_grid_named_region():
    """Grid Example 2: Named region with default parameters"""
    print("\n" + "="*70)
    print("EXAMPLE: Named Region")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output/agent_test"
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


def example_grid_custom():
    """Grid Example 3: Custom resolution and smoothing parameters"""
    print("\n" + "="*70)
    print("EXAMPLE: Custom Parameters")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output/agent_test"
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


def example_grid_compact():
    """Grid Example 4: Compact technical specification"""
    print("\n" + "="*70)
    print("EXAMPLE: Compact Format")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="."
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


def example_grid_without_llm():
    """Grid Example 5: Using fallback parsing (no LLM)"""
    print("\n" + "="*70)
    print("EXAMPLE: Without LLM (Fallback Parsing)")
    print("="*70)
    
    # Temporarily remove API key to demonstrate fallback
    old_key = os.environ.get('ANTHROPIC_API_KEY')
    if old_key:
        del os.environ['ANTHROPIC_API_KEY']
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output/agent_test"
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


# ==============================================================================
# INTELLIGENT GRID EXAMPLES
# ==============================================================================

def example_intelligent_submesoscale():
    """Intelligent Example 1: Submesoscale-resolving configuration"""
    print("\n" + "="*80)
    print("EXAMPLE: Submesoscale-Resolving Grid")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "I want to set up a submesoscale resolving grid of the Gulf Stream"
    
    print(f"\nPrompt: {prompt}")
    print("\nThis will:")
    print("  1. Parse location: Gulf Stream (~25-45°N, -80 to -50°W)")
    print("  2. Extract goal: 'submesoscale resolving'")
    print("  3. Suggest: ~1km resolution, 50 levels, theta_s=7, hc=-250m")
    print("  4. Ask user: Accept suggestions or customize?")
    
    result = agent.execute_workflow(prompt)
    return result


def example_intelligent_coastal():
    """Intelligent Example 2: Coastal configuration"""
    print("\n" + "="*80)
    print("EXAMPLE: Coastal Dynamics Grid")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a coastal upwelling model for the California coast"
    
    print(f"\nPrompt: {prompt}")
    print("\nThis will:")
    print("  1. Parse location: California coast (~32-42°N, -125 to -117°W)")
    print("  2. Extract goal: 'coastal upwelling model'")
    print("  3. Suggest: ~2km resolution, 40 levels, balanced stretching")
    print("  4. Adjust bathymetry smoothing for steep shelf")
    
    result = agent.execute_workflow(prompt)
    return result


def example_intelligent_location_only():
    """Intelligent Example 3: Location only (prompts for goals)"""
    print("\n" + "="*80)
    print("EXAMPLE: Location Only (No Goals)")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a grid for Puget Sound"
    
    print(f"\nPrompt: {prompt}")
    print("\nThis will:")
    print("  1. Parse location: Puget Sound (~47-49°N, -123.5 to -122°W)")
    print("  2. No goals found → Prompt user for simulation objectives")
    print("  3. User describes: e.g., 'tidal and estuarine circulation'")
    print("  4. Suggest parameters based on user's goals")
    print("  5. Ask: Accept or customize?")
    
    result = agent.execute_workflow(prompt)
    return result


def example_intelligent_explicit_params():
    """Intelligent Example 4: Explicit parameters provided"""
    print("\n" + "="*80)
    print("EXAMPLE: Explicit Parameters Provided")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a grid for lat 35-42, lon -75 to -65 with 0.01 degree resolution and 60 vertical levels"
    
    print(f"\nPrompt: {prompt}")
    print("\nThis will:")
    print("  1. Parse location: 35-42°N, -75 to -65°W")
    print("  2. Extract explicit params: dx_deg=0.01, dy_deg=0.01, N_layers=60")
    print("  3. Skip suggestion (user already specified resolution/levels)")
    print("  4. Prompt only for remaining parameters (theta_s, theta_b, hc, smoothing)")
    
    result = agent.execute_workflow(prompt)
    return result


def example_intelligent_direct_suggestions():
    """Intelligent Example 5: Direct parameter suggestion tests"""
    print("\n" + "="*80)
    print("EXAMPLE: Direct Parameter Suggestion")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    test_cases = [
        {
            'location': {'lat_min': 35, 'lat_max': 42, 'lon_min': -75, 'lon_max': -65},
            'goals': 'submesoscale resolving simulation'
        },
        {
            'location': {'lat_min': 35, 'lat_max': 42, 'lon_min': -75, 'lon_max': -65},
            'goals': 'mesoscale regional ocean model'
        },
        {
            'location': {'lat_min': 47, 'lat_max': 49, 'lon_min': -123.5, 'lon_max': -122},
            'goals': 'coastal shelf and estuarine dynamics'
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case['goals']}")
        suggestions = agent._suggest_grid_parameters_with_llm(
            case['location'], 
            case['goals']
        )
        
        print(f"  Suggested resolution: {suggestions.get('dx_deg', 'N/A')}° x {suggestions.get('dy_deg', 'N/A')}°")
        print(f"  Vertical levels: {suggestions.get('N_layers', 'N/A')}")
        print(f"  Stretching: theta_s={suggestions.get('theta_s', 'N/A')}, theta_b={suggestions.get('theta_b', 'N/A')}")
        print(f"  Critical depth: {suggestions.get('hc', 'N/A')} m")
        if 'reasoning' in suggestions:
            print(f"  Reasoning: {suggestions['reasoning']}")
    
    print("\n✓ Direct suggestion tests complete")


# ==============================================================================
# COMPLETE WORKFLOW EXAMPLES
# ==============================================================================

def example_complete_workflow():
    """Complete Example 1: Full setup from natural language"""
    print("\n" + "="*80)
    print("EXAMPLE: Complete Workflow (Grid + Initial Conditions)")
    print("="*80)
    
    agent = ROMSCompleteSetupAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    result = agent.execute_workflow(
        "Create a ROMS setup for Chesapeake Bay from lat 36.5-39.5, lon -77.5 to -75.5, "
        "initialized for January 1, 2024"
    )
    
    if result.get('success'):
        print("\n✅ Complete setup generated successfully!")
        print(f"   Grid: {result['files']['grid']}")
        print(f"   Initial conditions: {result['files']['initial_conditions']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


def example_complete_grid_only():
    """Complete Example 2: Grid only workflow"""
    print("\n" + "="*80)
    print("EXAMPLE: Grid Only Workflow")
    print("="*80)
    
    agent = ROMSCompleteSetupAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    result = agent.execute_workflow(
        "Create a ROMS grid for the Gulf of Maine",
        skip_initial_conditions=True
    )
    
    if result.get('success'):
        print("\n✅ Grid generated successfully!")
        print(f"   Grid: {result['files']['grid']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    return result


# ==============================================================================
# OUTPUT CONFIGURATION DEMO
# ==============================================================================

def demo_output_specified():
    """Demo 1: Specifying output directory at initialization"""
    print("\n" + "="*70)
    print("DEMO: Specifying Output Directory")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    print("\n✓ Agent initialized with specified output directory")
    print(f"  Files will be saved to: {agent.output_dir}")


def demo_output_interactive():
    """Demo 2: Interactive prompt for output directory"""
    print("\n" + "="*70)
    print("DEMO: Interactive Prompt (Will Ask for Directory)")
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


def demo_output_patterns():
    """Demo 3: Common usage patterns"""
    print("\n" + "="*70)
    print("DEMO: Common Usage Patterns")
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


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

EXAMPLES = {
    # Grid Agent Examples
    'grid_explicit': ('Grid: Explicit coordinates', example_grid_explicit),
    'grid_named': ('Grid: Named region', example_grid_named_region),
    'grid_custom': ('Grid: Custom parameters', example_grid_custom),
    'grid_compact': ('Grid: Compact format', example_grid_compact),
    'grid_no_llm': ('Grid: Without LLM', example_grid_without_llm),
    
    # Intelligent Grid Examples
    'intelligent_submesoscale': ('Intelligent: Submesoscale-resolving', example_intelligent_submesoscale),
    'intelligent_coastal': ('Intelligent: Coastal dynamics', example_intelligent_coastal),
    'intelligent_location': ('Intelligent: Location only', example_intelligent_location_only),
    'intelligent_explicit': ('Intelligent: Explicit parameters', example_intelligent_explicit_params),
    'intelligent_direct': ('Intelligent: Direct suggestions', example_intelligent_direct_suggestions),
    
    # Complete Workflow Examples
    'complete_full': ('Complete: Full setup', example_complete_workflow),
    'complete_grid': ('Complete: Grid only', example_complete_grid_only),
    
    # Output Configuration Demos
    'demo_output_specified': ('Demo: Specified output dir', demo_output_specified),
    'demo_output_interactive': ('Demo: Interactive output prompt', demo_output_interactive),
    'demo_output_patterns': ('Demo: Output usage patterns', demo_output_patterns),
}


def list_examples():
    """List all available examples"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "Available Examples" + " "*34 + "║")
    print("╚" + "="*68 + "╝\n")
    
    print("Grid Agent Examples:")
    for key, (desc, _) in EXAMPLES.items():
        if key.startswith('grid_'):
            print(f"  {key:30} - {desc}")
    
    print("\nIntelligent Grid Examples:")
    for key, (desc, _) in EXAMPLES.items():
        if key.startswith('intelligent_'):
            print(f"  {key:30} - {desc}")
    
    print("\nComplete Workflow Examples:")
    for key, (desc, _) in EXAMPLES.items():
        if key.startswith('complete_'):
            print(f"  {key:30} - {desc}")
    
    print("\nOutput Configuration Demos:")
    for key, (desc, _) in EXAMPLES.items():
        if key.startswith('demo_'):
            print(f"  {key:30} - {desc}")


def run_interactive():
    """Run examples interactively"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "ROMS LLM Agents - Interactive Examples" + " "*18 + "║")
    print("╚" + "="*68 + "╝\n")
    
    categories = {
        '1': ('Grid Agent Examples', [k for k in EXAMPLES if k.startswith('grid_')]),
        '2': ('Intelligent Grid Examples', [k for k in EXAMPLES if k.startswith('intelligent_')]),
        '3': ('Complete Workflow Examples', [k for k in EXAMPLES if k.startswith('complete_')]),
        '4': ('Output Configuration Demos', [k for k in EXAMPLES if k.startswith('demo_')]),
    }
    
    print("Select a category:")
    for key, (name, _) in categories.items():
        print(f"  {key}. {name}")
    print("  q. Quit")
    
    choice = input("\nCategory: ").strip()
    
    if choice == 'q':
        return
    
    if choice not in categories:
        print("Invalid choice")
        return
    
    cat_name, examples = categories[choice]
    print(f"\n{cat_name}:")
    
    for i, key in enumerate(examples, 1):
        desc, _ = EXAMPLES[key]
        print(f"  {i}. {desc}")
    print("  b. Back")
    
    example_choice = input("\nExample: ").strip()
    
    if example_choice == 'b':
        return run_interactive()
    
    try:
        idx = int(example_choice) - 1
        if 0 <= idx < len(examples):
            key = examples[idx]
            _, func = EXAMPLES[key]
            func()
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(
        description='Run ROMS LLM Agent examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py --list
  python run_examples.py --example grid_explicit
  python run_examples.py --example intelligent_submesoscale
  python run_examples.py --interactive
        """
    )
    
    parser.add_argument('--example', '-e', 
                       help='Run a specific example (use --list to see options)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available examples')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.list:
        list_examples()
    elif args.interactive:
        run_interactive()
    elif args.example:
        if args.example in EXAMPLES:
            desc, func = EXAMPLES[args.example]
            print(f"\nRunning: {desc}")
            func()
        else:
            print(f"Error: Unknown example '{args.example}'")
            print("Use --list to see available examples")
            sys.exit(1)
    else:
        # Default: show help and list examples
        parser.print_help()
        print("\n")
        list_examples()


if __name__ == "__main__":
    main()
