#!/usr/bin/env python3
"""
Test script demonstrating intelligent parameter suggestion in grid agent.

This script shows how the grid agent now:
1. Extracts simulation goals from natural language
2. Suggests appropriate parameters based on goals and region
3. Offers users a choice to accept suggestions or customize
"""

from llm_grid_agent import ROMSGridAgent

def test_submesoscale():
    """Test submesoscale-resolving configuration suggestion."""
    print("\n" + "="*80)
    print("TEST 1: Submesoscale-Resolving Grid")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    # Request with simulation goals specified
    prompt = "I want to set up a submesoscale resolving grid of the Gulf Stream"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: Gulf Stream (~25-45°N, -80 to -50°W)")
    print("  2. Extract goal: 'submesoscale resolving'")
    print("  3. Suggest: ~1km resolution, 50 levels, theta_s=7, hc=-250m")
    print("  4. Ask user: Accept suggestions or customize?")
    
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    print("\n✓ Test setup complete (uncomment to execute)")


def test_coastal():
    """Test coastal configuration suggestion."""
    print("\n" + "="*80)
    print("TEST 2: Coastal Dynamics Grid")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    # Request with simulation goals
    prompt = "Create a coastal upwelling model for the California coast"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: California coast (~32-42°N, -125 to -117°W)")
    print("  2. Extract goal: 'coastal upwelling model'")
    print("  3. Suggest: ~2km resolution, 40 levels, balanced stretching")
    print("  4. Adjust bathymetry smoothing for steep shelf")
    
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    print("\n✓ Test setup complete (uncomment to execute)")


def test_location_only():
    """Test workflow when no simulation goals provided."""
    print("\n" + "="*80)
    print("TEST 3: Location Only (No Goals)")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    # Request without simulation goals
    prompt = "Create a grid for Puget Sound"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: Puget Sound (~47-49°N, -123.5 to -122°W)")
    print("  2. No goals found → Prompt user for simulation objectives")
    print("  3. User describes: e.g., 'tidal and estuarine circulation'")
    print("  4. Suggest parameters based on user's goals")
    print("  5. Ask: Accept or customize?")
    
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    print("\n✓ Test setup complete (uncomment to execute)")


def test_explicit_params():
    """Test when user specifies some parameters explicitly."""
    print("\n" + "="*80)
    print("TEST 4: Explicit Parameters Provided")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    # Request with explicit parameters
    prompt = "Create a grid for lat 35-42, lon -75 to -65 with 0.01 degree resolution and 60 vertical levels"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: 35-42°N, -75 to -65°W")
    print("  2. Extract explicit params: dx_deg=0.01, dy_deg=0.01, N_layers=60")
    print("  3. Skip suggestion (user already specified resolution/levels)")
    print("  4. Prompt only for remaining parameters (theta_s, theta_b, hc, smoothing)")
    
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    print("\n✓ Test setup complete (uncomment to execute)")


def test_suggestion_methods():
    """Test the suggestion methods directly."""
    print("\n" + "="*80)
    print("TEST 5: Direct Parameter Suggestion")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    # Test different simulation types
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


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║  ROMS Grid Agent - Intelligent Parameter Suggestion Tests                 ║
║                                                                            ║
║  These tests demonstrate the new capability to suggest appropriate        ║
║  grid parameters based on simulation goals and regional characteristics.  ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests (demonstrations only, actual execution commented out)
    test_submesoscale()
    test_coastal()
    test_location_only()
    test_explicit_params()
    
    print("\n" + "="*80)
    print("To run actual workflow tests, uncomment the result = agent.execute_workflow(prompt) lines")
    print("="*80)
    
    # This one can run without user interaction
    test_suggestion_methods()
    
    print("\n" + "="*80)
    print("All tests complete!")
    print("="*80)
