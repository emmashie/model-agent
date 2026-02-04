#!/usr/bin/env python3
"""
Unified Test Script for ROMS LLM Agents

This script consolidates all test workflows into a single, organized interface.
Tests can be run individually or as a complete suite.

Available Tests:
  1. Basic Agent Tests
     - Basic coordinate parsing (no LLM)
     - Model-tools integration
     - Regex fallback parsing
  
  2. Intelligent Grid Tests
     - Submesoscale parameter suggestions
     - Coastal configuration suggestions
     - Location-only workflow
     - Explicit parameter handling
     - Direct suggestion methods

Usage:
  python run_tests.py --test basic_parsing
  python run_tests.py --test intelligent_all
  python run_tests.py --all
  python run_tests.py --list
"""

import os
import sys
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure model-tools is in path
sys.path.insert(0, '/global/cfs/cdirs/m4304/enuss/model-tools/code')

from agents.llm_grid_agent import ROMSGridAgent


# ==============================================================================
# BASIC AGENT TESTS
# ==============================================================================

def test_basic_parsing():
    """Test basic coordinate parsing without LLM"""
    print("\n" + "="*70)
    print("TEST: Basic Parsing Mode (No LLM)")
    print("="*70)
    
    # Temporarily remove API key to test fallback
    old_key = os.environ.get('ANTHROPIC_API_KEY')
    if old_key:
        del os.environ['ANTHROPIC_API_KEY']
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/tmp/roms_test_output"
    )
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Explicit lat/lon format
    print("\nTest 1: Explicit lat/lon format")
    try:
        params = agent.parse_location_basic(
            "latitude: 35 to 42, longitude: -75 to -65"
        )
        print(f"Extracted: {json.dumps(params, indent=2)}")
        assert params['lat_min'] == 35.0
        assert params['lat_max'] == 42.0
        assert params['lon_min'] == -75.0
        assert params['lon_max'] == -65.0
        print("✓ Test 1 passed")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Test 1 failed: {e}")
    except Exception as e:
        print(f"✗ Test 1 error: {e}")
    
    # Test 2: Simple number sequence
    print("\nTest 2: Simple number sequence")
    try:
        params = agent.parse_location_basic(
            "35 42 -75 -65"
        )
        print(f"Extracted: {json.dumps(params, indent=2)}")
        assert params['lat_min'] == 35.0
        assert params['lat_max'] == 42.0
        print("✓ Test 2 passed")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Test 2 failed: {e}")
    except Exception as e:
        print(f"✗ Test 2 error: {e}")
    
    # Test 3: Default parameters
    print("\nTest 3: Default parameters")
    try:
        params = agent.parse_location_basic(
            "lat: 30-40, lon: -80 to -70"
        )
        print(f"Extracted: {json.dumps(params, indent=2)}")
        assert params.get('resolution_km') == 1.0
        assert params.get('N_layers') == 50
        assert params.get('hmin') == 5
        assert params.get('smoothing') == True
        print("✓ Test 3 passed")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ Test 3 failed: {e}")
    except Exception as e:
        print(f"✗ Test 3 error: {e}")
    
    # Restore API key
    if old_key:
        os.environ['ANTHROPIC_API_KEY'] = old_key
    
    print("\n" + "="*70)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("="*70)
    
    return tests_passed == tests_total


def test_model_tools_integration():
    """Test that model-tools components are accessible"""
    print("\n" + "="*70)
    print("TEST: Model-Tools Integration")
    print("="*70)
    
    tests_passed = 0
    tests_total = 2
    
    # Test 1: Downloader
    print("\nTest 1: Downloader initialization")
    try:
        from download import Downloader
        downloader = Downloader()
        print("✓ Downloader initialized")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Downloader failed: {e}")
    
    # Test 2: grid_tools
    print("\nTest 2: grid_tools initialization")
    try:
        from grid import grid_tools
        gt = grid_tools()
        print("✓ grid_tools initialized")
        tests_passed += 1
    except Exception as e:
        print(f"✗ grid_tools failed: {e}")
    
    print("\n" + "="*70)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("="*70)
    
    return tests_passed == tests_total


def test_regex_fallback():
    """Test regex-based fallback parsing"""
    print("\n" + "="*70)
    print("TEST: Regex Fallback Parsing")
    print("="*70)
    
    # Ensure no API key for this test
    old_key = os.environ.get('ANTHROPIC_API_KEY')
    if old_key:
        del os.environ['ANTHROPIC_API_KEY']
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/tmp/roms_test_output"
    )
    
    test_cases = [
        ("latitude: 35 to 42, longitude: -75 to -65", (35, 42, -75, -65)),
        ("lat 36.5-39.5, lon -77.5 to -75.5", (36.5, 39.5, -77.5, -75.5)),
        ("35 42 -75 -65", (35, 42, -75, -65)),
    ]
    
    tests_passed = 0
    
    for i, (prompt, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {prompt}")
        try:
            params = agent.parse_location_basic(prompt)
            lat_min, lat_max, lon_min, lon_max = expected
            assert params['lat_min'] == lat_min
            assert params['lat_max'] == lat_max
            assert params['lon_min'] == lon_min
            assert params['lon_max'] == lon_max
            print(f"✓ Test {i} passed")
            tests_passed += 1
        except AssertionError as e:
            print(f"✗ Test {i} failed: Expected {expected}, got ({params.get('lat_min')}, {params.get('lat_max')}, {params.get('lon_min')}, {params.get('lon_max')})")
        except Exception as e:
            print(f"✗ Test {i} error: {e}")
    
    # Restore API key
    if old_key:
        os.environ['ANTHROPIC_API_KEY'] = old_key
    
    print("\n" + "="*70)
    print(f"Results: {tests_passed}/{len(test_cases)} tests passed")
    print("="*70)
    
    return tests_passed == len(test_cases)


# ==============================================================================
# INTELLIGENT GRID TESTS
# ==============================================================================

def test_intelligent_submesoscale():
    """Test submesoscale-resolving configuration suggestion (demonstration)"""
    print("\n" + "="*80)
    print("TEST: Submesoscale-Resolving Grid Suggestion")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "I want to set up a submesoscale resolving grid of the Gulf Stream"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: Gulf Stream (~25-45°N, -80 to -50°W)")
    print("  2. Extract goal: 'submesoscale resolving'")
    print("  3. Suggest: ~1km resolution, 50 levels, theta_s=7, hc=-250m")
    print("  4. Ask user: Accept suggestions or customize?")
    
    print("\n✓ Test setup validated (uncomment execution line to run workflow)")
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    
    return True


def test_intelligent_coastal():
    """Test coastal configuration suggestion (demonstration)"""
    print("\n" + "="*80)
    print("TEST: Coastal Dynamics Grid Suggestion")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a coastal upwelling model for the California coast"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: California coast (~32-42°N, -125 to -117°W)")
    print("  2. Extract goal: 'coastal upwelling model'")
    print("  3. Suggest: ~2km resolution, 40 levels, balanced stretching")
    print("  4. Adjust bathymetry smoothing for steep shelf")
    
    print("\n✓ Test setup validated (uncomment execution line to run workflow)")
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    
    return True


def test_intelligent_location_only():
    """Test workflow when no simulation goals provided (demonstration)"""
    print("\n" + "="*80)
    print("TEST: Location Only (No Goals)")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a grid for Puget Sound"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: Puget Sound (~47-49°N, -123.5 to -122°W)")
    print("  2. No goals found → Prompt user for simulation objectives")
    print("  3. User describes: e.g., 'tidal and estuarine circulation'")
    print("  4. Suggest parameters based on user's goals")
    print("  5. Ask: Accept or customize?")
    
    print("\n✓ Test setup validated (uncomment execution line to run workflow)")
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    
    return True


def test_intelligent_explicit():
    """Test when user specifies some parameters explicitly (demonstration)"""
    print("\n" + "="*80)
    print("TEST: Explicit Parameters Provided")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    prompt = "Create a grid for lat 35-42, lon -75 to -65 with 0.01 degree resolution and 60 vertical levels"
    
    print(f"\nPrompt: {prompt}")
    print("\nExpected behavior:")
    print("  1. Parse location: 35-42°N, -75 to -65°W")
    print("  2. Extract explicit params: dx_deg=0.01, dy_deg=0.01, N_layers=60")
    print("  3. Skip suggestion (user already specified resolution/levels)")
    print("  4. Prompt only for remaining parameters (theta_s, theta_b, hc, smoothing)")
    
    print("\n✓ Test setup validated (uncomment execution line to run workflow)")
    # Uncomment to actually run:
    # result = agent.execute_workflow(prompt)
    
    return True


def test_intelligent_direct_suggestions():
    """Test the suggestion methods directly"""
    print("\n" + "="*80)
    print("TEST: Direct Parameter Suggestion Methods")
    print("="*80)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools"
    )
    
    test_cases = [
        {
            'name': 'Submesoscale',
            'location': {'lat_min': 35, 'lat_max': 42, 'lon_min': -75, 'lon_max': -65},
            'goals': 'submesoscale resolving simulation'
        },
        {
            'name': 'Mesoscale',
            'location': {'lat_min': 35, 'lat_max': 42, 'lon_min': -75, 'lon_max': -65},
            'goals': 'mesoscale regional ocean model'
        },
        {
            'name': 'Coastal/Estuarine',
            'location': {'lat_min': 47, 'lat_max': 49, 'lon_min': -123.5, 'lon_max': -122},
            'goals': 'coastal shelf and estuarine dynamics'
        },
    ]
    
    tests_passed = 0
    
    for case in test_cases:
        print(f"\nCase: {case['name']} - {case['goals']}")
        try:
            suggestions = agent._suggest_grid_parameters_with_llm(
                case['location'], 
                case['goals']
            )
            
            print(f"  Suggested resolution: {suggestions.get('dx_deg', 'N/A')}° x {suggestions.get('dy_deg', 'N/A')}°")
            print(f"  Vertical levels: {suggestions.get('N_layers', 'N/A')}")
            print(f"  Stretching: theta_s={suggestions.get('theta_s', 'N/A')}, theta_b={suggestions.get('theta_b', 'N/A')}")
            print(f"  Critical depth: {suggestions.get('hc', 'N/A')} m")
            
            # Validate that suggestions were returned
            assert suggestions.get('dx_deg') is not None
            assert suggestions.get('N_layers') is not None
            print("✓ Suggestions generated successfully")
            tests_passed += 1
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print("\n" + "="*80)
    print(f"Results: {tests_passed}/{len(test_cases)} tests passed")
    print("="*80)
    
    return tests_passed == len(test_cases)


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

TESTS = {
    # Basic Tests
    'basic_parsing': ('Basic coordinate parsing (no LLM)', test_basic_parsing),
    'basic_integration': ('Model-tools integration', test_model_tools_integration),
    'basic_regex': ('Regex fallback parsing', test_regex_fallback),
    
    # Intelligent Grid Tests (demonstrations)
    'intelligent_submesoscale': ('Intelligent: Submesoscale suggestion', test_intelligent_submesoscale),
    'intelligent_coastal': ('Intelligent: Coastal suggestion', test_intelligent_coastal),
    'intelligent_location': ('Intelligent: Location only', test_intelligent_location_only),
    'intelligent_explicit': ('Intelligent: Explicit params', test_intelligent_explicit),
    'intelligent_direct': ('Intelligent: Direct suggestions', test_intelligent_direct_suggestions),
}

TEST_GROUPS = {
    'basic': ['basic_parsing', 'basic_integration', 'basic_regex'],
    'intelligent': ['intelligent_submesoscale', 'intelligent_coastal', 'intelligent_location', 
                   'intelligent_explicit', 'intelligent_direct'],
    'all': list(TESTS.keys()),
}


def list_tests():
    """List all available tests"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*20 + "Available Tests" + " "*33 + "║")
    print("╚" + "="*68 + "╝\n")
    
    print("Basic Tests:")
    for key, (desc, _) in TESTS.items():
        if key.startswith('basic_'):
            print(f"  {key:30} - {desc}")
    
    print("\nIntelligent Grid Tests:")
    for key, (desc, _) in TESTS.items():
        if key.startswith('intelligent_'):
            print(f"  {key:30} - {desc}")
    
    print("\nTest Groups:")
    print("  basic                          - Run all basic tests")
    print("  intelligent                    - Run all intelligent grid tests")
    print("  all                            - Run all tests")


def run_test(test_name):
    """Run a specific test"""
    if test_name in TESTS:
        desc, func = TESTS[test_name]
        print(f"\nRunning: {desc}")
        return func()
    elif test_name in TEST_GROUPS:
        print(f"\nRunning test group: {test_name}")
        results = []
        for t in TEST_GROUPS[test_name]:
            desc, func = TESTS[t]
            print(f"\n{'='*80}")
            print(f"Running: {desc}")
            print('='*80)
            results.append(func())
        
        passed = sum(results)
        total = len(results)
        print("\n" + "="*80)
        print(f"GROUP RESULTS: {passed}/{total} tests passed")
        print("="*80)
        return passed == total
    else:
        print(f"Error: Unknown test '{test_name}'")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run ROMS LLM Agent tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --list
  python run_tests.py --test basic_parsing
  python run_tests.py --test basic
  python run_tests.py --test intelligent_direct
  python run_tests.py --all
        """
    )
    
    parser.add_argument('--test', '-t',
                       help='Run a specific test or test group')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available tests')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
    elif args.all:
        run_test('all')
    elif args.test:
        success = run_test(args.test)
        sys.exit(0 if success else 1)
    else:
        # Default: show help and list tests
        parser.print_help()
        print("\n")
        list_tests()


if __name__ == "__main__":
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "ROMS LLM Agents - Test Suite" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    main()
