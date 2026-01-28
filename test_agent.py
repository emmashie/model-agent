#!/usr/bin/env python3
"""
Test script for the ROMS Grid Agent (without requiring LLM API key).

This demonstrates the agent's fallback parsing capabilities.
"""

import os
import sys
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Ensure model-tools is in path
sys.path.insert(0, '/global/cfs/cdirs/m4304/enuss/model-tools/code')

# Import without anthropic (will use fallback parsing)
os.environ.pop('ANTHROPIC_API_KEY', None)

# Import the agent - need to use importlib for hyphenated filename
import importlib.util
spec = importlib.util.spec_from_file_location(
    "llm_grid_agent", 
    os.path.join(os.path.dirname(__file__), "llm_grid_agent.py")
)
llm_grid_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_grid_agent)
ROMSGridAgent = llm_grid_agent.ROMSGridAgent


def test_basic_parsing():
    """Test basic coordinate parsing without LLM"""
    print("\n" + "="*70)
    print("Testing Grid Agent - Basic Parsing Mode (No LLM)")
    print("="*70)
    
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/tmp/roms_test_output"  # Use temp directory for tests
    )
    
    print("\nTest 1: Explicit lat/lon format")
    params = agent.parse_location_basic(
        "latitude: 35 to 42, longitude: -75 to -65"
    )
    print(f"Extracted: {json.dumps(params, indent=2)}")
    assert params['lat_min'] == 35.0
    assert params['lat_max'] == 42.0
    assert params['lon_min'] == -75.0
    assert params['lon_max'] == -65.0
    print("✓ Test 1 passed")
    
    print("\nTest 2: Simple number sequence")
    params = agent.parse_location_basic(
        "35 42 -75 -65"
    )
    print(f"Extracted: {json.dumps(params, indent=2)}")
    assert params['lat_min'] == 35.0
    assert params['lat_max'] == 42.0
    print("✓ Test 2 passed")
    
    print("\nTest 3: Default parameters")
    params = agent.parse_location_basic(
        "lat: 30-40, lon: -80 to -70"
    )
    print(f"Extracted: {json.dumps(params, indent=2)}")
    assert params.get('resolution_km') == 1.0
    assert params.get('N_layers') == 50
    assert params.get('hmin') == 5
    assert params.get('smoothing') == True
    print("✓ Test 3 passed")
    
    print("\n" + "="*70)
    print("All tests passed! ✅")
    print("="*70)
    print("\nThe agent is ready to use.")
    print("\nNote: For full LLM features, set ANTHROPIC_API_KEY environment variable.")
    print("Without LLM, the agent uses regex-based parsing (more limited).")


def test_model_tools_imports():
    """Test that model-tools components are accessible"""
    print("\n" + "="*70)
    print("Testing Model-Tools Integration")
    print("="*70)
    
    try:
        from download import Downloader
        downloader = Downloader()
        print("✓ Downloader initialized")
    except Exception as e:
        print(f"✗ Downloader failed: {e}")
        return False
    
    try:
        from grid import grid_tools
        gt = grid_tools()
        print("✓ grid_tools initialized")
    except Exception as e:
        print(f"✗ grid_tools failed: {e}")
        return False
    
    print("\n✅ Model-tools integration successful")
    return True


def main():
    """Run all tests"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "ROMS Grid Agent - Test Suite" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    # Test model-tools imports
    if not test_model_tools_imports():
        print("\n❌ Model-tools integration failed!")
        print("Ensure model-tools is installed and in the correct path.")
        return 1
    
    # Test basic parsing
    try:
        test_basic_parsing()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("🎉 All tests completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Set ANTHROPIC_API_KEY for LLM features")
    print("2. Run examples.py to see usage examples")
    print("3. Use llm-grid-agent.py for grid generation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
