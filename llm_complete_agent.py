#!/usr/bin/env python3
"""
ROMS Complete Setup Agent - Combined Grid and Initial Conditions Workflow

This agent combines the grid generation and initial conditions agents to provide
a complete ROMS model setup workflow.

The combined workflow:
1. Parses user's natural language request to extract region and initialization time
2. Prompts user for grid parameters (resolution, vertical levels, smoothing, etc.)
3. Generates ROMS grid with bathymetry
4. Prompts user for initialization parameters (data source, fill values, etc.)
5. Generates initial conditions file using the created grid
6. Provides summary of all generated files

Usage:
    agent = ROMSCompleteSetupAgent(model_tools_path="/path/to/model-tools")
    result = agent.execute_workflow(
        "Create a ROMS setup for Chesapeake Bay initialized for January 1, 2024"
    )
"""

import os
import sys
import json
from typing import Dict, Optional

# Import the individual agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_grid_agent import ROMSGridAgent
from llm_init_agent import ROMSInitAgent


class ROMSCompleteSetupAgent:
    """
    Combined ROMS Setup Agent for complete model configuration.
    
    This agent orchestrates both grid generation and initial conditions creation,
    providing a seamless workflow from a single natural language request.
    
    The workflow:
    1. Uses LLM to parse request for region bounds and initialization time
    2. Grid Generation Phase:
       - Prompts for grid parameters
       - Downloads bathymetry
       - Generates ROMS grid file
    3. Initial Conditions Phase:
       - Prompts for initialization parameters
       - Loads ocean data (GLORYS)
       - Generates initial conditions file
    4. Returns paths to all generated files
    
    Attributes:
        model_tools_path: Path to model-tools repository
        output_dir: Directory for output files
        grid_agent: ROMSGridAgent instance
        init_agent: ROMSInitAgent instance (created after grid)
    """
    
    def __init__(self, model_tools_path: str, api_key: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 model: str = "claude-haiku-4-5-20251001-v1-birthright"):
        """
        Initialize combined agent.
        
        Args:
            model_tools_path: Path to model-tools directory
            api_key: LLM API key (if not provided, will try to read from environment)
            output_dir: Directory for output files (if not provided, will prompt user)
            model: Model name to use (default: claude-haiku-4-5-20251001-v1-birthright)
        """
        self.model_tools_path = model_tools_path
        self.api_key = api_key
        self.output_dir = output_dir
        self.model = model
        
        print("=" * 60)
        print("ROMS Complete Setup Agent")
        print("Combined Grid Generation + Initial Conditions")
        print("=" * 60)
        
        # Initialize grid agent
        print("\n📐 Initializing Grid Generation Agent...")
        self.grid_agent = ROMSGridAgent(
            model_tools_path=model_tools_path,
            api_key=api_key,
            output_dir=output_dir,
            model=model
        )
        
        # init_agent will be created after grid is generated
        self.init_agent = None
    
    def execute_workflow(self, prompt: str, skip_init: bool = False) -> Dict:
        """
        Execute complete ROMS setup workflow.
        
        This combines both grid generation and initial conditions creation:
        1. Parse prompt for region and initialization time
        2. Generate ROMS grid (with user prompts for parameters)
        3. Generate initial conditions using the created grid
        4. Return summary of all generated files
        
        Args:
            prompt: Natural language description of desired ROMS setup.
                   Should include both region and initialization time.
                   Examples:
                   - "Create a ROMS setup for Chesapeake Bay initialized for January 1, 2024"
                   - "Set up model for US East Coast lat 35-42, lon -75 to -65, start date 2024-01-01"
                   - "Generate grid and initial conditions for Gulf of Maine, initialize mid-2023"
            skip_init: If True, only generate grid and skip initial conditions (default: False)
            
        Returns:
            Dictionary with workflow results including:
            - success: Boolean indicating if workflow completed
            - grid_result: Results from grid generation
            - init_result: Results from initial conditions generation (if not skipped)
            - files: List of all generated file paths
            - message: Status message
        """
        print("\n" + "=" * 60)
        print("COMPLETE ROMS SETUP WORKFLOW")
        print("=" * 60)
        print(f"\n📝 User request: {prompt}\n")
        
        # ====================================================================
        # PHASE 1: GRID GENERATION
        # ====================================================================
        
        print("\n" + "=" * 60)
        print("PHASE 1: GRID GENERATION")
        print("=" * 60)
        
        grid_result = self.grid_agent.execute_workflow(prompt)
        
        if not grid_result.get('success'):
            return {
                "success": False,
                "error": f"Grid generation failed: {grid_result.get('error', 'Unknown error')}",
                "grid_result": grid_result
            }
        
        grid_file = grid_result['grid_file']
        print(f"\n✅ Grid generation complete!")
        print(f"   Grid file: {grid_file}")
        
        # Check if user wants to skip initial conditions
        if skip_init:
            print("\n⏭️  Skipping initial conditions generation (skip_init=True)")
            return {
                "success": True,
                "grid_result": grid_result,
                "init_result": None,
                "files": {
                    "grid": grid_file,
                    "bathymetry": grid_result.get('bathymetry_file')
                },
                "message": "Grid generation complete (initial conditions skipped)"
            }
        
        # Ask user if they want to continue with initial conditions
        print("\n" + "=" * 60)
        try:
            continue_input = input("Continue with initial conditions generation? [Y/n]: ").strip().lower()
            if continue_input in ('n', 'no'):
                print("⏭️  Skipping initial conditions generation")
                return {
                    "success": True,
                    "grid_result": grid_result,
                    "init_result": None,
                    "files": {
                        "grid": grid_file,
                        "bathymetry": grid_result.get('bathymetry_file')
                    },
                    "message": "Grid generation complete (initial conditions skipped by user)"
                }
        except KeyboardInterrupt:
            print("\n⏭️  Skipping initial conditions generation")
            return {
                "success": True,
                "grid_result": grid_result,
                "init_result": None,
                "files": {
                    "grid": grid_file,
                    "bathymetry": grid_result.get('bathymetry_file')
                },
                "message": "Grid generation complete (initial conditions skipped)"
            }
        
        # ====================================================================
        # PHASE 2: INITIAL CONDITIONS GENERATION
        # ====================================================================
        
        print("\n" + "=" * 60)
        print("PHASE 2: INITIAL CONDITIONS GENERATION")
        print("=" * 60)
        
        # Initialize the init agent with the generated grid file
        print(f"\n🌊 Initializing Initial Conditions Agent...")
        self.init_agent = ROMSInitAgent(
            model_tools_path=self.model_tools_path,
            grid_file=grid_file,
            api_key=self.api_key,
            output_dir=self.grid_agent.output_dir,
            model=self.model
        )
        
        init_result = self.init_agent.execute_workflow(prompt)
        
        if not init_result.get('success'):
            print(f"\n⚠️  Initial conditions generation failed, but grid was created successfully")
            return {
                "success": False,
                "error": f"Initial conditions failed: {init_result.get('error', 'Unknown error')}",
                "grid_result": grid_result,
                "init_result": init_result,
                "files": {
                    "grid": grid_file,
                    "bathymetry": grid_result.get('bathymetry_file')
                }
            }
        
        init_file = init_result['init_file']
        print(f"\n✅ Initial conditions generation complete!")
        print(f"   Initial conditions file: {init_file}")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE ROMS SETUP FINISHED!")
        print("=" * 60)
        print("\nGenerated Files:")
        print(f"  1. Grid:              {grid_file}")
        print(f"  2. Bathymetry:        {grid_result.get('bathymetry_file')}")
        print(f"  3. Initial Conditions: {init_file}")
        print("\nYour ROMS model is ready to run!")
        print("=" * 60)
        
        return {
            "success": True,
            "grid_result": grid_result,
            "init_result": init_result,
            "files": {
                "grid": grid_file,
                "bathymetry": grid_result.get('bathymetry_file'),
                "initial_conditions": init_file
            },
            "message": "Complete ROMS setup successful"
        }
    
    def execute_grid_only(self, prompt: str) -> Dict:
        """
        Execute only grid generation workflow.
        
        Args:
            prompt: Natural language description of desired grid
            
        Returns:
            Dictionary with grid generation results
        """
        return self.execute_workflow(prompt, skip_init=True)
    
    def execute_init_with_existing_grid(self, prompt: str, grid_file: str) -> Dict:
        """
        Execute only initial conditions workflow with an existing grid file.
        
        Args:
            prompt: Natural language description of initialization request
            grid_file: Path to existing ROMS grid NetCDF file
            
        Returns:
            Dictionary with initial conditions results
        """
        print("\n" + "=" * 60)
        print("INITIAL CONDITIONS WITH EXISTING GRID")
        print("=" * 60)
        print(f"\n📝 User request: {prompt}")
        print(f"📐 Using existing grid: {grid_file}\n")
        
        # Initialize the init agent with existing grid
        self.init_agent = ROMSInitAgent(
            model_tools_path=self.model_tools_path,
            grid_file=grid_file,
            api_key=self.api_key,
            output_dir=self.grid_agent.output_dir,
            model=self.model
        )
        
        init_result = self.init_agent.execute_workflow(prompt)
        
        if init_result.get('success'):
            print("\n" + "=" * 60)
            print("✅ INITIAL CONDITIONS GENERATION COMPLETE!")
            print("=" * 60)
            print(f"\nGenerated Files:")
            print(f"  Grid (existing):       {grid_file}")
            print(f"  Initial Conditions:    {init_result['init_file']}")
            print("=" * 60)
        
        return {
            "success": init_result.get('success', False),
            "init_result": init_result,
            "files": {
                "grid": grid_file,
                "initial_conditions": init_result.get('init_file')
            },
            "message": "Initial conditions created with existing grid"
        }


def main():
    """
    Main function for testing the combined agent.
    
    Demonstrates three usage modes:
    1. Complete workflow (grid + initial conditions)
    2. Grid only
    3. Initial conditions with existing grid
    
    Example usage:
        # Set API key in environment
        export LLM_API_KEY=<your-key>
        
        # Run the complete workflow
        python llm_complete_agent.py
    """
    
    # Initialize combined agent
    agent = ROMSCompleteSetupAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"
    )
    
    # Test prompts
    complete_prompts = [
        "Create a ROMS setup for the US East Coast from latitude 35 to 42 and longitude -75 to -65, initialized for January 1, 2024",
        "Set up model for Chesapeake Bay, initialize for start of 2024",
        "Generate grid and initial conditions for Gulf of Maine, initialize mid-2023",
    ]
    
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Parse your request for region and initialization time")
    print("  2. Prompt for grid parameters (resolution, vertical levels, etc.)")
    print("  3. Generate ROMS grid with bathymetry")
    print("  4. Prompt for initialization parameters (data source, etc.)")
    print("  5. Generate initial conditions file")
    print("\n" + "=" * 60)
    
    # Execute complete workflow
    result = agent.execute_workflow(complete_prompts[0])
    
    print("\n" + "=" * 60)
    print("FINAL WORKFLOW RESULT:")
    print("=" * 60)
    
    # Create serializable result for JSON output
    result_json = {
        "success": result["success"],
        "message": result["message"],
        "files": result.get("files", {})
    }
    
    if result.get("grid_result"):
        result_json["grid_parameters"] = result["grid_result"].get("parameters")
    
    if result.get("init_result"):
        result_json["init_parameters"] = result["init_result"].get("parameters")
    
    print(json.dumps(result_json, indent=2))


if __name__ == "__main__":
    main()
