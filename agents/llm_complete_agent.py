#!/usr/bin/env python3
"""
ROMS Complete Setup Agent - Full Model Setup Workflow

This agent combines all four agents to provide a complete ROMS model setup workflow:
- Grid generation
- Initial conditions
- Boundary conditions
- Surface forcing

The combined workflow:
1. Parses user's natural language request to extract region and time information
2. Generates ROMS grid with bathymetry
3. Generates initial conditions file
4. Generates boundary conditions and climatology files
5. Generates surface forcing file
6. Provides summary of all generated files

Usage:
    agent = ROMSCompleteSetupAgent(model_tools_path="/path/to/model-tools")
    result = agent.execute_workflow(
        "Create a complete ROMS setup for Chesapeake Bay for January 2024"
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
from llm_boundary_agent import ROMSBoundaryAgent
from llm_forcing_agent import ROMSSurfaceForcingAgent


class ROMSCompleteSetupAgent:
    """
    Combined ROMS Setup Agent for complete model configuration.
    
    This agent orchestrates all four components for a complete ROMS setup:
    1. Grid generation
    2. Initial conditions
    3. Boundary conditions
    4. Surface forcing
    
    The workflow can be customized to include/exclude specific components.
    
    Attributes:
        model_tools_path: Path to model-tools repository
        output_dir: Directory for output files
        grid_agent: ROMSGridAgent instance
        init_agent: ROMSInitAgent instance (created after grid)
        boundary_agent: ROMSBoundaryAgent instance (created after grid)
        forcing_agent: ROMSSurfaceForcingAgent instance (created after grid)
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
        
        print("=" * 70)
        print("ROMS Complete Setup Agent")
        print("Grid + Initial + Boundary + Forcing")
        print("=" * 70)
        
        # Initialize grid agent
        print("\n📐 Initializing Grid Generation Agent...")
        self.grid_agent = ROMSGridAgent(
            model_tools_path=model_tools_path,
            api_key=api_key,
            output_dir=output_dir,
            model=model
        )
        
        # Other agents will be created after grid is generated
        self.init_agent = None
        self.boundary_agent = None
        self.forcing_agent = None
    
    def execute_workflow(self, prompt: str, skip_init: bool = False, 
                        skip_boundary: bool = False, skip_forcing: bool = False) -> Dict:
        """
        Execute complete ROMS setup workflow.
        
        This combines all four components: grid, initial, boundary, and forcing.
        1. Parse prompt for region and time information
        2. Generate ROMS grid (with user prompts for parameters)
        3. Generate initial conditions using the created grid
        4. Generate boundary conditions
        5. Generate surface forcing
        6. Return summary of all generated files
        
        Args:
            prompt: Natural language description of desired ROMS setup.
                   Should include region and time range.
                   Examples:
                   - "Create a complete ROMS setup for Chesapeake Bay for January 2024"
                   - "Set up model for US East Coast lat 35-42, lon -75 to -65, January 1-31 2024"
                   - "Generate complete model setup for Gulf of Maine, January 2024"
            skip_init: If True, skip initial conditions generation (default: False)
            skip_boundary: If True, skip boundary conditions generation (default: False)
            skip_forcing: If True, skip surface forcing generation (default: False)
            
        Returns:
            Dictionary with workflow results including:
            - success: Boolean indicating if workflow completed
            - grid_result: Results from grid generation
            - init_result: Results from initial conditions generation
            - boundary_result: Results from boundary generation
            - forcing_result: Results from forcing generation
            - files: Dictionary of all generated file paths
            - message: Status message
        """
        print("\n" + "=" * 70)
        print("COMPLETE ROMS SETUP WORKFLOW")
        print("=" * 70)
        print(f"\n📝 User request: {prompt}\n")
        
        results = {
            "success": True,
            "files": {},
            "grid_result": None,
            "init_result": None,
            "boundary_result": None,
            "forcing_result": None
        }
        
        # ====================================================================
        # PHASE 1: GRID GENERATION
        # ====================================================================
        
        print("\n" + "=" * 70)
        print("PHASE 1: GRID GENERATION")
        print("=" * 70)
        
        grid_result = self.grid_agent.execute_workflow(prompt)
        results["grid_result"] = grid_result
        
        if not grid_result.get('success'):
            return {
                "success": False,
                "error": f"Grid generation failed: {grid_result.get('error', 'Unknown error')}",
                **results
            }
        
        grid_file = grid_result['grid_file']
        results["files"]["grid"] = grid_file
        results["files"]["bathymetry"] = grid_result.get('bathymetry_file')
        print(f"\n✅ Grid generation complete!")
        print(f"   Grid file: {grid_file}")
        
        # ====================================================================
        # PHASE 2: INITIAL CONDITIONS GENERATION
        # ====================================================================
        
        if skip_init:
            print("\n⏭️  Skipping initial conditions generation")
        else:
            print("\n" + "=" * 70)
            try:
                continue_input = input("Continue with initial conditions generation? [Y/n]: ").strip().lower()
                if continue_input in ('n', 'no'):
                    skip_init = True
                    print("⏭️  Skipping initial conditions generation")
            except KeyboardInterrupt:
                skip_init = True
                print("\n⏭️  Skipping initial conditions generation")
        
        if not skip_init:
            print("\n" + "=" * 70)
            print("PHASE 2: INITIAL CONDITIONS GENERATION")
            print("=" * 70)
            
            print(f"\n🌊 Initializing Initial Conditions Agent...")
            self.init_agent = ROMSInitAgent(
                model_tools_path=self.model_tools_path,
                grid_file=grid_file,
                api_key=self.api_key,
                output_dir=self.grid_agent.output_dir,
                model=self.model
            )
            
            init_result = self.init_agent.execute_workflow(prompt)
            results["init_result"] = init_result
            
            if not init_result.get('success'):
                print(f"\n⚠️  Initial conditions generation failed")
                results["success"] = False
            else:
                results["files"]["initial_conditions"] = init_result['init_file']
                print(f"\n✅ Initial conditions generation complete!")
                print(f"   Initial conditions file: {init_result['init_file']}")
        
        # ====================================================================
        # PHASE 3: BOUNDARY CONDITIONS GENERATION
        # ====================================================================
        
        if skip_boundary:
            print("\n⏭️  Skipping boundary conditions generation")
        else:
            print("\n" + "=" * 70)
            try:
                continue_input = input("Continue with boundary conditions generation? [Y/n]: ").strip().lower()
                if continue_input in ('n', 'no'):
                    skip_boundary = True
                    print("⏭️  Skipping boundary conditions generation")
            except KeyboardInterrupt:
                skip_boundary = True
                print("\n⏭️  Skipping boundary conditions generation")
        
        if not skip_boundary:
            print("\n" + "=" * 70)
            print("PHASE 3: BOUNDARY CONDITIONS GENERATION")
            print("=" * 70)
            
            print(f"\n🌐 Initializing Boundary Conditions Agent...")
            self.boundary_agent = ROMSBoundaryAgent(
                model_tools_path=self.model_tools_path,
                grid_file=grid_file,
                output_dir=self.grid_agent.output_dir,
                api_key=self.api_key,
                model=self.model
            )
            
            boundary_result = self.boundary_agent.execute_workflow(prompt)
            results["boundary_result"] = boundary_result
            
            if not boundary_result.get('success'):
                print(f"\n⚠️  Boundary conditions generation failed")
                results["success"] = False
            else:
                results["files"]["boundary"] = boundary_result['files'].get('boundary')
                results["files"]["climatology"] = boundary_result['files'].get('climatology')
                print(f"\n✅ Boundary conditions generation complete!")
                if boundary_result['files'].get('boundary'):
                    print(f"   Boundary forcing: {boundary_result['files']['boundary']}")
                if boundary_result['files'].get('climatology'):
                    print(f"   Climatology: {boundary_result['files']['climatology']}")
        
        # ====================================================================
        # PHASE 4: SURFACE FORCING GENERATION
        # ====================================================================
        
        if skip_forcing:
            print("\n⏭️  Skipping surface forcing generation")
        else:
            print("\n" + "=" * 70)
            try:
                continue_input = input("Continue with surface forcing generation? [Y/n]: ").strip().lower()
                if continue_input in ('n', 'no'):
                    skip_forcing = True
                    print("⏭️  Skipping surface forcing generation")
            except KeyboardInterrupt:
                skip_forcing = True
                print("\n⏭️  Skipping surface forcing generation")
        
        if not skip_forcing:
            print("\n" + "=" * 70)
            print("PHASE 4: SURFACE FORCING GENERATION")
            print("=" * 70)
            
            print(f"\n☀️ Initializing Surface Forcing Agent...")
            self.forcing_agent = ROMSSurfaceForcingAgent(
                model_tools_path=self.model_tools_path,
                grid_file=grid_file,
                output_dir=self.grid_agent.output_dir,
                api_key=self.api_key,
                model=self.model
            )
            
            forcing_result = self.forcing_agent.execute_workflow(prompt)
            results["forcing_result"] = forcing_result
            
            if not forcing_result.get('success'):
                print(f"\n⚠️  Surface forcing generation failed")
                results["success"] = False
            else:
                results["files"]["forcing"] = forcing_result['file']
                print(f"\n✅ Surface forcing generation complete!")
                print(f"   Surface forcing: {forcing_result['file']}")
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        
        print("\n" + "=" * 70)
        if results["success"]:
            print("✅ COMPLETE ROMS SETUP FINISHED!")
        else:
            print("⚠️  ROMS SETUP COMPLETED WITH SOME ERRORS")
        print("=" * 70)
        print("\nGenerated Files:")
        file_num = 1
        for key, path in results["files"].items():
            if path:
                print(f"  {file_num}. {key.replace('_', ' ').title()}: {path}")
                file_num += 1
        
        if results["success"]:
            print("\nYour ROMS model is ready to run!")
        print("=" * 70)
        
        results["message"] = "Complete ROMS setup successful" if results["success"] else "Setup completed with some errors"
        return results
    
    def execute_grid_only(self, prompt: str) -> Dict:
        """
        Execute only grid generation workflow.
        
        Args:
            prompt: Natural language description of desired grid
            
        Returns:
            Dictionary with grid generation results
        """
        return self.execute_workflow(prompt, skip_init=True, skip_boundary=True, skip_forcing=True)


def main():
    """
    Main function for testing the combined agent.
    
    Demonstrates complete workflow with all four components:
    1. Grid generation
    2. Initial conditions
    3. Boundary conditions
    4. Surface forcing
    
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
        "Create a complete ROMS setup for Chesapeake Bay for January 2024",
        "Set up model for US East Coast lat 35-42, lon -75 to -65, January 1-31 2024",
        "Generate complete model setup for Gulf of Maine, January 2024",
    ]
    
    print("\n" + "=" * 70)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Parse your request for region and time range")
    print("  2. Generate ROMS grid with bathymetry")
    print("  3. Generate initial conditions file")
    print("  4. Generate boundary conditions and climatology")
    print("  5. Generate surface forcing file")
    print("\n" + "=" * 70)
    
    # Execute complete workflow
    result = agent.execute_workflow(complete_prompts[0])
    
    print("\n" + "=" * 70)
    print("FINAL WORKFLOW RESULT:")
    print("=" * 70)
    
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
    
    if result.get("boundary_result"):
        result_json["boundary_parameters"] = result["boundary_result"].get("parameters")
    
    if result.get("forcing_result"):
        result_json["forcing_parameters"] = result["forcing_result"].get("parameters")
    
    print(json.dumps(result_json, indent=2))


if __name__ == "__main__":
    main()
