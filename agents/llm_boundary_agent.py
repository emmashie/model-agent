#!/usr/bin/env python3
"""
ROMS Boundary Conditions Agent with LLM Integration

This agent automates the generation of boundary condition files for ROMS,
including:
- Automatic detection of ocean boundaries from grid
- GLORYS data download
- Interpolation to grid boundaries
- Creation of boundary forcing files
- Optional climatology file generation

Usage:
    from agents.llm_boundary_agent import ROMSBoundaryAgent
    
    agent = ROMSBoundaryAgent(
        model_tools_path="/path/to/model-tools",
        grid_file="/path/to/roms_grid.nc"
    )
    
    result = agent.execute_workflow(
        "Create boundary conditions from January 1-31, 2024"
    )
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import xarray as xr

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ OpenAI library not available. Will use basic parsing only.")


class ROMSBoundaryAgent:
    """
    Agent for generating ROMS boundary condition files using LLM-assisted parsing.
    """
    
    def __init__(
        self,
        model_tools_path: str,
        grid_file: str,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001-v1-birthright"
    ):
        """
        Initialize the ROMS Boundary Conditions Agent.
        
        Args:
            model_tools_path: Path to model-tools directory
            grid_file: Path to ROMS grid file
            output_dir: Directory for output files (optional, will prompt if not provided)
            api_key: LLM API key (optional, reads from LLM_API_KEY env var)
            model: LLM model to use
        """
        self.model_tools_path = Path(model_tools_path)
        self.grid_file = grid_file
        self.model = model
        
        # Add model-tools code directory to path
        code_dir = self.model_tools_path / 'code'
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        
        # Import model-tools modules
        try:
            from boundary import boundary_tools
            from initialization import init_tools
            from conversions import convert_tools
            from grid import grid_tools
            
            self.boundary_tools = boundary_tools()
            self.init_tools = init_tools()
            self.convert_tools = convert_tools()
            self.grid_tools = grid_tools()
            
            print("✓ Model-tools modules loaded successfully")
        except ImportError as e:
            raise ImportError(f"Failed to import model-tools modules: {e}")
        
        # Load grid file
        try:
            self.grid = xr.open_dataset(grid_file)
            print(f"✓ Grid loaded: {grid_file}")
        except Exception as e:
            raise ValueError(f"Failed to load grid file: {e}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input("Enter output directory path: ").strip()
            if not output_dir:
                output_dir = "."
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")
        
        # Setup LLM client
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = api_key or os.environ.get('LLM_API_KEY')
            if api_key:
                try:
                    self.client = OpenAI(
                        base_url="https://ai-incubator-api.pnnl.gov",
                        api_key=api_key
                    )
                    print("✓ LLM client initialized")
                except Exception as e:
                    print(f"⚠ LLM initialization failed: {e}. Using basic parsing.")
    
    def detect_ocean_boundaries(self) -> Dict[str, bool]:
        """
        Automatically detect which boundaries are ocean boundaries based on the grid mask.
        
        Returns:
            Dictionary with boundary names and whether they need data
        """
        print("\n" + "="*70)
        print("Detecting Ocean Boundaries")
        print("="*70)
        
        mask_rho = self.grid.mask_rho.values
        eta_rho, xi_rho = mask_rho.shape
        
        boundaries = {}
        
        # Check west boundary (left edge)
        west_ocean = np.any(mask_rho[:, 0] == 1)
        boundaries['west'] = west_ocean
        print(f"  West boundary:  {'OCEAN' if west_ocean else 'LAND'}")
        
        # Check east boundary (right edge)
        east_ocean = np.any(mask_rho[:, -1] == 1)
        boundaries['east'] = east_ocean
        print(f"  East boundary:  {'OCEAN' if east_ocean else 'LAND'}")
        
        # Check south boundary (bottom edge)
        south_ocean = np.any(mask_rho[0, :] == 1)
        boundaries['south'] = south_ocean
        print(f"  South boundary: {'OCEAN' if south_ocean else 'LAND'}")
        
        # Check north boundary (top edge)
        north_ocean = np.any(mask_rho[-1, :] == 1)
        boundaries['north'] = north_ocean
        print(f"  North boundary: {'OCEAN' if north_ocean else 'LAND'}")
        
        ocean_count = sum(boundaries.values())
        print(f"\n✓ Detected {ocean_count} ocean boundaries")
        
        return boundaries
    
    def parse_time_range_with_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Parse time range from natural language using LLM.
        
        Args:
            prompt: Natural language request
            
        Returns:
            Dictionary with parsed parameters
        """
        system_prompt = """You are a helpful assistant that extracts time range parameters from natural language requests for ocean model boundary conditions.

Extract the following information:
1. start_time: Start date/time in YYYY-MM-DD format
2. end_time: End date/time in YYYY-MM-DD format
3. save_climatology: Whether to create climatology file (true/false, default: true)
4. use_api: Whether to use API for data download (true/false, default: null - will prompt)
5. glorys_data_path: Path to existing GLORYS data files (null if using API)

Return ONLY a JSON object with these keys. Set to null if not mentioned.

Examples:
- "Create boundary conditions for January 2024" → {"start_time": "2024-01-01", "end_time": "2024-01-31", "save_climatology": null, "use_api": null, "glorys_data_path": null}
- "Generate boundaries from 2024-01-01 to 2024-01-31 with climatology" → {"start_time": "2024-01-01", "end_time": "2024-01-31", "save_climatology": true, "use_api": null, "glorys_data_path": null}
- "Make boundary conditions for summer 2024, skip climatology" → {"start_time": "2024-06-01", "end_time": "2024-08-31", "save_climatology": false, "use_api": null, "glorys_data_path": null}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            print("✓ LLM parsed request successfully")
            return result
            
        except Exception as e:
            print(f"⚠ LLM parsing failed: {e}")
            return {}
    
    def parse_time_range_basic(self, prompt: str) -> Dict[str, Any]:
        """
        Basic regex-based parsing for time range (fallback).
        
        Args:
            prompt: Natural language request
            
        Returns:
            Dictionary with parsed parameters
        """
        params = {
            'start_time': None,
            'end_time': None,
            'save_climatology': None,
            'use_api': None,
            'glorys_data_path': None
        }
        
        # Try to extract dates in YYYY-MM-DD format
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        dates = re.findall(date_pattern, prompt)
        
        if len(dates) >= 2:
            params['start_time'] = dates[0].replace('/', '-')
            params['end_time'] = dates[1].replace('/', '-')
        elif len(dates) == 1:
            params['start_time'] = dates[0].replace('/', '-')
        
        # Check for climatology keywords
        if 'no climatology' in prompt.lower() or 'skip climatology' in prompt.lower():
            params['save_climatology'] = False
        elif 'climatology' in prompt.lower():
            params['save_climatology'] = True
        
        # Check for API keywords
        if 'use api' in prompt.lower() or 'download' in prompt.lower():
            params['use_api'] = True
        elif 'local' in prompt.lower() or 'file' in prompt.lower():
            params['use_api'] = False
        
        return params
    
    def prompt_for_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactively prompt user for missing parameters.
        
        Args:
            params: Partially filled parameters dictionary
            
        Returns:
            Complete parameters dictionary
        """
        print("\n" + "="*70)
        print("Boundary Conditions Configuration")
        print("="*70)
        
        # Prompt for start time
        if params.get('start_time') is None:
            while True:
                start_input = input("\nStart date (YYYY-MM-DD): ").strip()
                try:
                    datetime.strptime(start_input, '%Y-%m-%d')
                    params['start_time'] = start_input
                    print(f"  ✓ Start time: {start_input}")
                    break
                except ValueError:
                    print("  ⚠ Invalid date format. Please use YYYY-MM-DD")
        else:
            print(f"\n  ✓ Start time: {params['start_time']} (from prompt)")
        
        # Prompt for end time
        if params.get('end_time') is None:
            while True:
                end_input = input("End date (YYYY-MM-DD): ").strip()
                try:
                    datetime.strptime(end_input, '%Y-%m-%d')
                    params['end_time'] = end_input
                    print(f"  ✓ End time: {end_input}")
                    break
                except ValueError:
                    print("  ⚠ Invalid date format. Please use YYYY-MM-DD")
        else:
            print(f"  ✓ End time: {params['end_time']} (from prompt)")
        
        # Prompt for data source
        if params.get('use_api') is None:
            while True:
                api_input = input("\nUse Copernicus Marine API for data download? (y/n, default: y): ").strip().lower()
                if not api_input or api_input == 'y':
                    params['use_api'] = True
                    params['glorys_data_path'] = None
                    print("  ✓ Will use Copernicus Marine API")
                    break
                elif api_input == 'n':
                    params['use_api'] = False
                    glorys_path = input("  Path to GLORYS data files (pattern with *): ").strip()
                    params['glorys_data_path'] = glorys_path
                    print(f"  ✓ Using local data: {glorys_path}")
                    break
                else:
                    print("  ⚠ Please enter 'y' or 'n'")
        else:
            if params['use_api']:
                print("  ✓ Using Copernicus Marine API (from prompt)")
            else:
                print(f"  ✓ Using local data (from prompt)")
        
        # If using API, prompt for time buffer
        if params['use_api']:
            while True:
                try:
                    buffer_input = input("Time buffer for API download in hours (default: 24): ").strip()
                    if not buffer_input:
                        params['time_buffer_hours'] = 24
                        print("  ✓ Using default: 24 hours")
                        break
                    buffer = int(buffer_input)
                    if buffer < 0:
                        print("  ⚠ Buffer must be non-negative. Please try again.")
                        continue
                    params['time_buffer_hours'] = buffer
                    print(f"  ✓ Using time buffer: {buffer} hours")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid integer.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 24 hours")
                    params['time_buffer_hours'] = 24
                    break
        
        # Prompt for climatology
        if params.get('save_climatology') is None:
            clim_input = input("\nCreate climatology file? (y/n, default: y): ").strip().lower()
            if not clim_input or clim_input == 'y':
                params['save_climatology'] = True
                print("  ✓ Will create climatology file")
            else:
                params['save_climatology'] = False
                print("  ✓ Will skip climatology file")
        else:
            status = "enabled" if params['save_climatology'] else "disabled"
            print(f"  ✓ Climatology: {status} (from prompt)")
        
        # Set source variable names (GLORYS defaults)
        params.setdefault('source_vars', {
            'salt': 'so',
            'temp': 'thetao',
            'u': 'uo',
            'v': 'vo',
            'zeta': 'zos',
            'depth': 'depth',
            'lat': 'latitude',
            'lon': 'longitude'
        })
        
        # Set reference time for ROMS
        params.setdefault('ref_time', '2000-01-01 00:00:00')
        
        # Set source name
        params.setdefault('source_name', 'GLORYS12v1')
        
        return params
    
    def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the complete boundary conditions generation workflow.
        
        Args:
            prompt: Natural language description of boundary conditions request
            
        Returns:
            Dictionary with results and file paths
        """
        print("\n" + "="*70)
        print("ROMS Boundary Conditions Agent")
        print("="*70)
        print(f"\nRequest: {prompt}")
        
        # Step 1: Parse request
        print("\n" + "="*70)
        print("Step 1: Parsing Request")
        print("="*70)
        
        if self.client:
            print("🤖 Querying LLM to parse request...")
            params = self.parse_time_range_with_llm(prompt)
        else:
            print("📝 Using basic regex parsing...")
            params = self.parse_time_range_basic(prompt)
        
        # Step 2: Detect ocean boundaries
        boundaries = self.detect_ocean_boundaries()
        params['boundaries'] = boundaries
        
        # Step 3: Prompt for missing parameters
        params = self.prompt_for_parameters(params)
        
        # Step 4: Load or download GLORYS data
        print("\n" + "="*70)
        print("Step 2: Loading GLORYS Data")
        print("="*70)
        
        try:
            if params['use_api']:
                print("📥 Downloading data from Copernicus Marine API...")
                bc_data = self.download_glorys_data(params)
            else:
                print(f"📂 Loading data from: {params['glorys_data_path']}")
                bc_data = xr.open_mfdataset(params['glorys_data_path'])
                bc_data = bc_data.sel(time=slice(params['start_time'], params['end_time']))
            
            print(f"✓ Loaded {len(bc_data.time)} time steps")
            
            # Debug: Print data structure
            print(f"   Data variables: {list(bc_data.data_vars)}")
            print(f"   Dimensions: {dict(bc_data.dims)}")
            if 'uo' in bc_data:
                print(f"   'uo' shape: {bc_data['uo'].shape}")
            if 'thetao' in bc_data:
                print(f"   'thetao' shape: {bc_data['thetao'].shape}")
        except Exception as e:
            return {'success': False, 'error': f"Failed to load GLORYS data: {e}"}
        
        # Step 5: Interpolate to ROMS grid
        print("\n" + "="*70)
        print("Step 3: Interpolating to ROMS Grid")
        print("="*70)
        
        try:
            interpolated_data = self.interpolate_to_grid(bc_data, params)
        except Exception as e:
            return {'success': False, 'error': f"Failed to interpolate data: {e}"}
        
        # Step 6: Compute derived variables
        print("\n" + "="*70)
        print("Step 4: Computing Derived Variables")
        print("="*70)
        
        try:
            derived_data = self.compute_derived_variables(interpolated_data, params)
        except Exception as e:
            return {'success': False, 'error': f"Failed to compute derived variables: {e}"}
        
        # Step 7: Save output files
        print("\n" + "="*70)
        print("Step 5: Saving Output Files")
        print("="*70)
        
        output_files = {}
        
        try:
            # Save climatology file if requested
            if params['save_climatology']:
                clim_file = self.save_climatology(derived_data, params)
                output_files['climatology'] = clim_file
                print(f"✓ Climatology saved: {clim_file}")
            
            # Save boundary forcing file
            bry_file = self.save_boundary(derived_data, params)
            output_files['boundary'] = bry_file
            print(f"✓ Boundary forcing saved: {bry_file}")
            
        except Exception as e:
            return {'success': False, 'error': f"Failed to save output files: {e}"}
        
        # Success!
        print("\n" + "="*70)
        print("✅ Boundary Conditions Generated Successfully!")
        print("="*70)
        
        return {
            'success': True,
            'files': output_files,
            'parameters': params,
            'grid_file': self.grid_file
        }
    
    def download_glorys_data(self, params: Dict[str, Any]) -> xr.Dataset:
        """
        Download GLORYS data using Copernicus Marine API.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            xarray Dataset with GLORYS data
        """
        # Import the downloader
        sys.path.insert(0, str(self.model_tools_path / 'code'))
        from download import CopernicusMarineDownloader
        
        # Get grid bounds
        lat_min = float(self.grid.lat_rho.min())
        lat_max = float(self.grid.lat_rho.max())
        lon_min = float(self.grid.lon_rho.min())
        lon_max = float(self.grid.lon_rho.max())
        
        # Add buffer to ensure we have enough data
        buffer = 1.0  # degrees
        lat_range = (lat_min - buffer, lat_max + buffer)
        lon_range = (lon_min - buffer, lon_max + buffer)
        
        print(f"   Downloading region: lat [{lat_range[0]:.2f}, {lat_range[1]:.2f}], "
              f"lon [{lon_range[0]:.2f}, {lon_range[1]:.2f}]")
        print(f"   Time range: {params['start_time']} to {params['end_time']}")
        
        # Download data using Copernicus Marine API directly
        downloader = CopernicusMarineDownloader()
        bc_data = downloader.get_glorys_dataset(
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=(params['start_time'], params['end_time']),
            variables=['thetao', 'so', 'uo', 'vo', 'zos'],
            use_daily=True
        )
        
        print(f"   ✓ Downloaded {len(bc_data.time)} time steps")
        
        return bc_data
    
    def interpolate_to_grid(self, bc_data: xr.Dataset, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Interpolate GLORYS data to ROMS grid.
        
        Args:
            bc_data: GLORYS dataset
            params: Parameters dictionary
            
        Returns:
            Dictionary with interpolated arrays
        """
        # Prepare source coordinates
        print("   Preparing source coordinates...")
        print(f"   Looking for depth='{params['source_vars']['depth']}', lat='{params['source_vars']['lat']}', lon='{params['source_vars']['lon']}'")
        print(f"   Available coords: {list(bc_data.coords)}")
        print(f"   Available dims: {list(bc_data.dims)}")
        
        source_coords = self.init_tools.prepare_source_coords(
            bc_data,
            params['source_vars']['depth'],
            params['source_vars']['lat'],
            params['source_vars']['lon']
        )
        
        # Compute ROMS vertical coordinates
        print("   Computing ROMS vertical coordinates...")
        eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = self.grid_tools.get_grid_dims(self.grid)
        print(f"   Grid dims: eta_rho={eta_rho}, xi_rho={xi_rho}, s_rho={s_rho}")
        
        print("   Computing z_rho...")
        # Extract scalar value from hc if it's an xarray DataArray
        hc_value = float(self.grid.hc) if hasattr(self.grid.hc, 'values') else self.grid.hc
        
        z_rho = self.grid_tools.compute_z(
            self.grid.sigma_r.values,
            hc_value,
            self.grid.Cs_r.values,
            self.grid.h.values,
            np.zeros((eta_rho, xi_rho, 1))
        )
        print(f"   z_rho shape after compute_z: {z_rho.shape}")
        z_rho = np.squeeze(z_rho)
        print(f"   z_rho shape after squeeze: {z_rho.shape}")
        z_rho = np.transpose(z_rho, (1, 2, 0))  # Shape: (eta_rho, xi_rho, s_rho)
        print(f"   z_rho shape after transpose: {z_rho.shape}")
        
        # Convert to positive depths
        print("   Extracting grid coordinates...")
        roms_depth_3d = np.abs(z_rho)
        roms_lat_2d = self.grid.lat_rho.values
        roms_lon_2d = self.grid.lon_rho.values
        roms_latu_2d = self.grid.lat_u.values
        roms_lonu_2d = self.grid.lon_u.values
        roms_latv_2d = self.grid.lat_v.values
        roms_lonv_2d = self.grid.lon_v.values
        
        # Get valid time indices
        print("   Finding valid time indices...")
        u_var = params['source_vars']['u']
        u_data_values = bc_data[u_var].values
        valid_time_indices = np.where(np.isfinite(u_data_values[:, 0, -1, -1]))[0]
        nt = len(valid_time_indices)
        print(f"   Found {nt} valid time indices")
        
        print(f"   Interpolating {nt} time steps...")
        
        # Initialize arrays
        temp_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
        salt_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
        u_interp = np.empty((nt, s_rho, eta_rho, xi_u))
        v_interp = np.empty((nt, s_rho, eta_v, xi_rho))
        zeta_interp = np.empty((nt, eta_rho, xi_rho))
        
        # Time in days since reference
        import pandas as pd
        ref_time_pd = pd.Timestamp(params['ref_time'])
        time_days = np.empty(nt)
        
        # Interpolate each time step
        for t in range(nt):
            idx = valid_time_indices[t]
            
            # Calculate time
            time = bc_data['time'][idx].values
            current_time = pd.to_datetime(str(time))
            time_days[t] = (current_time - ref_time_pd).total_seconds() / 86400.0
            
            # Interpolate 3D variables
            temp_data = bc_data[params['source_vars']['temp']][idx, :, :, :].values
            temp_interp[t] = self.init_tools.interpolate_and_mask_3d(
                temp_data, source_coords,
                roms_lon_2d, roms_lat_2d, roms_depth_3d,
                self.grid.mask_rho.values, fill_value=5.0
            )
            
            salt_data = bc_data[params['source_vars']['salt']][idx, :, :, :].values
            salt_interp[t] = self.init_tools.interpolate_and_mask_3d(
                salt_data, source_coords,
                roms_lon_2d, roms_lat_2d, roms_depth_3d,
                self.grid.mask_rho.values, fill_value=32.0
            )
            
            u_data = bc_data[params['source_vars']['u']][idx, :, :, :].values
            u_interp[t] = self.init_tools.interpolate_and_mask_3d(
                u_data, source_coords,
                roms_lonu_2d, roms_latu_2d, roms_depth_3d,
                self.grid.mask_u.values, fill_value=0.0
            )
            
            v_data = bc_data[params['source_vars']['v']][idx, :, :, :].values
            v_interp[t] = self.init_tools.interpolate_and_mask_3d(
                v_data, source_coords,
                roms_lonv_2d, roms_latv_2d, roms_depth_3d,
                self.grid.mask_v.values, fill_value=0.0
            )
            
            # Interpolate 2D zeta
            zeta_data = bc_data[params['source_vars']['zeta']][idx, :, :].values
            if zeta_data.ndim > 2:
                surf_idx = np.where(~np.isnan(zeta_data))[0][0]
                zeta_data = zeta_data[surf_idx, :, :]
            zeta_interp[t] = self.init_tools.interpolate_and_mask_2d(
                zeta_data, source_coords['lon_2d'], source_coords['lat_2d'],
                roms_lon_2d, roms_lat_2d, fill_value=0.0
            )
            
            if (t + 1) % 10 == 0 or t == nt - 1:
                print(f"     Interpolated {t + 1}/{nt} time steps")
        
        return {
            'temp': temp_interp,
            'salt': salt_interp,
            'u': u_interp,
            'v': v_interp,
            'zeta': zeta_interp,
            'time_days': time_days,
            'z_rho': z_rho
        }
    
    def compute_derived_variables(self, interp_data: Dict[str, np.ndarray], 
                                 params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute derived variables (ubar, vbar, w).
        
        Args:
            interp_data: Interpolated data dictionary
            params: Parameters dictionary
            
        Returns:
            Dictionary with all variables including derived ones
        """
        print("   Computing ubar, vbar, w...")
        
        nt = interp_data['temp'].shape[0]
        eta_rho, xi_rho = self.grid.lat_rho.shape
        eta_v, xi_u = self.grid.lat_v.shape[0], self.grid.lon_u.shape[1]
        s_rho = interp_data['temp'].shape[1]
        
        # Transpose z_rho for computations
        z_rho_transposed = np.transpose(interp_data['z_rho'], (2, 0, 1))
        
        ubar = np.empty((nt, eta_rho, xi_u))
        vbar = np.empty((nt, eta_v, xi_rho))
        w = np.empty((nt, s_rho, eta_rho, xi_rho))
        
        for t in range(nt):
            ubar[t], vbar[t] = self.convert_tools.compute_uvbar(
                interp_data['u'][t],
                interp_data['v'][t],
                z_rho_transposed
            )
            w[t] = self.convert_tools.compute_w(
                interp_data['u'][t],
                interp_data['v'][t],
                self.grid.pm.values,
                self.grid.pn.values,
                z_rho_transposed
            )
        
        print(f"   ✓ Computed derived variables for {nt} time steps")
        
        # Add derived variables to data dictionary
        result = dict(interp_data)
        result['ubar'] = ubar
        result['vbar'] = vbar
        result['w'] = w
        
        return result
    
    def _prepare_grid_for_netcdf(self, grid: xr.Dataset) -> xr.Dataset:
        """
        Prepare grid by converting scalar DataArrays to plain values.
        This is needed because netCDF attributes cannot be DataArrays.
        
        Args:
            grid: Original grid dataset
            
        Returns:
            Grid with scalar attributes converted
        """
        grid_copy = grid.copy(deep=False)
        
        # Convert scalar DataArray variables to plain Python types
        scalar_vars = ['theta_s', 'theta_b', 'hc', 'Tcline']
        for var in scalar_vars:
            if var in grid_copy:
                val = grid_copy[var]
                if hasattr(val, 'values'):
                    # Replace the DataArray with its scalar value as an attribute
                    scalar_value = float(val.values)
                    grid_copy = grid_copy.drop_vars(var)
                    grid_copy.attrs[var] = scalar_value
        
        return grid_copy
    
    def save_climatology(self, data: Dict[str, np.ndarray], 
                        params: Dict[str, Any]) -> str:
        """
        Save climatology file.
        
        Args:
            data: Data dictionary
            params: Parameters dictionary
            
        Returns:
            Path to saved climatology file
        """
        print("   Creating climatology dataset...")
        
        # Prepare grid with scalar attributes
        grid_prepared = self._prepare_grid_for_netcdf(self.grid)
        
        ds_clim = self.boundary_tools.create_climatology_dataset(
            data['temp'], data['salt'], data['u'], data['v'], data['w'],
            data['ubar'], data['vbar'], data['zeta'],
            data['time_days'], grid_prepared,
            source_name=params['source_name']
        )
        
        # Generate filename
        start = params['start_time'].replace('-', '')
        end = params['end_time'].replace('-', '')
        clim_file = self.output_dir / f"roms_climatology_{start}_{end}.nc"
        
        print(f"   Writing to: {clim_file}")
        ds_clim.to_netcdf(clim_file, mode='w')
        
        return str(clim_file)
    
    def save_boundary(self, data: Dict[str, np.ndarray], 
                     params: Dict[str, Any]) -> str:
        """
        Save boundary forcing file.
        
        Args:
            data: Data dictionary
            params: Parameters dictionary
            
        Returns:
            Path to saved boundary file
        """
        print("   Extracting boundary transects...")
        
        # Prepare grid with scalar attributes
        grid_prepared = self._prepare_grid_for_netcdf(self.grid)
        
        boundary_transects = self.boundary_tools.extract_boundary_transects(
            data['temp'], data['salt'], data['u'], data['v'],
            data['ubar'], data['vbar'], data['zeta'],
            grid_prepared, params['boundaries']
        )
        
        print("   Creating boundary dataset...")
        
        ds_bry = self.boundary_tools.create_boundary_dataset(
            boundary_transects, data['time_days'], grid_prepared,
            params['start_time'], params['end_time'],
            source_name=params['source_name']
        )
        
        # Generate filename
        start = params['start_time'].replace('-', '')
        end = params['end_time'].replace('-', '')
        bry_file = self.output_dir / f"roms_boundary_{start}_{end}.nc"
        
        print(f"   Writing to: {bry_file}")
        ds_bry.to_netcdf(bry_file, mode='w')
        
        return str(bry_file)


def main():
    """
    Command-line interface for the boundary conditions agent.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate ROMS boundary conditions using LLM-assisted workflow'
    )
    parser.add_argument('--model-tools-path', required=True,
                       help='Path to model-tools directory')
    parser.add_argument('--grid-file', required=True,
                       help='Path to ROMS grid file')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (optional, will prompt if not provided)')
    parser.add_argument('--prompt', default=None,
                       help='Natural language prompt (optional, will prompt if not provided)')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ROMSBoundaryAgent(
        model_tools_path=args.model_tools_path,
        grid_file=args.grid_file,
        output_dir=args.output_dir
    )
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        print("\nDescribe the boundary conditions you want to create:")
        print("Example: 'Create boundary conditions for January 2024'")
        prompt = input("\nPrompt: ").strip()
    
    # Execute workflow
    result = agent.execute_workflow(prompt)
    
    # Print results
    if result['success']:
        print("\n✅ Success!")
        print(f"\nGenerated files:")
        for key, path in result['files'].items():
            print(f"  {key}: {path}")
    else:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
