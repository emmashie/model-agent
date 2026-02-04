#!/usr/bin/env python3
"""
ROMS Surface Forcing Agent with LLM Integration

This agent automates the generation of surface forcing files for ROMS,
including:
- ERA5 data download or loading from NetCDF
- Processing and unit conversions
- Interpolation to ROMS grid
- Creation of surface forcing files

Usage:
    from agents.llm_forcing_agent import ROMSSurfaceForcingAgent
    
    agent = ROMSSurfaceForcingAgent(
        model_tools_path="/path/to/model-tools",
        grid_file="/path/to/roms_grid.nc"
    )
    
    result = agent.execute_workflow(
        "Create surface forcing from January 1-31, 2024"
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


class ROMSSurfaceForcingAgent:
    """
    Agent for generating ROMS surface forcing files using LLM-assisted parsing.
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
        Initialize the ROMS Surface Forcing Agent.
        
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
            from forcing import forcing_tools
            from conversions import convert_tools
            from interpolate import interp_tools
            from grid import grid_tools
            
            self.forcing_tools = forcing_tools()
            self.convert_tools = convert_tools()
            self.interp_tools = interp_tools
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
    
    def parse_request_with_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Parse time range and options from natural language using LLM.
        
        Args:
            prompt: Natural language request
            
        Returns:
            Dictionary with parsed parameters
        """
        system_prompt = """You are a helpful assistant that extracts parameters from natural language requests for ocean model surface forcing.

Extract the following information:
1. start_time: Start date/time in YYYY-MM-DD format
2. end_time: End date/time in YYYY-MM-DD format
3. use_api: Whether to use CDS API for data download (true/false, default: null - will prompt)
4. era5_data_path: Path to existing ERA5 data files (null if using API)
5. include_radiation: Whether to include radiation variables (true/false, default: true)
6. time_resolution: Time resolution - '1hour', '3hour', '6hour', or 'daily' (default: '6hour')

Return ONLY a JSON object with these keys. Set to null if not mentioned.

Examples:
- "Create surface forcing for January 2024" → {"start_time": "2024-01-01", "end_time": "2024-01-31", "use_api": null, "era5_data_path": null, "include_radiation": true, "time_resolution": "6hour"}
- "Generate forcing from 2024-01-01 to 2024-01-31, hourly resolution" → {"start_time": "2024-01-01", "end_time": "2024-01-31", "use_api": null, "era5_data_path": null, "include_radiation": true, "time_resolution": "1hour"}
- "Make forcing for summer 2024, no radiation, daily" → {"start_time": "2024-06-01", "end_time": "2024-08-31", "use_api": null, "era5_data_path": null, "include_radiation": false, "time_resolution": "daily"}
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
    
    def parse_request_basic(self, prompt: str) -> Dict[str, Any]:
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
            'use_api': None,
            'era5_data_path': None,
            'include_radiation': True,
            'time_resolution': '6hour'
        }
        
        # Try to extract dates in YYYY-MM-DD format
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        dates = re.findall(date_pattern, prompt)
        
        if len(dates) >= 2:
            params['start_time'] = dates[0].replace('/', '-')
            params['end_time'] = dates[1].replace('/', '-')
        elif len(dates) == 1:
            params['start_time'] = dates[0].replace('/', '-')
        
        # Check for radiation keywords
        if 'no radiation' in prompt.lower() or 'skip radiation' in prompt.lower():
            params['include_radiation'] = False
        
        # Check for API keywords
        if 'use api' in prompt.lower() or 'download' in prompt.lower():
            params['use_api'] = True
        elif 'local' in prompt.lower() or 'file' in prompt.lower():
            params['use_api'] = False
        
        # Check for time resolution
        if 'hourly' in prompt.lower() or '1hour' in prompt.lower() or '1-hour' in prompt.lower():
            params['time_resolution'] = '1hour'
        elif '3hour' in prompt.lower() or '3-hour' in prompt.lower():
            params['time_resolution'] = '3hour'
        elif 'daily' in prompt.lower():
            params['time_resolution'] = 'daily'
        
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
        print("Surface Forcing Configuration")
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
                api_input = input("\nUse Copernicus Climate Data Store API for data download? (y/n, default: y): ").strip().lower()
                if not api_input or api_input == 'y':
                    params['use_api'] = True
                    params['era5_data_path'] = None
                    print("  ✓ Will use Copernicus CDS API")
                    break
                elif api_input == 'n':
                    params['use_api'] = False
                    era5_path = input("  Path to ERA5 data files (pattern with * or dict): ").strip()
                    params['era5_data_path'] = era5_path
                    print(f"  ✓ Using local data: {era5_path}")
                    break
                else:
                    print("  ⚠ Please enter 'y' or 'n'")
        else:
            if params['use_api']:
                print("  ✓ Using Copernicus CDS API (from prompt)")
            else:
                print(f"  ✓ Using local data (from prompt)")
        
        # Prompt for time resolution
        res_input = input("\nTime resolution (1hour/3hour/6hour/daily, default: 6hour): ").strip().lower()
        if res_input in ['1hour', '3hour', '6hour', 'daily']:
            params['time_resolution'] = res_input
        elif not res_input:
            params['time_resolution'] = params.get('time_resolution', '6hour')
        print(f"  ✓ Time resolution: {params['time_resolution']}")
        
        # Prompt for radiation
        if params.get('include_radiation') is None:
            rad_input = input("\nInclude radiation variables? (y/n, default: y): ").strip().lower()
            if not rad_input or rad_input == 'y':
                params['include_radiation'] = True
                print("  ✓ Will include radiation variables")
            else:
                params['include_radiation'] = False
                print("  ✓ Will skip radiation variables")
        else:
            status = "included" if params['include_radiation'] else "excluded"
            print(f"  ✓ Radiation: {status} (from prompt)")
        
        # Convert time resolution to API hours format
        resolution_map = {
            '1hour': [f"{h:02d}:00" for h in range(24)],
            '3hour': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
            '6hour': ['00:00', '06:00', '12:00', '18:00'],
            'daily': ['12:00']
        }
        params['api_hours'] = resolution_map.get(params['time_resolution'], resolution_map['6hour'])
        
        # Set ERA5 variable names
        params.setdefault('era5_vars', {
            'shortwave': 'ssrd',
            'shortwave_net': 'ssrd',
            'longwave': 'strd',
            'longwave_net': 'str',
            'sst': 'sst',
            'airtemp': 't2m',
            'dewpoint': 'd2m',
            'precip': 'tp',
            'u10': 'u10',
            'v10': 'v10',
            'press': 'msl'
        })
        
        # Set reference time for ROMS
        params.setdefault('ref_time', '2000-01-01T00:00:00')
        
        return params
    
    def execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the complete surface forcing generation workflow.
        
        Args:
            prompt: Natural language description of forcing request
            
        Returns:
            Dictionary with results and file paths
        """
        print("\n" + "="*70)
        print("ROMS Surface Forcing Agent")
        print("="*70)
        print(f"\nRequest: {prompt}")
        
        # Step 1: Parse request
        print("\n" + "="*70)
        print("Step 1: Parsing Request")
        print("="*70)
        
        if self.client:
            print("🤖 Querying LLM to parse request...")
            params = self.parse_request_with_llm(prompt)
        else:
            print("📝 Using basic regex parsing...")
            params = self.parse_request_basic(prompt)
        
        # Step 2: Prompt for missing parameters
        params = self.prompt_for_parameters(params)
        
        # Step 3: Load or download ERA5 data
        print("\n" + "="*70)
        print("Step 2: Loading ERA5 Data")
        print("="*70)
        
        try:
            if params['use_api']:
                print("📥 Downloading data from Copernicus Climate Data Store API...")
                era5_data = self.download_era5_data(params)
            else:
                print(f"📂 Loading data from: {params['era5_data_path']}")
                era5_data = self.forcing_tools.load_era5_data(
                    time_range=(np.datetime64(params['start_time']), np.datetime64(params['end_time'])),
                    lon_range=None,
                    lat_range=None,
                    use_api=False,
                    netcdf_paths=params['era5_data_path']
                )
            
            print(f"✓ Loaded ERA5 data")
            print(f"   Data variables: {list(era5_data.data_vars)}")
            print(f"   Dimensions: {list(era5_data.dims)}")
        except Exception as e:
            return {'success': False, 'error': f"Failed to load ERA5 data: {e}"}
        
        # Step 4: Process ERA5 variables
        print("\n" + "="*70)
        print("Step 3: Processing ERA5 Variables")
        print("="*70)
        
        try:
            processed_vars = self.process_era5_variables(era5_data, params)
        except Exception as e:
            return {'success': False, 'error': f"Failed to process ERA5 variables: {e}"}
        
        # Step 5: Interpolate to ROMS grid
        print("\n" + "="*70)
        print("Step 4: Interpolating to ROMS Grid")
        print("="*70)
        
        try:
            interpolated_vars = self.interpolate_to_grid(processed_vars, era5_data, params)
        except Exception as e:
            return {'success': False, 'error': f"Failed to interpolate data: {e}"}
        
        # Step 6: Save output file
        print("\n" + "="*70)
        print("Step 5: Saving Surface Forcing File")
        print("="*70)
        
        try:
            forcing_file = self.save_forcing_file(interpolated_vars, processed_vars['time'], params)
            print(f"✓ Surface forcing saved: {forcing_file}")
        except Exception as e:
            return {'success': False, 'error': f"Failed to save output file: {e}"}
        
        # Success!
        print("\n" + "="*70)
        print("✅ Surface Forcing Generated Successfully!")
        print("="*70)
        
        return {
            'success': True,
            'file': forcing_file,
            'parameters': params,
            'grid_file': self.grid_file
        }
    
    def download_era5_data(self, params: Dict[str, Any]) -> xr.Dataset:
        """
        Download ERA5 data using Copernicus CDS API.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            xarray Dataset with ERA5 data
        """
        # Get grid bounds
        lat_min = float(self.grid.lat_rho.min())
        lat_max = float(self.grid.lat_rho.max())
        lon_min = float(self.grid.lon_rho.min())
        lon_max = float(self.grid.lon_rho.max())
        
        # Add buffer to ensure we have enough data
        lon_buffer = (lon_max - lon_min) * 0.1
        lat_buffer = (lat_max - lat_min) * 0.1
        lat_range = (lat_min - lat_buffer, lat_max + lat_buffer)
        lon_range = (lon_min - lon_buffer, lon_max + lon_buffer)
        
        print(f"   Downloading region: lat [{lat_range[0]:.2f}, {lat_range[1]:.2f}], "
              f"lon [{lon_range[0]:.2f}, {lon_range[1]:.2f}]")
        print(f"   Time range: {params['start_time']} to {params['end_time']}")
        print(f"   Time resolution: {params['time_resolution']}")
        
        # Download data using forcing_tools
        era5_data = self.forcing_tools.load_era5_data(
            time_range=(np.datetime64(params['start_time']), np.datetime64(params['end_time'])),
            lon_range=lon_range,
            lat_range=lat_range,
            use_api=True,
            hours=params['api_hours'],
            include_radiation=params['include_radiation']
        )
        
        # Handle different time dimension names
        time_dim = 'time' if 'time' in era5_data.dims else 'valid_time'
        print(f"   ✓ Downloaded {len(era5_data[time_dim])} time steps")
        
        return era5_data
    
    def process_era5_variables(self, era5_data: xr.Dataset, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process ERA5 variables with unit conversions.
        
        Args:
            era5_data: ERA5 dataset
            params: Parameters dictionary
            
        Returns:
            Dictionary with processed arrays
        """
        # Handle both 'time' and 'valid_time' dimension names
        time_dim = 'time' if 'time' in era5_data.dims else 'valid_time'
        
        # Get time coordinate
        time = era5_data[time_dim]
        start_time = np.datetime64(params['start_time'])
        end_time = np.datetime64(params['end_time'])
        time = time[(time >= start_time) & (time <= end_time)]
        
        # Compute dt (time step in seconds)
        if len(time) > 1:
            dt = (time[1] - time[0]).astype('timedelta64[s]').values.astype(float)
        else:
            dt = 3600.0  # Default 1 hour
        
        vars_map = params['era5_vars']
        
        print("   Processing variables with unit conversions...")
        
        # Process radiation if available
        if params['include_radiation'] and vars_map['shortwave'] in era5_data:
            print("     - Shortwave radiation")
            swdn = self.convert_tools.convert_to_flux_density(
                era5_data[vars_map['shortwave']].sel({time_dim: slice(start_time, end_time)}).values, dt
            )
            swrad = swdn - swdn * 0.06  # Apply albedo correction
            
            print("     - Longwave radiation")
            lwrad = self.convert_tools.convert_to_flux_density(
                era5_data[vars_map['longwave_net']].sel({time_dim: slice(start_time, end_time)}).values, dt
            )
        else:
            # Create zero arrays if radiation not included
            nt = len(time)
            nlat = len(era5_data.latitude)
            nlon = len(era5_data.longitude)
            swrad = np.zeros((nt, nlat, nlon))
            lwrad = np.zeros((nt, nlat, nlon))
            print("     - Radiation variables set to zero")
        
        # Air temperature
        print("     - Air temperature")
        Tair = self.convert_tools.convert_K_to_C(
            era5_data[vars_map['sst']].sel({time_dim: slice(start_time, end_time)}).values
        )
        Tair = np.nan_to_num(Tair, nan=np.nanmean(Tair))
        
        # Relative humidity
        print("     - Relative humidity")
        qair = self.convert_tools.compute_relative_humidity(
            era5_data[vars_map['airtemp']].sel({time_dim: slice(start_time, end_time)}).values,
            era5_data[vars_map['dewpoint']].sel({time_dim: slice(start_time, end_time)}).values
        )
        
        # Pressure
        print("     - Air pressure")
        Pair = self.convert_tools.convert_Pa_to_mbar(
            era5_data[vars_map['press']].sel({time_dim: slice(start_time, end_time)}).values
        )
        
        # Precipitation
        print("     - Precipitation")
        rain_rate = self.convert_tools.compute_rainfall_cm_per_day(
            era5_data[vars_map['precip']].sel({time_dim: slice(start_time, end_time)}).values, dt
        )
        # Convert cm/day to m/s
        rain = np.array(rain_rate) * 0.01 / 86400
        
        # Wind components
        print("     - Wind components")
        uwnd = self.convert_tools.calculate_surface_wind(
            era5_data[vars_map['u10']].sel({time_dim: slice(start_time, end_time)}).values
        )
        vwnd = self.convert_tools.calculate_surface_wind(
            era5_data[vars_map['v10']].sel({time_dim: slice(start_time, end_time)}).values
        )
        
        print(f"   ✓ Processed {len(time)} time steps")
        
        return {
            'swrad': swrad,
            'lwrad': lwrad,
            'Tair': Tair,
            'qair': qair,
            'Pair': Pair,
            'rain': rain,
            'Uwind': uwnd,
            'Vwind': vwnd,
            'time': time
        }
    
    def interpolate_to_grid(self, processed_vars: Dict[str, np.ndarray], 
                           era5_data: xr.Dataset, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Interpolate processed variables to ROMS grid.
        
        Args:
            processed_vars: Dictionary with processed ERA5 variables
            era5_data: Original ERA5 dataset (for coordinates)
            params: Parameters dictionary
            
        Returns:
            Dictionary with interpolated arrays
        """
        print("   Preparing coordinates...")
        source_coords = self.forcing_tools.prepare_forcing_coords(
            era5_data, lat_var='latitude', lon_var='longitude'
        )
        
        target_lon = self.grid['lon_rho'].values
        target_lat = self.grid['lat_rho'].values
        
        # Prepare variables dictionary for batch interpolation
        variables_to_interp = {
            key: val for key, val in processed_vars.items() if key != 'time'
        }
        
        print("   Interpolating forcing variables to ROMS grid...")
        interpolated_vars = self.forcing_tools.interpolate_forcing_timeseries(
            variables_to_interp,
            source_coords['lon_2d'], source_coords['lat_2d'],
            target_lon, target_lat,
            interp_method='linear',
            verbose=True
        )
        
        print(f"   ✓ Interpolated {len(variables_to_interp)} variables")
        
        return interpolated_vars
    
    def save_forcing_file(self, interpolated_vars: Dict[str, np.ndarray], 
                         time: xr.DataArray, params: Dict[str, Any]) -> str:
        """
        Save surface forcing file.
        
        Args:
            interpolated_vars: Dictionary with interpolated variables
            time: Time coordinate
            params: Parameters dictionary
            
        Returns:
            Path to saved forcing file
        """
        print("   Creating surface forcing dataset...")
        
        grid_dims = (self.grid['lat_rho'].shape[0], self.grid['lat_rho'].shape[1])
        
        ds = self.forcing_tools.create_surface_forcing_dataset(
            interpolated_vars,
            time,
            grid_dims,
            ref_time=params['ref_time'],
            source_name='ERA5',
            add_zero_fields=True
        )
        
        # Generate filename
        start = params['start_time'].replace('-', '')
        end = params['end_time'].replace('-', '')
        forcing_file = self.output_dir / f"roms_forcing_{start}_{end}.nc"
        
        print(f"   Writing to: {forcing_file}")
        ds.to_netcdf(
            forcing_file, 
            format='NETCDF4', 
            encoding={var: {'_FillValue': np.nan} for var in ds.data_vars}
        )
        
        return str(forcing_file)


def main():
    """
    Command-line interface for the surface forcing agent.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate ROMS surface forcing using LLM-assisted workflow'
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
    agent = ROMSSurfaceForcingAgent(
        model_tools_path=args.model_tools_path,
        grid_file=args.grid_file,
        output_dir=args.output_dir
    )
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        print("\nDescribe the surface forcing you want to create:")
        print("Example: 'Create surface forcing for January 2024'")
        prompt = input("\nPrompt: ").strip()
    
    # Execute workflow
    result = agent.execute_workflow(prompt)
    
    # Print results
    if result['success']:
        print("\n✅ Success!")
        print(f"\nGenerated file: {result['file']}")
    else:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
