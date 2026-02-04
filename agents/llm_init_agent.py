#!/usr/bin/env python3
"""
ROMS Initial Conditions Agent with LLM Integration

This agent uses large language models to parse natural language requests
and automate ROMS initial condition file generation using the model-tools library.

The agent workflow:
1. Parses user's natural language request to extract initialization parameters
2. Prompts user for comprehensive initialization configuration:
   - Initialization time/date
   - Data source (API or local NetCDF file)
   - Variable mapping and fill values
   - Deep ocean parameters
3. Loads ocean data from specified source (GLORYS via API or NetCDF file)
4. Interpolates data to ROMS grid
5. Computes derived variables (barotropic velocities, vertical velocity)
6. Creates ROMS initial conditions NetCDF file

Usage:
    agent = ROMSInitAgent(
        model_tools_path="/path/to/model-tools",
        grid_file="/path/to/roms_grid.nc"
    )
    result = agent.execute_workflow("Initialize for January 1, 2024")
"""

import re
import os
import sys
from typing import Dict, Optional, Tuple
import json
import numpy as np
import xarray as xr
from datetime import datetime

# Try to import openai, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    print("Warning: openai library not available. Install with: pip install openai==2.6.1")

# Add model-tools to path
sys.path.insert(0, '/global/cfs/cdirs/m4304/enuss/model-tools/code')
from initialization import init_tools
from conversions import convert_tools
from grid import grid_tools


class ROMSInitAgent:
    """
    ROMS Initial Conditions Agent with LLM integration.
    
    This agent automates ROMS initial conditions file generation by:
    1. Using LLM to parse natural language requests for initialization parameters
    2. Interactively collecting configuration parameters:
       - Initialization time/date
       - Data source (Copernicus Marine API or local NetCDF)
       - Variable names mapping
       - Fill values for interpolation
       - Deep ocean parameters
    3. Loading ocean data (GLORYS) from API or file
    4. Interpolating to ROMS grid with proper masking
    5. Computing derived variables (ubar, vbar, w)
    6. Creating ROMS initial conditions NetCDF file
    
    Attributes:
        model_tools_path: Path to model-tools repository
        grid_file: Path to ROMS grid NetCDF file
        output_dir: Directory for output files
        llm: OpenAI client for LLM parsing (optional)
        model: LLM model name to use
        base_url: API base URL for LLM service
    """
    
    def __init__(self, model_tools_path: str, grid_file: str, 
                 api_key: Optional[str] = None, output_dir: Optional[str] = None,
                 model: str = "claude-haiku-4-5-20251001-v1-birthright"):
        """
        Initialize agent with paths and optional API key.
        
        Args:
            model_tools_path: Path to model-tools directory
            grid_file: Path to ROMS grid NetCDF file
            api_key: LLM API key (if not provided, will try to read from environment)
            output_dir: Directory for output files (if not provided, uses model-tools/output)
            model: Model name to use (default: claude-haiku-4-5-20251001-v1-birthright)
        """
        self.model_tools_path = model_tools_path
        self.grid_file = grid_file
        self.model = model
        self.base_url = "https://ai-incubator-api.pnnl.gov"
        
        # Validate grid file exists
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(model_tools_path, 'output')
        self.output_dir = os.path.abspath(output_dir)
        
        if not os.path.exists(self.output_dir):
            print(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Grid file: {grid_file}")
        
        # Initialize LLM client
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.llm = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                print(f"✓ LLM initialized successfully (model: {self.model})")
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                self.llm = None
        else:
            if not OPENAI_AVAILABLE:
                print("Warning: OpenAI library not installed. LLM features disabled.")
            else:
                print("Warning: No API key provided. LLM features will be disabled.")
                print("Set LLM_API_KEY environment variable to enable LLM features.")
            self.llm = None
    
    def _prompt_for_init_parameters(self, params: Dict) -> Dict:
        """
        Prompt user for initialization configuration parameters.
        
        Collects the following parameters interactively:
        - Initialization time/date
        - Data source type (API or NetCDF file)
        - NetCDF file path (if using file)
        - Time buffer for API downloads
        - Fill values for variables
        - Deep ocean parameters
        
        Only prompts for parameters not already specified in the initial request.
        All prompts include default values and input validation.
        
        Args:
            params: Dictionary with existing parameters
            
        Returns:
            Updated dictionary with all initialization parameters specified
        """
        print("\n" + "="*60)
        print("Initial Conditions Configuration")
        print("="*60)
        print("\nPlease specify initialization parameters:\n")
        
        # Prompt for initialization time
        if 'init_time' not in params or params['init_time'] is None:
            while True:
                try:
                    time_input = input("Initialization date and time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS, default: 2024-01-01T00:00:00): ").strip()
                    if not time_input:
                        params['init_time'] = np.datetime64('2024-01-01T00:00:00')
                        print("  ✓ Using default: 2024-01-01T00:00:00")
                        break
                    init_time = np.datetime64(time_input)
                    params['init_time'] = init_time
                    print(f"  ✓ Using initialization time: {init_time}")
                    break
                except ValueError:
                    print("  ⚠ Invalid date/time format. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 2024-01-01T00:00:00")
                    params['init_time'] = np.datetime64('2024-01-01T00:00:00')
                    break
        else:
            print(f"  ✓ Using init_time = {params['init_time']} (from prompt)")
        
        # Prompt for data source
        if 'use_api' not in params or params['use_api'] is None:
            while True:
                try:
                    api_input = input("Use Copernicus Marine API for data? [Y/n] (default: Y): ").strip().lower()
                    if not api_input or api_input in ('y', 'yes'):
                        params['use_api'] = True
                        print("  ✓ Will use Copernicus Marine API")
                        break
                    elif api_input in ('n', 'no'):
                        params['use_api'] = False
                        print("  ✓ Will use local NetCDF file")
                        break
                    else:
                        print("  ⚠ Please enter Y or N")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: API")
                    params['use_api'] = True
                    break
        else:
            print(f"  ✓ Using {'API' if params['use_api'] else 'NetCDF file'} (from prompt)")
        
        # If using NetCDF file, prompt for path
        if not params['use_api']:
            if 'netcdf_path' not in params or params['netcdf_path'] is None:
                while True:
                    try:
                        nc_path = input("Path to NetCDF file: ").strip()
                        if not nc_path:
                            print("  ⚠ Path cannot be empty. Please try again.")
                            continue
                        nc_path = os.path.expanduser(nc_path)
                        if not os.path.exists(nc_path):
                            print(f"  ⚠ File not found: {nc_path}")
                            retry = input("    Try again? [Y/n]: ").strip().lower()
                            if retry in ('n', 'no'):
                                print("  ⚠ Cannot proceed without valid NetCDF file")
                                sys.exit(1)
                            continue
                        params['netcdf_path'] = nc_path
                        print(f"  ✓ Using NetCDF file: {nc_path}")
                        break
                    except KeyboardInterrupt:
                        print("\n  ⚠ Interrupted. Cannot proceed without NetCDF file")
                        sys.exit(1)
            else:
                print(f"  ✓ Using NetCDF file: {params['netcdf_path']} (from prompt)")
        
        # If using API, prompt for time buffer
        if params['use_api']:
            if 'time_buffer_hours' not in params or params['time_buffer_hours'] is None:
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
            else:
                print(f"  ✓ Using time buffer: {params['time_buffer_hours']} hours (from prompt)")
        
        # Set default variable names (GLORYS standard)
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
        
        # Set default fill values
        params.setdefault('fill_values', {
            'temp': 5.0,
            'salt': 32.0,
            'u': 0.0,
            'v': 0.0,
            'zeta': 0.0
        })
        
        # Set default deep ocean parameters
        params.setdefault('new_depth', 10000.0)
        params.setdefault('deep_ocean_fill_values', {
            'zos': np.nan,
            'uo': 0.0,
            'vo': 0.0,
            'so': 35.0,
            'thetao': 1.0
        })
        
        # Set minimum temperature
        params.setdefault('min_temp', 0.1)
        
        print("\n" + "="*60)
        print("Initialization Configuration Summary")
        print("="*60)
        print(f"Initialization time: {params['init_time']}")
        print(f"Data source:         {'Copernicus Marine API' if params['use_api'] else 'NetCDF file'}")
        if not params['use_api']:
            print(f"NetCDF path:         {params['netcdf_path']}")
        else:
            print(f"Time buffer:         {params['time_buffer_hours']} hours")
        print(f"Fill values:")
        print(f"  Temperature:       {params['fill_values']['temp']} °C")
        print(f"  Salinity:          {params['fill_values']['salt']} PSU")
        print(f"  U/V velocity:      {params['fill_values']['u']}/{params['fill_values']['v']} m/s")
        print(f"  Sea surface:       {params['fill_values']['zeta']} m")
        print("="*60 + "\n")
        
        return params
    
    def parse_request_with_llm(self, prompt: str) -> Dict:
        """
        Use LLM to intelligently parse natural language prompts for initialization parameters.
        
        The LLM attempts to extract:
        - init_time: Initialization date/time (required)
        - use_api: Whether to use API vs NetCDF file (optional)
        - netcdf_path: Path to NetCDF file if not using API (optional)
        - time_buffer_hours: Buffer for API downloads in hours (optional)
        
        Args:
            prompt: User's natural language request
            
        Returns:
            Dictionary with parsed parameters
        """
        if not self.llm:
            print("LLM not available, using basic parsing...")
            return self.parse_request_basic(prompt)
        
        system_prompt = """You are a specialized assistant for ROMS ocean modeling initialization.
Your task is to extract initialization configuration parameters from user requests.

CRITICAL: ONLY extract parameters that are EXPLICITLY mentioned in the user's request.
Do NOT provide default values or make assumptions. Return null for any parameter not explicitly stated.

Extract the following information if explicitly mentioned:
1. init_time: Initialization date and time (format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) - ONLY if explicitly stated
2. use_api: Whether to use Copernicus Marine API (true) or local NetCDF file (false) - ONLY if explicitly mentioned
3. netcdf_path: Path to local NetCDF file - ONLY if a file path is explicitly provided
4. time_buffer_hours: Hours before/after init_time to download from API - ONLY if explicitly mentioned

Common date interpretations (extract if mentioned):
- "January 1, 2024" -> "2024-01-01T00:00:00"
- "start of 2024" -> "2024-01-01T00:00:00"
- "mid-July 2023" -> "2023-07-15T00:00:00"
- "December 31 2023 at noon" -> "2023-12-31T12:00:00"

For data source (use_api):
- "using API", "use Copernicus API", "from API" -> use_api: true
- "from file", "using NetCDF file", "local file" -> use_api: false
- If not mentioned at all -> use_api: null

Return ONLY a valid JSON object with the extracted parameters. Use null for any parameter not explicitly mentioned.

Example 1 - "Initialize for January 1, 2024":
{"init_time": "2024-01-01T00:00:00", "use_api": null, "netcdf_path": null, "time_buffer_hours": null}

Example 2 - "Initialize for January 1, 2024 using Copernicus API":
{"init_time": "2024-01-01T00:00:00", "use_api": true, "netcdf_path": null, "time_buffer_hours": null}

Example 3 - "Initialize for 2024-01-01 from file /path/to/data.nc":
{"init_time": "2024-01-01T00:00:00", "use_api": false, "netcdf_path": "/path/to/data.nc", "time_buffer_hours": null}"""
        
        try:
            print("🤖 Querying LLM to parse request...")
            response = self.llm.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            print(f"LLM response: {content}")
            
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                # Convert init_time string to numpy datetime64
                if params.get('init_time'):
                    params['init_time'] = np.datetime64(params['init_time'])
                print(f"✓ LLM parsed parameters:\n{json.dumps({k: str(v) if isinstance(v, np.datetime64) else v for k, v in params.items()}, indent=2)}")
                return params
            else:
                print("⚠ Could not extract JSON from LLM response. Falling back to basic parsing.")
                return self.parse_request_basic(prompt)
                
        except Exception as e:
            print(f"⚠ Error using LLM: {e}. Falling back to basic parsing.")
            return self.parse_request_basic(prompt)
    
    def parse_request_basic(self, prompt: str) -> Dict:
        """
        Extract initialization parameters from natural language prompt using regex.
        Fallback method when LLM is not available.
        
        This basic parser attempts to extract date/time information.
        All other parameters will be prompted for interactively.
        
        Args:
            prompt: Natural language request containing initialization info
            
        Returns:
            Dictionary with extracted parameters (may be empty)
        """
        result = {}
        
        # Try to extract date in various formats
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # ISO format with time
            r'(\d{4}-\d{2}-\d{2})',  # ISO date only
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',  # Month Day, Year
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    if 'T' in date_str or '-' in date_str[:10]:
                        result['init_time'] = np.datetime64(date_str)
                    else:
                        # Convert month name format
                        result['init_time'] = np.datetime64(datetime.strptime(date_str, '%B %d, %Y').strftime('%Y-%m-%d'))
                    break
                except:
                    continue
        
        return result
    
    def generate_initial_conditions(self, params: Dict, output_file: str = "initial_conditions.nc") -> str:
        """
        Generate ROMS initial conditions file.
        
        Steps:
        1. Load ROMS grid
        2. Load ocean data from API or NetCDF file
        3. Add deep ocean layer to prevent extrapolation
        4. Interpolate all variables to ROMS grid
        5. Compute derived variables (ubar, vbar, w)
        6. Create and save initial conditions dataset
        
        Args:
            params: Dictionary with all initialization parameters
            output_file: Output filename
            
        Returns:
            Path to generated initial conditions file
        """
        print(f"\n⚙️  Generating ROMS initial conditions...")
        
        # Load grid
        print(f"   Loading ROMS grid from {self.grid_file}...")
        grid = xr.open_dataset(self.grid_file)
        
        # Auto-detect lon/lat ranges from grid for API
        if params['use_api']:
            lon_min, lon_max = float(grid.lon_rho.min()), float(grid.lon_rho.max())
            lat_min, lat_max = float(grid.lat_rho.min()), float(grid.lat_rho.max())
            
            # Add buffer for interpolation
            lon_buffer = (lon_max - lon_min) * 0.1
            lat_buffer = (lat_max - lat_min) * 0.1
            
            lon_range = (lon_min - lon_buffer, lon_max + lon_buffer)
            lat_range = (lat_min - lat_buffer, lat_max + lat_buffer)
            
            print(f"   Auto-detected spatial extent from grid:")
            print(f"     Longitude: {lon_range}")
            print(f"     Latitude: {lat_range}")
        else:
            lon_range = None
            lat_range = None
        
        # Load initialization data
        print(f"\n   Loading ocean data...")
        init_data = init_tools.load_glorys_data(
            init_time=params['init_time'],
            lon_range=lon_range,
            lat_range=lat_range,
            use_api=params['use_api'],
            netcdf_path=params.get('netcdf_path'),
            time_buffer_hours=params.get('time_buffer_hours', 24)
        )
        
        # Add deep ocean layer
        print(f"   Adding deep ocean layer...")
        init_data = init_tools.add_deep_ocean_layer(
            init_data, params['new_depth'], params['deep_ocean_fill_values'],
            time_var='time', depth_var=params['source_vars']['depth'],
            lat_var=params['source_vars']['lat'], lon_var=params['source_vars']['lon']
        )
        
        # Compute time since reference
        seconds_since_2000, days_since_2000 = init_tools.compute_time_since_reference(params['init_time'])
        print(f"   Initialization time: {params['init_time']}")
        print(f"   Days since 2000-01-01: {days_since_2000:.6f}")
        
        # Select time step
        time_idx = np.argmin(np.abs(init_data.time.values - params['init_time']))
        source_time_step = init_data.isel(time=time_idx)
        
        # Get grid dimensions
        eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)
        
        # Prepare source coordinates
        source_coords = init_tools.prepare_source_coords(
            init_data, params['source_vars']['depth'], 
            params['source_vars']['lat'], params['source_vars']['lon']
        )
        
        # Compute ROMS vertical coordinates
        print(f"   Computing vertical coordinates...")
        # Get hc from grid or use from params or default
        if hasattr(grid, 'hc'):
            hc = float(grid.hc.values) if hasattr(grid.hc, 'values') else float(grid.hc)
        else:
            hc = params.get('hc', 100.0)
            print(f"   Warning: hc not found in grid file, using {hc} m")
        
        z_rho = grid_tools.compute_z(
            grid.sigma_r.values, hc, grid.Cs_r.values,
            grid.h.values, np.zeros((eta_rho, xi_rho, 1))
        )
        z_rho = np.squeeze(z_rho)  # Remove singleton dimension
        z_rho = np.transpose(z_rho, (1, 2, 0))  # Shape: (eta_rho, xi_rho, s_rho)
        
        # Get ROMS target coordinates
        roms_coords = {
            'lon_rho': grid.lon_rho.values,
            'lat_rho': grid.lat_rho.values,
            'lon_u': grid.lon_u.values,
            'lat_u': grid.lat_u.values,
            'lon_v': grid.lon_v.values,
            'lat_v': grid.lat_v.values,
            'z_rho': -z_rho  # Negative for interpolation
        }
        
        # Interpolate temperature
        print(f"\n   Interpolating Temperature...")
        temp_interp = init_tools.interpolate_and_mask_3d(
            source_time_step[params['source_vars']['temp']].values,
            source_coords,
            roms_coords['lon_rho'], roms_coords['lat_rho'], roms_coords['z_rho'],
            grid.mask_rho.values.astype(bool),
            params['fill_values']['temp'],
            interp_method='linear',
            min_value=params['min_temp']
        )
        print(f"     Shape: {temp_interp.shape}")
        
        # Interpolate salinity
        print(f"   Interpolating Salinity...")
        sal_interp = init_tools.interpolate_and_mask_3d(
            source_time_step[params['source_vars']['salt']].values,
            source_coords,
            roms_coords['lon_rho'], roms_coords['lat_rho'], roms_coords['z_rho'],
            grid.mask_rho.values.astype(bool),
            params['fill_values']['salt'],
            interp_method='linear'
        )
        print(f"     Shape: {sal_interp.shape}")
        
        # Interpolate U velocity
        print(f"   Interpolating U Velocity...")
        u_interp = init_tools.interpolate_and_mask_3d(
            source_time_step[params['source_vars']['u']].values,
            source_coords,
            roms_coords['lon_u'], roms_coords['lat_u'], roms_coords['z_rho'],
            grid.mask_u.values.astype(bool),
            params['fill_values']['u'],
            interp_method='linear'
        )
        print(f"     Shape: {u_interp.shape}")
        
        # Interpolate V velocity
        print(f"   Interpolating V Velocity...")
        v_interp = init_tools.interpolate_and_mask_3d(
            source_time_step[params['source_vars']['v']].values,
            source_coords,
            roms_coords['lon_v'], roms_coords['lat_v'], roms_coords['z_rho'],
            grid.mask_v.values.astype(bool),
            params['fill_values']['v'],
            interp_method='linear'
        )
        print(f"     Shape: {v_interp.shape}")
        
        # Interpolate sea surface height
        print(f"   Interpolating Sea Surface Height...")
        zeta_data = source_time_step[params['source_vars']['zeta']].values
        
        # Handle 3D zeta data (select surface)
        zeta_shape = zeta_data.shape
        expected_y = source_coords['lat_2d'].shape[0]
        expected_x = source_coords['lon_2d'].shape[1]
        
        if len(zeta_shape) == 3:
            for i, dim_size in enumerate(zeta_shape):
                if dim_size != expected_y and dim_size != expected_x:
                    zeta_data = np.take(zeta_data, 0, axis=i)
                    break
        
        zeta_interp = init_tools.interpolate_and_mask_2d(
            zeta_data,
            source_coords['lon_2d'], source_coords['lat_2d'],
            roms_coords['lon_rho'], roms_coords['lat_rho'],
            params['fill_values']['zeta'],
            interp_method='linear'
        )
        print(f"     Shape: {zeta_interp.shape}")
        
        # Compute barotropic velocities
        print(f"\n   Computing Barotropic Velocities...")
        # Reshape z_rho from (ny, nx, nz) to (nz, ny, nx) for compute_uvbar
        z_rho_transposed = np.transpose(z_rho, (2, 0, 1))
        ubar, vbar = convert_tools.compute_uvbar(u_interp, v_interp, z_rho_transposed)
        print(f"     Ubar shape: {ubar.shape}")
        print(f"     Vbar shape: {vbar.shape}")
        
        # Compute vertical velocity
        print(f"   Computing Vertical Velocity...")
        w = convert_tools.compute_w(u_interp, v_interp, grid.pm.values, grid.pn.values, z_rho_transposed)
        print(f"     W shape: {w.shape}")
        
        # Create initial conditions dataset
        print(f"\n   Creating Initial Conditions Dataset...")
        ds = init_tools.create_initial_conditions_dataset(
            grid, temp_interp, sal_interp, u_interp, v_interp, w,
            ubar, vbar, zeta_interp, days_since_2000, params['init_time'],
            source_name='GLORYS'
        )
        
        # Save to file
        output_path = os.path.join(self.output_dir, output_file)
        print(f"   Saving to {output_path}...")
        ds.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
        
        print(f"✓ Initial conditions saved to: {output_path}")
        
        return output_path
    
    def execute_workflow(self, prompt: str) -> Dict:
        """
        Main workflow: parse prompt -> prompt for parameters -> generate initial conditions.
        
        Workflow steps:
        1. Parse natural language prompt to extract initialization parameters
        2. Interactively prompt user for configuration:
           - Initialization time/date
           - Data source (API or NetCDF file)
           - Fill values and deep ocean parameters
        3. Load ocean data from specified source
        4. Interpolate to ROMS grid
        5. Compute derived variables
        6. Create and save initial conditions file
        
        Args:
            prompt: Natural language description of initialization request.
                   Examples:
                   - "Initialize for January 1, 2024"
                   - "Create initial conditions for start of 2024"
                   - "Initialize model for mid-July 2023"
            
        Returns:
            Dictionary with workflow results including:
            - success: Boolean indicating if workflow completed
            - parameters: Dict of all parameters used
            - init_file: Path to generated initial conditions file
            - message: Status message
            - error: Error message (if workflow failed)
        """
        print("=" * 60)
        print("ROMS Initial Conditions Agent with LLM")
        print("=" * 60)
        print(f"\n📝 User request: {prompt}\n")
        
        # Step 1: Parse request using LLM
        params = self.parse_request_with_llm(prompt)
        
        # Step 2: Prompt user for initialization parameters
        params = self._prompt_for_init_parameters(params)
        
        # Step 3: Generate initial conditions
        try:
            init_file = self.generate_initial_conditions(params)
        except Exception as e:
            import traceback
            print(f"\n⚠ Error generating initial conditions: {e}")
            traceback.print_exc()
            return {"error": f"Initial conditions generation failed: {str(e)}"}
        
        print("\n" + "=" * 60)
        print("✅ ROMS Initial Conditions Generation Complete!")
        print("=" * 60)
        
        return {
            "success": True,
            "parameters": {k: str(v) if isinstance(v, np.datetime64) else v for k, v in params.items()},
            "init_file": init_file,
            "message": "Initial conditions created successfully"
        }


def main():
    """
    Main function for testing the initialization agent.
    
    Example usage:
        # Set API key in environment
        export LLM_API_KEY=<your-key>
        
        # Run the agent
        python llm_init_agent.py
    """
    # Initialize agent
    agent = ROMSInitAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        grid_file="/global/cfs/cdirs/m4304/enuss/model-tools/output/roms_grid.nc"
    )
    
    # Test prompts
    test_prompts = [
        "Initialize for January 1, 2024",
        "Create initial conditions for start of 2024 using API",
        "Initialize model for 2024-01-01",
    ]
    
    print("\nNote: After parsing, you will be prompted to enter:")
    print("  - Initialization date/time")
    print("  - Data source (API or NetCDF file)")
    print("  - File path (if using NetCDF file)\n")
    
    result = agent.execute_workflow(test_prompts[0])
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
