#!/usr/bin/env python3
"""
RODMS (Regional Ocean Data Modeling System) Agent with LLM Integration

This agent uses large language models to parse natural language requests
and automate ROMS grid generation workflows using the model-tools library.

The agent workflow:
1. Parses user's natural language request to extract region bounds
2. Prompts user for comprehensive grid configuration parameters:
   - Horizontal resolution (dx, dy in degrees)
   - Vertical levels (N_layers)
   - Vertical stretching parameters (theta_s, theta_b, hc)
   - Bathymetry smoothing parameters (initial_smooth_sigma, hmin, rx0_thresh,
     max_iter, smooth_sigma, buffer_size)
3. Downloads bathymetry data for the specified region
4. Generates ROMS-compliant grid with all specified parameters
5. Creates visualization plots of the generated grid

Usage:
    agent = ROMSGridAgent(model_tools_path="/path/to/model-tools")
    result = agent.execute_workflow("Create a grid for Chesapeake Bay")
"""

import re
import os
import sys
from typing import Dict, Optional
import json
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
from download import Downloader
from grid import grid_tools


class ROMSGridAgent:
    """
    ROMS Grid Agent with LLM integration for natural language grid generation.
    
    This agent automates ROMS grid generation by:
    1. Using LLM to parse natural language requests for region bounds
    2. Interactively collecting comprehensive grid configuration parameters:
       - Horizontal resolution (dx, dy in degrees)  
       - Vertical levels and stretching (N_layers, theta_s, theta_b, hc)
       - Bathymetry smoothing parameters (initial_smooth_sigma, hmin, rx0_thresh,
         max_iter, smooth_sigma, buffer_size)
    3. Downloading bathymetry data from SRTM15+
    4. Generating ROMS-compliant NetCDF grid files
    5. Creating visualization plots
    
    Attributes:
        model_tools_path: Path to model-tools repository
        output_dir: Directory for output files
        llm: OpenAI client for LLM parsing (optional)
        model: LLM model name to use
        base_url: API base URL for LLM service
        downloader: Downloader instance for bathymetry data
    """
    
    def __init__(self, model_tools_path: str, api_key: Optional[str] = None, 
                 output_dir: Optional[str] = None, model: str = "claude-haiku-4-5-20251001-v1-birthright"):
        """
        Initialize agent with path to model-tools repository and optional API key.
        
        Args:
            model_tools_path: Path to model-tools directory
            api_key: LLM API key (if not provided, will try to read from environment)
            output_dir: Directory for output files (if not provided, will prompt user)
            model: Model name to use (default: gpt-4o-birthright)
        """
        self.model_tools_path = model_tools_path
        self.model = model
        self.base_url = "https://ai-incubator-api.pnnl.gov"
        
        # Set or prompt for output directory
        if output_dir is None:
            output_dir = self._prompt_for_output_dir()
        
        # Validate and create output directory if needed
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if not os.path.exists(self.output_dir):
            print(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
        
        # Initialize LLM client
        self.api_key = api_key or os.getenv('LLM_API_KEY') or "sk-GnJ20ijdjJyZ5QuqFmj2ag"
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
        
        # Initialize model-tools components
        self.downloader = Downloader()
        print("✓ Model tools initialized")
    
    def _prompt_for_output_dir(self) -> str:
        """
        Prompt user for output directory path.
        
        Returns:
            Path to output directory
        """
        print("\n" + "="*60)
        print("Output Directory Configuration")
        print("="*60)
        print("\nWhere should the output NetCDF files be saved?")
        print("Examples:")
        print("  - Current directory: . or ./")
        print("  - Model-tools output: /global/cfs/cdirs/m4304/enuss/model-tools/output")
        print("  - Custom path: /path/to/your/output")
        
        while True:
            try:
                output_dir = input("\nEnter output directory path: ").strip()
                
                # Handle empty input
                if not output_dir:
                    print("⚠ Output directory cannot be empty. Please try again.")
                    continue
                
                # Expand user home directory if needed
                output_dir = os.path.expanduser(output_dir)
                
                # Convert to absolute path
                if not os.path.isabs(output_dir):
                    output_dir = os.path.abspath(output_dir)
                
                # Check if directory exists or can be created
                if os.path.exists(output_dir):
                    if not os.path.isdir(output_dir):
                        print(f"⚠ {output_dir} exists but is not a directory. Please try again.")
                        continue
                    print(f"✓ Using existing directory: {output_dir}")
                else:
                    # Ask for confirmation to create
                    create = input(f"Directory doesn't exist. Create {output_dir}? [Y/n]: ").strip().lower()
                    if create in ('', 'y', 'yes'):
                        print(f"✓ Will create directory: {output_dir}")
                    else:
                        print("Please enter a different path.")
                        continue
                
                return output_dir
                
            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted. Using current directory as default.")
                return os.getcwd()
            except Exception as e:
                print(f"⚠ Error: {e}. Please try again.")
        
    def _prompt_for_grid_parameters(self, params: Dict) -> Dict:
        """
        Prompt user for comprehensive grid configuration parameters.
        
        Collects the following parameters interactively:
        - Horizontal resolution: dx_deg, dy_deg (degrees)
        - Vertical configuration: N_layers, theta_s, theta_b, hc
        - Bathymetry smoothing: initial_smooth_sigma, hmin, rx0_thresh,
          max_iter, smooth_sigma, buffer_size
        
        Only prompts for parameters not already specified in the initial request.
        All prompts include default values and input validation.
        
        Args:
            params: Dictionary with existing parameters (at minimum lat/lon ranges)
            
        Returns:
            Updated dictionary with all grid parameters specified
        """
        print("\n" + "="*60)
        print("Grid Configuration")
        print("="*60)
        print(f"\nRegion: Lat [{params['lat_min']:.2f}° to {params['lat_max']:.2f}°], "
              f"Lon [{params['lon_min']:.2f}° to {params['lon_max']:.2f}°]")
        print("\nPlease specify grid parameters:\n")
        
        # Prompt for number of vertical levels
        while True:
            try:
                n_layers_input = input("Number of vertical levels (default: 50): ").strip()
                if not n_layers_input:
                    params['N_layers'] = 50
                    print("  ✓ Using default: 50 vertical levels")
                    break
                n_layers = int(n_layers_input)
                if n_layers < 1:
                    print("  ⚠ Number of vertical levels must be at least 1. Please try again.")
                    continue
                params['N_layers'] = n_layers
                print(f"  ✓ Using {n_layers} vertical levels")
                break
            except ValueError:
                print("  ⚠ Please enter a valid integer.")
            except KeyboardInterrupt:
                print("\n  ⚠ Interrupted. Using default: 50 vertical levels")
                params['N_layers'] = 50
                break
        
        # Prompt for horizontal resolution in X direction (longitude)
        while True:
            try:
                dx_input = input("Horizontal resolution in X (longitude) in degrees (e.g., 0.01 = ~1 km): ").strip()
                if not dx_input:
                    print("  ⚠ Resolution cannot be empty. Please try again.")
                    continue
                dx = float(dx_input)
                if dx <= 0:
                    print("  ⚠ Resolution must be positive. Please try again.")
                    continue
                params['dx_deg'] = dx
                print(f"  ✓ Using {dx}° resolution in X direction")
                break
            except ValueError:
                print("  ⚠ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n  ⚠ Interrupted. Using default: 0.01°")
                params['dx_deg'] = 0.01
                break
        
        # Prompt for horizontal resolution in Y direction (latitude)
        while True:
            try:
                dy_input = input("Horizontal resolution in Y (latitude) in degrees (e.g., 0.01 = ~1 km): ").strip()
                if not dy_input:
                    print("  ⚠ Resolution cannot be empty. Please try again.")
                    continue
                dy = float(dy_input)
                if dy <= 0:
                    print("  ⚠ Resolution must be positive. Please try again.")
                    continue
                params['dy_deg'] = dy
                print(f"  ✓ Using {dy}° resolution in Y direction")
                break
            except ValueError:
                print("  ⚠ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n  ⚠ Interrupted. Using default: 0.01°")
                params['dy_deg'] = 0.01
                break
        
        # Prompt for vertical stretching parameters if not already specified
        print("\nVertical Stretching Parameters:\n")
        
        # theta_s - surface stretching parameter
        if 'theta_s' not in params or params['theta_s'] is None:
            while True:
                try:
                    theta_s_input = input("Surface stretching parameter (theta_s, default: 5): ").strip()
                    if not theta_s_input:
                        params['theta_s'] = 5
                        print("  ✓ Using default: 5")
                        break
                    theta_s = float(theta_s_input)
                    if theta_s < 0:
                        print("  ⚠ theta_s should be non-negative. Please try again.")
                        continue
                    params['theta_s'] = theta_s
                    print(f"  ✓ Using theta_s = {theta_s}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 5")
                    params['theta_s'] = 5
                    break
        else:
            print(f"  ✓ Using theta_s = {params['theta_s']} (from prompt)")
        
        # theta_b - bottom stretching parameter
        if 'theta_b' not in params or params['theta_b'] is None:
            while True:
                try:
                    theta_b_input = input("Bottom stretching parameter (theta_b, default: 0.5): ").strip()
                    if not theta_b_input:
                        params['theta_b'] = 0.5
                        print("  ✓ Using default: 0.5")
                        break
                    theta_b = float(theta_b_input)
                    if theta_b < 0:
                        print("  ⚠ theta_b should be non-negative. Please try again.")
                        continue
                    params['theta_b'] = theta_b
                    print(f"  ✓ Using theta_b = {theta_b}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 0.5")
                    params['theta_b'] = 0.5
                    break
        else:
            print(f"  ✓ Using theta_b = {params['theta_b']} (from prompt)")
        
        # hc - critical depth
        if 'hc' not in params or params['hc'] is None:
            while True:
                try:
                    hc_input = input("Critical depth (hc, negative value, default: -500): ").strip()
                    if not hc_input:
                        params['hc'] = -500
                        print("  ✓ Using default: -500 m")
                        break
                    hc = float(hc_input)
                    params['hc'] = hc
                    print(f"  ✓ Using hc = {hc} m")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: -500 m")
                    params['hc'] = -500
                    break
        else:
            print(f"  ✓ Using hc = {params['hc']} m (from prompt)")
        
        # Prompt for bathymetry smoothing parameters if not already specified
        print("\nBathymetry Smoothing Parameters:\n")
        
        # initial_smooth_sigma - Initial Gaussian smoothing strength
        if 'initial_smooth_sigma' not in params or params['initial_smooth_sigma'] is None:
            while True:
                try:
                    sigma_input = input("Initial Gaussian smoothing strength (default: 10): ").strip()
                    if not sigma_input:
                        params['initial_smooth_sigma'] = 10
                        print("  ✓ Using default: 10")
                        break
                    sigma = float(sigma_input)
                    if sigma < 0:
                        print("  ⚠ Smoothing strength should be non-negative. Please try again.")
                        continue
                    params['initial_smooth_sigma'] = sigma
                    print(f"  ✓ Using initial_smooth_sigma = {sigma}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 10")
                    params['initial_smooth_sigma'] = 10
                    break
        else:
            print(f"  ✓ Using initial_smooth_sigma = {params['initial_smooth_sigma']} (from prompt)")
        
        # hmin - Minimum depth threshold
        if 'hmin' not in params or params['hmin'] is None:
            while True:
                try:
                    hmin_input = input("Minimum depth threshold (hmin, negative value, default: -5): ").strip()
                    if not hmin_input:
                        params['hmin'] = -5
                        print("  ✓ Using default: -5 m")
                        break
                    hmin = float(hmin_input)
                    params['hmin'] = hmin
                    print(f"  ✓ Using hmin = {hmin} m")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: -5 m")
                    params['hmin'] = -5
                    break
        else:
            print(f"  ✓ Using hmin = {params['hmin']} m (from prompt)")
        
        # rx0_thresh - rx0 threshold for iterative smoothing
        if 'rx0_thresh' not in params or params['rx0_thresh'] is None:
            while True:
                try:
                    rx0_input = input("rx0 threshold for iterative smoothing (default: 0.2): ").strip()
                    if not rx0_input:
                        params['rx0_thresh'] = 0.2
                        print("  ✓ Using default: 0.2")
                        break
                    rx0 = float(rx0_input)
                    if rx0 <= 0:
                        print("  ⚠ rx0 threshold must be positive. Please try again.")
                        continue
                    params['rx0_thresh'] = rx0
                    print(f"  ✓ Using rx0_thresh = {rx0}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 0.2")
                    params['rx0_thresh'] = 0.2
                    break
        else:
            print(f"  ✓ Using rx0_thresh = {params['rx0_thresh']} (from prompt)")
        
        # max_iter - Maximum iterations for smoothing
        if 'max_iter' not in params or params['max_iter'] is None:
            while True:
                try:
                    iter_input = input("Maximum iterations for smoothing (default: 10): ").strip()
                    if not iter_input:
                        params['max_iter'] = 10
                        print("  ✓ Using default: 10")
                        break
                    max_iter = int(iter_input)
                    if max_iter < 0:
                        print("  ⚠ Maximum iterations must be non-negative. Please try again.")
                        continue
                    params['max_iter'] = max_iter
                    print(f"  ✓ Using max_iter = {max_iter}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid integer.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 10")
                    params['max_iter'] = 10
                    break
        else:
            print(f"  ✓ Using max_iter = {params['max_iter']} (from prompt)")
        
        # smooth_sigma - Smoothing strength for iterative method
        if 'smooth_sigma' not in params or params['smooth_sigma'] is None:
            while True:
                try:
                    smooth_input = input("Smoothing strength for iterative method (default: 6): ").strip()
                    if not smooth_input:
                        params['smooth_sigma'] = 6
                        print("  ✓ Using default: 6")
                        break
                    smooth_sig = float(smooth_input)
                    if smooth_sig < 0:
                        print("  ⚠ Smoothing strength should be non-negative. Please try again.")
                        continue
                    params['smooth_sigma'] = smooth_sig
                    print(f"  ✓ Using smooth_sigma = {smooth_sig}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 6")
                    params['smooth_sigma'] = 6
                    break
        else:
            print(f"  ✓ Using smooth_sigma = {params['smooth_sigma']} (from prompt)")
        
        # buffer_size - Buffer around steep regions
        if 'buffer_size' not in params or params['buffer_size'] is None:
            while True:
                try:
                    buffer_input = input("Buffer size around steep regions (default: 5): ").strip()
                    if not buffer_input:
                        params['buffer_size'] = 5
                        print("  ✓ Using default: 5")
                        break
                    buffer = int(buffer_input)
                    if buffer < 0:
                        print("  ⚠ Buffer size must be non-negative. Please try again.")
                        continue
                    params['buffer_size'] = buffer
                    print(f"  ✓ Using buffer_size = {buffer}")
                    break
                except ValueError:
                    print("  ⚠ Please enter a valid integer.")
                except KeyboardInterrupt:
                    print("\n  ⚠ Interrupted. Using default: 5")
                    params['buffer_size'] = 5
                    break
        else:
            print(f"  ✓ Using buffer_size = {params['buffer_size']} (from prompt)")
        
        # Set legacy defaults for backward compatibility
        params.setdefault('smoothing', True)
        params.setdefault('rx0_threshold', params.get('rx0_thresh', 0.2))
        
        print("\n" + "="*60)
        print("Grid Configuration Summary")
        print("="*60)
        print(f"Region:")
        print(f"  Latitude:  {params['lat_min']:.2f}° to {params['lat_max']:.2f}°")
        print(f"  Longitude: {params['lon_min']:.2f}° to {params['lon_max']:.2f}°")
        print(f"Grid Resolution:")
        print(f"  X (longitude): {params['dx_deg']:.4f}°")
        print(f"  Y (latitude):  {params['dy_deg']:.4f}°")
        print(f"Vertical Configuration:")
        print(f"  Levels:         {params['N_layers']}")
        print(f"  theta_s:        {params['theta_s']}")
        print(f"  theta_b:        {params['theta_b']}")
        print(f"  Critical depth: {params['hc']} m")
        print(f"Bathymetry Smoothing:")
        print(f"  Initial smooth sigma: {params['initial_smooth_sigma']}")
        print(f"  Minimum depth:        {params['hmin']} m")
        print(f"  rx0 threshold:        {params['rx0_thresh']}")
        print(f"  Max iterations:       {params['max_iter']}")
        print(f"  Smooth sigma:         {params['smooth_sigma']}")
        print(f"  Buffer size:          {params['buffer_size']}")
        print("="*60 + "\n")
        
        return params
    
    def parse_location_with_llm(self, prompt: str) -> Dict:
        """
        Use LLM to intelligently parse natural language prompts for region bounds
        and any explicitly specified grid parameters.
        
        The LLM attempts to extract:
        - Region bounds: lat_min, lat_max, lon_min, lon_max (required)
        - Grid parameters: dx_deg, dy_deg, N_layers, theta_s, theta_b, hc,
          initial_smooth_sigma, hmin, rx0_thresh, max_iter, smooth_sigma,
          buffer_size (all optional)
        
        Any parameters not extracted will be prompted for interactively later.
        
        Args:
            prompt: User's natural language request
            
        Returns:
            Dictionary with parsed parameters. At minimum should contain lat/lon bounds.
            Returns empty dict or error if bounds cannot be extracted.
        """
        if not self.llm:
            # Fallback to regex-based parsing
            print("LLM not available, using basic parsing...")
            return self.parse_location_basic(prompt)
        
        system_prompt = """You are a specialized assistant for ROMS ocean modeling grid generation.
Your task is to extract grid configuration parameters from user requests.

Extract the following information:
1. lat_min, lat_max: Latitude range (decimal degrees, North is positive)
2. lon_min, lon_max: Longitude range (decimal degrees, East is positive)
3. dx_deg, dy_deg: Horizontal grid resolution in degrees (if specified)
4. N_layers: Number of vertical layers (default 50)
5. theta_s: Surface stretching parameter (default 5)
6. theta_b: Bottom stretching parameter (default 0.5)
7. hc: Critical depth in meters, negative value (default -500)
8. initial_smooth_sigma: Initial Gaussian smoothing strength (default 10)
9. hmin: Minimum depth threshold in meters, negative value (default -5)
10. rx0_thresh: rx0 threshold for iterative smoothing (default 0.2)
11. max_iter: Maximum smoothing iterations (default 10)
12. smooth_sigma: Smoothing strength for iterative method (default 6)
13. buffer_size: Buffer around steep regions (default 5)

Common location references:
- US East Coast: approximately 24°N to 45°N, -81°W to -65°W
- Gulf of Mexico: approximately 18°N to 31°N, -98°W to -80°W
- California Coast: approximately 32°N to 42°N, -125°W to -117°W
- Chesapeake Bay: approximately 36.5°N to 39.5°N, -77.5°W to -75.5°W
- Gulf of Maine: approximately 41°N to 45°N, -71°W to -66°W
- Florida Keys: approximately 24.5°N to 25.5°N, -82°W to -80°W
- Cape Cod: approximately 41°N to 42.5°N, -71°W to -69.5°W

Return ONLY a valid JSON object with the extracted parameters. Use null for missing values.
Example: {"lat_min": 35.0, "lat_max": 42.0, "lon_min": -75.0, "lon_max": -65.0, "dx_deg": 0.01, "dy_deg": 0.01, "N_layers": 50, "theta_s": 5, "theta_b": 0.5, "hc": -500, "initial_smooth_sigma": 10, "hmin": -5, "rx0_thresh": 0.2, "max_iter": 10, "smooth_sigma": 6, "buffer_size": 5}"""
        
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
            
            # Extract JSON from response
            content = response.choices[0].message.content
            print(f"LLM response: {content}")
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                print(f"✓ LLM parsed parameters:\n{json.dumps(params, indent=2)}")
                return params
            else:
                print("⚠ Could not extract JSON from LLM response. Falling back to basic parsing.")
                return self.parse_location_basic(prompt)
                
        except Exception as e:
            print(f"⚠ Error using LLM: {e}. Falling back to basic parsing.")
            return self.parse_location_basic(prompt)
    
    def parse_location_basic(self, prompt: str) -> Dict:
        """
        Extract latitude and longitude ranges from natural language prompt using regex.
        Fallback method when LLM is not available.
        
        This basic parser only extracts lat/lon bounds. All other parameters will
        be prompted for interactively.
        
        Args:
            prompt: Natural language request containing lat/lon information
            
        Returns:
            Dictionary with lat_min, lat_max, lon_min, lon_max if found,
            otherwise empty dict
        """
        # Regex patterns for different coordinate formats
        patterns = {
            'lat_keywords': r'lat(?:itude)?[:\s]*(-?\d+\.?\d*)[°]?\s*(?:to|-)?\s*(-?\d+\.?\d*)[°]?',
            'lon_keywords': r'lon(?:gitude)?[:\s]*(-?\d+\.?\d*)[°]?\s*(?:to|-)?\s*(-?\d+\.?\d*)[°]?'
        }
        
        result = {}
        
        # Try to extract lat/lon with keywords first
        lat_match = re.search(patterns['lat_keywords'], prompt, re.IGNORECASE)
        lon_match = re.search(patterns['lon_keywords'], prompt, re.IGNORECASE)
        
        if lat_match and lon_match:
            result['lat_min'] = float(lat_match.group(1))
            result['lat_max'] = float(lat_match.group(2))
            result['lon_min'] = float(lon_match.group(1))
            result['lon_max'] = float(lon_match.group(2))
        else:
            # Fallback: look for number pairs
            numbers = re.findall(r'-?\d+\.?\d*', prompt)
            if len(numbers) >= 4:
                result = {
                    'lat_min': float(numbers[0]),
                    'lat_max': float(numbers[1]),
                    'lon_min': float(numbers[2]),
                    'lon_max': float(numbers[3])
                }
        
        # Set defaults - these will be prompted for if not in result
        # Just ensure we have the minimum required (lat/lon bounds)
        
        return result
    
    def download_bathymetry(self, lat_range: tuple, lon_range: tuple, 
                           output_file: str = "downloaded_bathy.nc") -> str:
        """
        Download and subset bathymetry data using model-tools.
        
        Args:
            lat_range: Tuple of (lat_min, lat_max)
            lon_range: Tuple of (lon_min, lon_max)
            output_file: Output filename
            
        Returns:
            Path to downloaded bathymetry file
        """
        print(f"\n📥 Downloading bathymetry for region:")
        print(f"   Latitude: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°")
        print(f"   Longitude: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°")
        
        # Download SRTM15+ bathymetry
        url = "https://topex.ucsd.edu/pub/global_topo_1min/topo_20.1.nc"
        full_file = os.path.join(self.output_dir, "topo_1min.nc")
        output_path = os.path.join(self.output_dir, output_file)
        
        # Download full file if not exists
        self.downloader.download_file(url, full_file)
        
        # Subset to region
        print(f"   Subsetting to region...")
        self.downloader.subset_dataset(full_file, output_path, lat_range, lon_range)
        
        print(f"✓ Bathymetry saved to: {output_path}")
        return output_path
    
    def generate_grid(self, bathy_file: str, params: Dict, 
                     output_file: str = "roms_grid.nc") -> str:
        """
        Generate ROMS grid using model-tools and bathymetry data.
        
        Creates a complete ROMS grid file with:
        - Horizontal grid at specified resolution (dx_deg, dy_deg)
        - Vertical levels with stretching (N_layers, theta_s, theta_b, hc)
        - Smoothed bathymetry using specified smoothing parameters
        - Land/sea masks
        - Staggered grids (u, v, psi points)
        - Grid metrics (pm, pn, dndx, dmde, etc.)
        - Visualization plots
        
        Args:
            bathy_file: Path to bathymetry NetCDF file
            params: Dictionary with all grid parameters including:
                   - Region bounds: lat_min, lat_max, lon_min, lon_max
                   - Resolution: dx_deg, dy_deg
                   - Vertical: N_layers, theta_s, theta_b, hc
                   - Smoothing: initial_smooth_sigma, hmin, rx0_thresh,
                     max_iter, smooth_sigma, buffer_size
            output_file: Output grid filename (default: "roms_grid.nc")
            
        Returns:
            Path to generated grid file
        """
        print(f"\n⚙️  Generating ROMS grid...")
        
        # Extract parameters
        lat_min = params['lat_min']
        lat_max = params['lat_max']
        lon_min = params['lon_min']
        lon_max = params['lon_max']
        resolution_km = params.get('resolution_km', 1.0)
        resolution_deg = params.get('resolution_deg')
        N_layers = params.get('N_layers', 50)
        hmin = params.get('hmin', 5)
        smoothing = params.get('smoothing', True)
        rx0_thresh = params.get('rx0_threshold', 0.2)
        
        # Load bathymetry data
        print(f"   Loading bathymetry from {bathy_file}...")
        bathy_ds = xr.open_dataset(bathy_file)
        
        # Determine resolution
        if resolution_deg is None:
            # Convert km to degrees (approximate at mid-latitude)
            mid_lat = (lat_min + lat_max) / 2
            resolution_deg = resolution_km / (111.32 * np.cos(np.deg2rad(mid_lat)))
        
        print(f"   Grid resolution: {resolution_km:.2f} km (~{resolution_deg:.4f}°)")
        
        # Create grid arrays
        nx = int((lon_max - lon_min) / resolution_deg) + 1
        ny = int((lat_max - lat_min) / resolution_deg) + 1
        print(f"   Grid dimensions: {ny} x {nx} points")
        
        lon_rho = np.linspace(lon_min, lon_max, nx)
        lat_rho = np.linspace(lat_min, lat_max, ny)
        lon_rho_grid, lat_rho_grid = np.meshgrid(lon_rho, lat_rho)
        
        # Create staggered grids using model-tools
        print(f"   Creating staggered grids (u, v, psi)...")
        staggered = grid_tools.create_staggered_grids(lon_rho_grid, lat_rho_grid)
        
        # Compute grid metrics using model-tools
        print(f"   Computing grid metrics (pm, pn, f)...")
        metrics = grid_tools.compute_grid_metrics(lon_rho_grid, lat_rho_grid)
        
        # Interpolate bathymetry to new grid
        print(f"   Interpolating bathymetry to grid...")
        if hasattr(bathy_ds, 'z'):
            bathy = bathy_ds.z.values
            bathy_lon = bathy_ds.lon.values
            bathy_lat = bathy_ds.lat.values
        elif hasattr(bathy_ds, 'elevation'):
            bathy = bathy_ds.elevation.values
            bathy_lon = bathy_ds.lon.values
            bathy_lat = bathy_ds.lat.values
        else:
            raise ValueError("Cannot find bathymetry variable (z or elevation) in dataset")
        
        # Interpolate and smooth bathymetry (following model-tools/scripts/grid_generation.py)
        # Use initial smoothing during interpolation for better results
        initial_smooth_sigma = 10 if smoothing else None
        h = grid_tools.interpolate_bathymetry(
            bathy, bathy_lon, bathy_lat, 
            lon_rho_grid, lat_rho_grid,
            smooth_sigma=initial_smooth_sigma,
            use_log_smoothing=True
        )
        
        # Create initial masks using model-tools
        print(f"   Creating land/sea masks...")
        masks = grid_tools.create_masks(h, -hmin)  # Note: hmin is positive, so pass negative
        
        # Fill h values where h is shallower than hmin with hmin (h is negative depths)
        h[h > -hmin] = -hmin
        h = np.nan_to_num(h, nan=-hmin)
        
        # Check initial rx0 before iterative smoothing
        if smoothing:
            rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h))
            rx0_x = np.pad(rx0_x, ((0, 0), (0, 1)), mode='edge')
            rx0_y = np.pad(rx0_y, ((0, 1), (0, 0)), mode='edge')
            initial_max_rx0 = max(np.nanmax(rx0_x), np.nanmax(rx0_y))
            print(f"   Initial max(rx0) = {initial_max_rx0:.4f}")
            
            # Apply iterative localized smoothing for steep regions
            print(f"   Applying iterative bathymetry smoothing (rx0 < {rx0_thresh})...")
            h = grid_tools.iterative_smoothing(
                h, rx0_thresh=rx0_thresh, max_iter=20, sigma=6, buffer_size=5
            )
        
        # Create vertical coordinate arrays using model-tools
        theta_s = params.get('theta_s', 5.0)
        theta_b = params.get('theta_b', 0.4)
        sigma_r = grid_tools.compute_sigma(N_layers, type='r')
        sigma_w = grid_tools.compute_sigma(N_layers, type='w')
        Cs_r = grid_tools.compute_cs(sigma_r, theta_s, theta_b)
        Cs_w = grid_tools.compute_cs(sigma_w, theta_s, theta_b)
        
        # Prepare sigma parameters dictionary for model-tools function
        sigma_params = {
            'N': N_layers,
            'sigma_r': sigma_r,
            'sigma_w': sigma_w,
            'Cs_r': Cs_r,
            'Cs_w': Cs_w
        }
        
        # Prepare global attributes
        global_attrs = {
            "title": "ROMS grid created by LLM-enabled Grid Agent",
            "size_x": nx,
            "size_y": ny,
            "center_lon": (lon_min + lon_max) / 2,
            "center_lat": (lat_min + lat_max) / 2,
            "resolution_km": resolution_km,
            "N_layers": N_layers,
            "hmin": hmin,
            "smoothing_applied": int(smoothing),
            "rx0_threshold": rx0_thresh,
            "theta_s": theta_s,
            "theta_b": theta_b,
        }
        
        # Create ROMS grid dataset using model-tools function
        print(f"   Assembling grid dataset...")
        grid_ds = grid_tools.create_roms_grid_dataset(
            lon_rho_grid, lat_rho_grid, -h,  # ROMS uses positive depths
            masks, staggered, metrics, sigma_params, global_attrs
        )
        
        # Save grid
        output_path = os.path.join(self.output_dir, output_file)
        print(f"   Saving grid to {output_path}...")
        grid_ds.to_netcdf(output_path)
        bathy_ds.close()
        
        print(f"✓ ROMS grid saved to: {output_path}")
        print(f"   Grid size: {ny} x {nx} = {ny*nx:,} points")
        print(f"   Domain size: {metrics['xl']/1000:.1f} km x {metrics['el']/1000:.1f} km")
        
        # Compute and display rx0 statistics
        rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h))
        print(f"   Bathymetry steepness (rx0): max = {max(np.max(rx0_x), np.max(rx0_y)):.4f}")
        
        # Create bathymetry plot
        print(f"   Creating bathymetry plot...")
        plot_file = output_path.replace('.nc', '_bathymetry.png')
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(
            central_longitude=(lon_min + lon_max) / 2,
            central_latitude=(lat_min + lat_max) / 2
        ))
        
        # Plot bathymetry (convert to positive depths for display)
        pc = ax.pcolormesh(lon_rho_grid, lat_rho_grid, -h, 
                          cmap=cmocean.cm.deep, shading='auto', 
                          transform=ccrs.PlateCarree())
        
        # Add coastlines and land features
        ax.coastlines(resolution='10m', color='k')
        ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='0.8')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add colorbar and labels
        plt.colorbar(pc, ax=ax, orientation='vertical', label='Depth (m)')
        plt.title('Bathymetry (h) on ROMS Grid')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Bathymetry plot saved to: {plot_file}")
        
        return output_path
    
    def execute_workflow(self, prompt: str) -> Dict:
        """
        Main workflow: parse prompt -> prompt for parameters -> download bathymetry -> generate grid.
        
        Workflow steps:
        1. Parse natural language prompt to extract region bounds (lat/lon)
        2. Interactively prompt user for comprehensive grid parameters:
           - Horizontal resolution (dx, dy in degrees)
           - Vertical configuration (N_layers, theta_s, theta_b, hc)
           - Bathymetry smoothing (initial_smooth_sigma, hmin, rx0_thresh,
             max_iter, smooth_sigma, buffer_size)
        3. Download bathymetry data for the region
        4. Generate ROMS grid with all specified parameters
        5. Create visualization plots
        
        Args:
            prompt: Natural language description of desired ROMS grid region.
                   Should include location/region name or lat/lon bounds.
                   Examples:
                   - "Create a grid for Chesapeake Bay"
                   - "Set up a model from lat 35-42, lon -75 to -65"
                   - "Generate a grid for the US East Coast"
            
        Returns:
            Dictionary with workflow results including:
            - success: Boolean indicating if workflow completed
            - parameters: Dict of all grid parameters used
            - bathymetry_file: Path to downloaded bathymetry file
            - grid_file: Path to generated ROMS grid file
            - message: Status message
            - error: Error message (if workflow failed)
        """
        print("=" * 60)
        print("ROMS Grid Generation Agent with LLM")
        print("=" * 60)
        print(f"\n📝 User request: {prompt}\n")
        
        # Step 1: Parse location using LLM
        params = self.parse_location_with_llm(prompt)
        if not params or 'lat_min' not in params:
            return {"error": "Could not extract lat/lon bounds from prompt"}
        
        # Step 2: Prompt user for grid parameters
        params = self._prompt_for_grid_parameters(params)
        
        # Step 3: Download bathymetry
        try:
            bathy_file = self.download_bathymetry(
                (params['lat_min'], params['lat_max']),
                (params['lon_min'], params['lon_max'])
            )
        except Exception as e:
            return {"error": f"Bathymetry download failed: {str(e)}"}
        
        # Step 4: Generate ROMS grid
        try:
            grid_file = self.generate_grid(bathy_file, params)
        except Exception as e:
            return {"error": f"Grid generation failed: {str(e)}"}
        
        print("\n" + "=" * 60)
        print("✅ ROMS Grid Generation Complete!")
        print("=" * 60)
        
        return {
            "success": True,
            "parameters": params,
            "bathymetry_file": bathy_file,
            "grid_file": grid_file,
            "message": "ROMS setup completed successfully"
        }


def main():
    """
    Main function for testing the agent.
    
    The agent will:
    1. Parse the natural language request to extract region bounds
    2. Prompt the user interactively for all grid parameters:
       - Horizontal resolution (dx, dy in degrees)
       - Number of vertical levels
       - Vertical stretching parameters (theta_s, theta_b, hc)
       - Bathymetry smoothing parameters (initial_smooth_sigma, hmin, 
         rx0_thresh, max_iter, smooth_sigma, buffer_size)
    3. Download bathymetry data
    4. Generate ROMS grid with specified parameters
    5. Create visualization plots
    
    Example usage:
        # Set API key in environment
        export LLM_API_KEY=<your-key>
        
        # Run the agent
        python llm_grid_agent.py
    """
    # Initialize agent - will prompt for output directory if not provided
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"  # Or omit to be prompted
    )
    
    # Test prompts demonstrating different capabilities
    # Note: The agent will prompt for detailed parameters regardless of what's in the request
    test_prompts = [
        "Create a grid for the US East Coast from latitude 35.0 to 42.0 and longitude -75.0 to -65.0",
        "I need a ROMS grid for Chesapeake Bay",
        "Set up a model for the Gulf of Maine region",
        "Generate a grid: lat 30-40, lon -80 to -70",
        "Create a grid for the California coast",
    ]
    
    # Run with first prompt - user will be prompted for all grid parameters
    print("\nNote: After region parsing, you will be prompted to enter:")
    print("  - Horizontal resolution (dx, dy in degrees)")
    print("  - Number of vertical levels")
    print("  - Vertical stretching parameters (theta_s, theta_b, hc)")
    print("  - Bathymetry smoothing parameters\n")
    
    result = agent.execute_workflow(test_prompts[0])
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
