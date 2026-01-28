#!/usr/bin/env python3
"""
RODMS (Regional Ocean Data Modeling System) Agent with LLM Integration

This agent uses large language models to parse natural language requests
and automate ROMS grid generation workflows using the model-tools library.
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

# Try to import anthropic, but make it optional
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    print("Warning: anthropic library not available. Install with: pip install anthropic")

# Add model-tools to path
sys.path.insert(0, '/global/cfs/cdirs/m4304/enuss/model-tools/code')
from download import Downloader
from grid import grid_tools


class ROMSGridAgent:
    """
    ROMS Grid Agent with LLM integration for natural language grid generation.
    """
    
    def __init__(self, model_tools_path: str, api_key: Optional[str] = None, 
                 output_dir: Optional[str] = None):
        """
        Initialize agent with path to model-tools repository and optional API key.
        
        Args:
            model_tools_path: Path to model-tools directory
            api_key: Anthropic API key (if not provided, will try to read from environment)
            output_dir: Directory for output files (if not provided, will prompt user)
        """
        self.model_tools_path = model_tools_path
        
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
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.llm = Anthropic(api_key=self.api_key)
                print("✓ LLM initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                self.llm = None
        else:
            if not ANTHROPIC_AVAILABLE:
                print("Warning: Anthropic library not installed. LLM features disabled.")
            else:
                print("Warning: No API key provided. LLM features will be disabled.")
                print("Set ANTHROPIC_API_KEY environment variable to enable LLM features.")
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
        
    def parse_location_with_llm(self, prompt: str) -> Dict:
        """
        Use LLM to intelligently parse natural language prompts for grid parameters.
        
        Args:
            prompt: User's natural language request
            
        Returns:
            Dictionary with parsed parameters including bounds, resolution, etc.
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
3. resolution_km: Desired grid resolution in kilometers (if specified, default 1.0)
4. resolution_deg: Desired grid resolution in degrees (if specified)
5. N_layers: Number of vertical layers (default 50)
6. hmin: Minimum depth in meters (default 5)
7. smoothing: Whether to apply bathymetry smoothing (default true)
8. rx0_threshold: Steepness parameter threshold (default 0.2)

Common location references:
- US East Coast: approximately 24°N to 45°N, -81°W to -65°W
- Gulf of Mexico: approximately 18°N to 31°N, -98°W to -80°W
- California Coast: approximately 32°N to 42°N, -125°W to -117°W
- Chesapeake Bay: approximately 36.5°N to 39.5°N, -77.5°W to -75.5°W
- Gulf of Maine: approximately 41°N to 45°N, -71°W to -66°W
- Florida Keys: approximately 24.5°N to 25.5°N, -82°W to -80°W
- Cape Cod: approximately 41°N to 42.5°N, -71°W to -69.5°W

Return ONLY a valid JSON object with the extracted parameters. Use null for missing values.
Example: {"lat_min": 35.0, "lat_max": 42.0, "lon_min": -75.0, "lon_max": -65.0, "resolution_km": 1.0, "N_layers": 50, "hmin": 5, "smoothing": true, "rx0_threshold": 0.2}"""
        
        try:
            print("🤖 Querying LLM to parse request...")
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
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
        
        # Set defaults
        result.setdefault('resolution_km', 1.0)
        result.setdefault('N_layers', 50)
        result.setdefault('hmin', 5)
        result.setdefault('smoothing', True)
        result.setdefault('rx0_threshold', 0.2)
        
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
        
        Args:
            bathy_file: Path to bathymetry NetCDF file
            params: Dictionary with grid parameters
            output_file: Output grid filename
            
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
        Main workflow: parse prompt -> download bathymetry -> generate grid.
        Uses LLM to intelligently parse the user's request.
        
        Args:
            prompt: Natural language description of desired ROMS grid
            
        Returns:
            Dictionary with workflow results and output file paths
        """
        print("=" * 60)
        print("ROMS Grid Generation Agent with LLM")
        print("=" * 60)
        print(f"\n📝 User request: {prompt}\n")
        
        # Step 1: Parse location using LLM
        params = self.parse_location_with_llm(prompt)
        if not params or 'lat_min' not in params:
            return {"error": "Could not extract lat/lon bounds from prompt"}
        
        # Step 2: Download bathymetry
        try:
            bathy_file = self.download_bathymetry(
                (params['lat_min'], params['lat_max']),
                (params['lon_min'], params['lon_max'])
            )
        except Exception as e:
            return {"error": f"Bathymetry download failed: {str(e)}"}
        
        # Step 3: Generate ROMS grid
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
    """Main function for testing the agent"""
    # Initialize agent - will prompt for output directory if not provided
    agent = ROMSGridAgent(
        model_tools_path="/global/cfs/cdirs/m4304/enuss/model-tools",
        output_dir="/global/cfs/cdirs/m4304/enuss/model-tools/output"  # Or omit to be prompted
    )
    
    # Test prompts demonstrating different capabilities
    test_prompts = [
        "Create a 1 km resolution grid for the US East Coast from latitude 35.0 to 42.0 and longitude -75.0 to -65.0",
        "I need a ROMS grid for Chesapeake Bay with 50 vertical layers",
        "Set up a model for the Gulf of Maine region with 2km resolution",
        "Generate a grid: lat 30-40, lon -80 to -70, 1km resolution, 40 layers",
    ]
    
    # Run with first prompt
    result = agent.execute_workflow(test_prompts[0])
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
