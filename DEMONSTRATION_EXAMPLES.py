"""
DEMONSTRATION: Intelligent Grid Parameter Suggestion

This script shows example interactions with the enhanced grid agent.
Run the examples below to see how the agent suggests parameters.
"""

def example_1_submesoscale_gulf_stream():
    """
    Example 1: Submesoscale-Resolving Gulf Stream
    
    User Request: "I want to set up a submesoscale resolving grid of the Gulf Stream"
    
    AGENT WORKFLOW:
    ================
    
    Step 1: Parse with LLM
    ----------------------
    🤖 Querying LLM to parse request...
    
    Extracted:
    {
      "lat_min": 25.0,
      "lat_max": 45.0,
      "lon_min": -80.0,
      "lon_max": -50.0,
      "simulation_goals": "submesoscale resolving simulation of the Gulf Stream",
      "dx_deg": null,
      "dy_deg": null,
      ...
    }
    
    ✓ Simulation goals identified: submesoscale resolving simulation of the Gulf Stream
    
    
    Step 2: Generate Suggestions
    -----------------------------
    🤖 Analyzing region and simulation goals to suggest parameters...
    
    
    Step 3: Offer Suggestions
    --------------------------
    ============================================================
    Suggested Grid Configuration
    ============================================================
    
    📊 Recommendation: Submesoscale-resolving configuration for Gulf Stream region.
    
    The Gulf Stream is characterized by strong mesoscale and submesoscale 
    variability (O(1-10 km) eddies and fronts). To resolve these features:
    - ~1 km horizontal resolution captures submesoscale eddies and fronts
    - 50 vertical levels with strong surface stretching (theta_s=7) provides
      fine resolution in the mixed layer and surface boundary layer
    - Moderate bottom stretching (theta_b=2) resolves slope currents
    - Critical depth hc=-250m appropriate for shelf-slope transition region
    
    Suggested Parameters:
      Horizontal Resolution:
        • dx (longitude): 0.0100° (~1.1 km)
        • dy (latitude):  0.0100° (~1.1 km)
      Vertical Configuration:
        • N_layers:  50
        • theta_s:   7.0 (surface stretching)
        • theta_b:   2.0 (bottom stretching)
        • hc:        -250.0 m (critical depth)
      Bathymetry Smoothing:
        • Initial smooth:  10.0
        • Min depth:       -5.0 m
        • rx0 threshold:   0.2
        • Max iterations:  15
        • Smooth sigma:    6.0
        • Buffer size:     5
    
    ============================================================
    
    Use these suggested parameters? [Y/n] (Y to accept, n to customize): Y
    ✓ Using suggested parameters
    
    📥 Downloading bathymetry for region:
       Latitude: 25.00° to 45.00°
       Longitude: -80.00° to -50.00°
    ...
    """


def example_2_coastal_puget_sound():
    """
    Example 2: Coastal Model - Location Only
    
    User Request: "Create a grid for Puget Sound"
    
    AGENT WORKFLOW:
    ================
    
    Step 1: Parse with LLM
    ----------------------
    🤖 Querying LLM to parse request...
    
    Extracted:
    {
      "lat_min": 47.0,
      "lat_max": 49.0,
      "lon_min": -123.5,
      "lon_max": -122.0,
      "simulation_goals": null,
      ...
    }
    
    
    Step 2: Prompt for Goals
    -------------------------
    ============================================================
    Simulation Objectives
    ============================================================
    
    To suggest appropriate grid parameters, please describe:
      - What phenomena do you want to resolve?
        (e.g., submesoscale eddies, coastal upwelling, tides, etc.)
      - What type of simulation are you running?
        (e.g., process study, operational forecast, climate, etc.)
      - Any specific scale requirements?
        (e.g., O(1km) resolution, fine vertical structure, etc.)
    
    Examples:
      - 'Submesoscale resolving simulation for process studies'
      - 'Mesoscale circulation with focus on shelf dynamics'
      - 'Coastal model for tidal and estuarine processes'
      - 'Large-scale ocean circulation for climate studies'
    
    Describe your simulation objectives: tidal and estuarine circulation with focus on mixing processes
    ✓ Simulation goals: tidal and estuarine circulation with focus on mixing processes
    
    
    Step 3: Generate Suggestions
    -----------------------------
    🤖 Analyzing region and simulation goals to suggest parameters...
    
    
    Step 4: Offer Suggestions
    --------------------------
    ============================================================
    Suggested Grid Configuration
    ============================================================
    
    📊 Recommendation: Coastal estuarine configuration for Puget Sound.
    
    Puget Sound is a complex estuarine system with strong tidal forcing and 
    mixing processes. For tidal and estuarine circulation:
    - ~2 km horizontal resolution resolves major channels and basins
    - 40 vertical levels with balanced stretching captures stratification
    - Shallow critical depth (hc=-50m) appropriate for shelf/estuarine depths
    - Strong bottom stretching (theta_b=2.5) resolves bottom boundary layer
    
    Suggested Parameters:
      Horizontal Resolution:
        • dx (longitude): 0.0200° (~2.2 km)
        • dy (latitude):  0.0200° (~2.2 km)
      Vertical Configuration:
        • N_layers:  40
        • theta_s:   6.0 (surface stretching)
        • theta_b:   2.5 (bottom stretching)
        • hc:        -50.0 m (critical depth)
      Bathymetry Smoothing:
        • Initial smooth:  10.0
        • Min depth:       -5.0 m
        • rx0 threshold:   0.2
        • Max iterations:  12
        • Smooth sigma:    6.0
        • Buffer size:     5
    
    ============================================================
    
    Use these suggested parameters? [Y/n] (Y to accept, n to customize): Y
    ✓ Using suggested parameters
    
    ...
    """


def example_3_user_customization():
    """
    Example 3: User Wants to Customize
    
    User Request: "Create a mesoscale model for the California coast"
    
    AGENT WORKFLOW:
    ================
    
    Step 1-3: Parse and Generate Suggestions (same as above)
    
    Step 4: User Chooses to Customize
    -----------------------------------
    ============================================================
    Suggested Grid Configuration
    ============================================================
    
    📊 Recommendation: Mesoscale coastal configuration for California coast.
    
    California coast features strong upwelling dynamics and mesoscale eddies.
    Standard mesoscale configuration with ~5km resolution appropriate for
    coastal upwelling and shelf circulation...
    
    [Suggested parameters displayed...]
    
    Use these suggested parameters? [Y/n] (Y to accept, n to customize): n
    ✓ Will prompt for custom parameters
    
    
    Step 5: Interactive Parameter Prompting
    ----------------------------------------
    ============================================================
    Grid Configuration
    ============================================================
    
    Region: Lat [32.00° to 42.00°], Lon [-125.00° to -117.00°]
    
    Please specify grid parameters:
    
    Number of vertical levels (default: 50): 60
      ✓ Using 60 vertical levels
    Horizontal resolution in X (longitude) in degrees (e.g., 0.01 = ~1 km): 0.02
      ✓ Using 0.02° resolution in X direction
    Horizontal resolution in Y (latitude) in degrees (e.g., 0.01 = ~1 km): 0.02
      ✓ Using 0.02° resolution in Y direction
    
    Vertical Stretching Parameters:
    
    Surface stretching parameter (theta_s, default: 5): 7
      ✓ Using theta_s = 7.0
    Bottom stretching parameter (theta_b, default: 0.5): 2.0
      ✓ Using theta_b = 2.0
    Critical depth (hc, negative value, default: -500): -150
      ✓ Using hc = -150.0 m
    
    [... continues with bathymetry smoothing parameters ...]
    """


def example_4_explicit_parameters():
    """
    Example 4: User Specifies Some Parameters Explicitly
    
    User Request: "Create a grid for lat 35-42, lon -75 to -65 with 0.01 degree resolution and 60 levels"
    
    AGENT WORKFLOW:
    ================
    
    Step 1: Parse with LLM
    ----------------------
    🤖 Querying LLM to parse request...
    
    Extracted:
    {
      "lat_min": 35.0,
      "lat_max": 42.0,
      "lon_min": -75.0,
      "lon_max": -65.0,
      "simulation_goals": null,
      "dx_deg": 0.01,
      "dy_deg": 0.01,
      "N_layers": 60,
      "theta_s": null,
      ...
    }
    
    
    Step 2: Skip Suggestions (Explicit Params Found)
    -------------------------------------------------
    ✓ Found 3 explicitly specified parameters
    
    
    Step 3: Prompt Only for Missing Parameters
    -------------------------------------------
    ============================================================
    Grid Configuration
    ============================================================
    
    Region: Lat [35.00° to 42.00°], Lon [-75.00° to -65.00°]
    
    Please specify grid parameters:
    
      ✓ Using 60 vertical levels (from prompt)
      ✓ Using 0.01° resolution in X direction (from prompt)
      ✓ Using 0.01° resolution in Y direction (from prompt)
    
    Vertical Stretching Parameters:
    
    Surface stretching parameter (theta_s, default: 5): 
      ✓ Using default: 5.0
    Bottom stretching parameter (theta_b, default: 0.5): 
      ✓ Using default: 0.5
    Critical depth (hc, negative value, default: -500): 
      ✓ Using default: -500 m
    
    [... continues with bathymetry smoothing ...]
    
    NOTE: When explicit parameters are provided, suggestion system is bypassed
          and agent only prompts for unspecified parameters.
    """


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║      ROMS Grid Agent - Intelligent Parameter Suggestion Examples          ║
║                                                                            ║
║  These examples show the complete interaction flow with the enhanced      ║
║  grid agent, demonstrating how it suggests scientifically appropriate     ║
║  parameters based on simulation goals and regional characteristics.       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

The examples below show different workflows:

1. Submesoscale Gulf Stream - Goals specified in prompt
2. Puget Sound - Location only, agent prompts for goals  
3. California Coast - User chooses to customize suggestions
4. Explicit Parameters - User specifies some params, agent fills in rest

""")
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Submesoscale-Resolving Gulf Stream")
    print("="*80)
    print(example_1_submesoscale_gulf_stream.__doc__)
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Coastal Model - Location Only")
    print("="*80)
    print(example_2_coastal_puget_sound.__doc__)
    
    print("\n" + "="*80)
    print("EXAMPLE 3: User Customization")
    print("="*80)
    print(example_3_user_customization.__doc__)
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Explicit Parameters")
    print("="*80)
    print(example_4_explicit_parameters.__doc__)
    
    print("\n" + "="*80)
    print("\nTo actually run these workflows:")
    print("  from llm_grid_agent import ROMSGridAgent")
    print("  agent = ROMSGridAgent(model_tools_path='/path/to/model-tools')")
    print("  result = agent.execute_workflow('your request here')")
    print("="*80)
