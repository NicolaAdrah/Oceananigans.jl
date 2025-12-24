using BenchmarkTools
using Oceananigans
using Oceananigans.Units
using Printf
using Dates
using Random

function decide_arch()
    if !isnothing(Base.find_package("CUDA"))
        try
            @eval using CUDA
            if CUDA.functional()
                @info "CUDA found and functional. Using CUDA."
                return GPU()
            end
        catch e
            @warn "CUDA installed but not functional: $e"
        end
    end

    if !isnothing(Base.find_package("Metal"))
        try
            @eval using Metal
            @info "Metal found. Using Metal."
            return GPU()
        catch e
             @warn "Metal installed but failed to load: $e"
        end
    end
    
    @info "No GPU found. Using CPU."
    return CPU()
end

arch = decide_arch()
FT = Float32
Oceananigans.defaults.FloatType = FT

@info "Using architecture: $arch"

# --- Simulation Parameters ---
# Replicate the resolution from adriatic_simulation_skipped.jl approximately
# Original: x_faces ~ 1:7:end, y_faces ~ 1:7:end, z_faces ~ 1:4:end
# We will use a regular grid for synthetic fallback to keep it simple but representative in size.
Nx, Ny, Nz = 0, 0, 0

# Try to load real data if available, otherwise synthetic
using NCDatasets
mesh_mask_path = "data/mesh_mask2D_NA.nc" # Assuming relative path
if isfile(mesh_mask_path)
    @info "Found mesh mask at $mesh_mask_path. Using realistic grid size."
    ds = Dataset(mesh_mask_path)
    xc = ds["XC"][:]
    yc = ds["YC"][:]
    
    skip_x = 1
    skip_y = 1
    skip_z = 1
    
    # Calculate sizes as in the original script
    # This logic mimics: x_faces = xc[1:skip_x:end] ...
    # We just need the count
    Nx = length(1:skip_x:length(xc))
    Ny = length(1:skip_y:length(yc))
    
    # z_faces logic from script
    z_all = [-228.9898, -217.6341, -206.7042, -196.1894, -186.0793, -176.3635, 
        -167.032, -158.075, -149.4827, -141.2458, -133.3548, -125.8009, -118.575, 
        -111.6685, -105.0728, -98.77982, -92.78125, -87.06921, -81.63593, 
        -76.47385, -71.57558, -66.93387, -62.54169, -58.39215, -54.47855, 
        -50.79433, -47.33311, -44.08865, -41.05489, -38.2259, -35.59592, 
        -33.15932, -30.91062, -28.8445, -26.95574, -25.2393, -23.69024, 
        -22.30376, -21.0752, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 
        -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
    Nz = length(z_all[1:skip_z:end])
    
    grid = LatitudeLongitudeGrid(arch;
        size = (Nx, Ny, Nz),
        latitude  = (minimum(yc), maximum(yc)), 
        longitude = (minimum(xc), maximum(xc)),
        z = (minimum(z_all), 0),
        halo = (7, 7, 7)
    )
    
    # Try to load bathymetry
    try
        bottom_height = - ds["Depth"][:, :]
        # Subsample
        # Note: simplistic subsampling here, might not match exact indices of faces due to Center/Face alignment
        # but good enough for performance benchmark size.
        bh_sub = bottom_height[1:skip_x:end, 1:skip_y:end]
        # Ensure size match
        bh_sub = bh_sub[1:Nx, 1:Ny] 
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bh_sub))
    catch e
        @warn "Could not load/apply bathymetry from file, using flat bottom."
    end
    close(ds)
else
    @info "Mesh mask not found. Using synthetic grid."
    # Approximate sizes from the script comments/logic
    # "Loaded horizontal grid with ... x-cells and ... y-cells"
    # Typically regional Med models are ~ hundreds by hundreds
    Nx = 150 # Placeholder based on typical skip_x=7 for 1/12 deg
    Ny = 50 
    Nz = 30
    
    grid = LatitudeLongitudeGrid(arch;
        size = (Nx, Ny, Nz),
        latitude  = (30, 45),
        longitude = (0, 20),
        z = (-1000, 0),
        halo = (7, 7, 7)
    )
    # Simple bathymetry
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(-500))
end

@info "Grid size: ($Nx, $Ny, $Nz)"

# --- Physics ---
momentum_advection = VectorInvariant() # Default or what's compatible
tracer_advection = WENO(order=5)

# --- Boundary Conditions ---
# We use simple BCs for the benchmark to avoid IO and external package dependencies (like ClimaOcean) 
# if they are not strictly needed for the core solver performance.
# If actual performance depends on the complex BCs, we might underestimate cost, but this is a baseline.

model = HydrostaticFreeSurfaceModel(; grid,
    clock = Clock{FT}(time=0.0f0),
    momentum_advection = momentum_advection,
    tracer_advection = tracer_advection,
    free_surface = ImplicitFreeSurface(gravitational_acceleration=9.81f0),
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(gravitational_acceleration=9.81f0),
    # closers, buoyancy, etc. default
)

# --- Initialization ---
set!(model, T=15, S=35, u=0, v=0)

# --- Warmup ---
@info "Warming up..."
time_step!(model, 1.0f0)

# --- Benchmark ---
@info "Benchmarking..."
# We measure the time for a sequence of time steps
N_benchmark_steps = 10
trial = @benchmark begin
    # We wrap in CUDA.@sync if on GPU to measure full kernel time, 
    # though Oceananigans.time_step! usually handles synchronization for correctness,
    # for benchmarking we want to be sure.
    if $arch isa GPU
        if isdefined(Main, :CUDA)
            CUDA.@sync for _ in 1:$N_benchmark_steps
                time_step!($model, 1.0f0)
            end
        elseif isdefined(Main, :Metal)
            Metal.@sync for _ in 1:$N_benchmark_steps
                time_step!($model, 1.0f0)
            end
        else
            for _ in 1:$N_benchmark_steps
                time_step!($model, 1.0f0)
            end
        end
    else
        for _ in 1:$N_benchmark_steps
            time_step!($model, 1.0f0)
        end
    end
end samples=10

display(trial)

# --- Reporting ---
t_min_total = minimum(trial).time / 1e9 # seconds
t_min = t_min_total / N_benchmark_steps
@info "Minimum wall time per step: $(t_min) seconds"

# Calculate Simulated Years Per Day (SYPD) if Δt is realistic
# In script Δt = 60s
Δt_sim = 60.0
steps_per_day = 86400 / t_min
simulated_seconds_per_day = steps_per_day * Δt_sim
sypd = simulated_seconds_per_day / (365 * 86400)

@printf("Estimated SYPD (assuming Δt=60s): %.3f\n", sypd)
