using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvection
using Oceananigans.Advection: WENO
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval, Checkpointer
using Oceananigans.Grids: Face, Center, nodes, MutableVerticalDiscretization
using Oceananigans.Fields: Field
using Oceananigans: Simulation, run!, set!, FieldTimeSeries, PrescribedVelocityFields, architecture, launch!
using Oceananigans: fill_halo_regions!
using ClimaOcean
using Oceananigans.Units
using Printf
using NCDatasets
using Dates
using CopernicusMarine

const OTime = Oceananigans.Units.Time
const ENABLE_ADRIATIC_DEBUG = false

function debug_print(args...)
    ENABLE_ADRIATIC_DEBUG || return nothing
    println("[AdriaticDebug] ", join(string.(args), ""))
    return nothing
end
# For debugging
# using Logging
# ENV["JULIA_DEBUG"] = "ClimaOcean.DataWrangling"
# global_logger(SimpleLogger(stderr, Logging.Debug))


# --------- Modification on ClimaOcean functions
import ClimaOcean.DataWrangling: propagate_horizontally!, NearestNeighborInpainting, propagating
import ClimaOcean.DataWrangling: _nan_mask!, _propagate_field!, _substitute_values!, _fill_nans!
using KernelAbstractions: @kernel, @index

# ---------

import ClimaOcean.DataWrangling: restrict

ClimaOcean.DataWrangling.restrict(bbox_interfaces, interfaces, N) = begin
    Δ  = interfaces[2] - interfaces[1]
    rΔ = bbox_interfaces[2] - bbox_interfaces[1]
    ϵ  = rΔ / Δ
    rN = round(Int, ϵ * N)      # <- here it was Integer(...)
    return bbox_interfaces, rN
end

# ---------- End of modifications

arch = CPU()

ds = Dataset("data/mesh_mask2D_NA.nc")
# ds = Dataset("/leonardo_scratch/large/userexternal/naladrah/cluster_med/data/mesh_mask2D_NA.nc")
xc = ds["XC"][:]
yc = ds["YC"][:]

x_faces = [xc..., xc[end] + (xc[end] - xc[end-1])]
y_faces = [yc..., yc[end] + (yc[end] - yc[end-1])]
debug_print("Loaded horizontal grid with ", length(x_faces) - 1, " x-cells and ", length(y_faces) - 1, " y-cells")

z_faces = [-228.9898, -217.6341, -206.7042, -196.1894, -186.0793, -176.3635, 
    -167.032, -158.075, -149.4827, -141.2458, -133.3548, -125.8009, -118.575, 
    -111.6685, -105.0728, -98.77982, -92.78125, -87.06921, -81.63593, 
    -76.47385, -71.57558, -66.93387, -62.54169, -58.39215, -54.47855, 
    -50.79433, -47.33311, -44.08865, -41.05489, -38.2259, -35.59592, 
    -33.15932, -30.91062, -28.8445, -26.95574, -25.2393, -23.69024, 
    -22.30376, -21.0752, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 
    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

Nz = length(z_faces) - 1 # 140 vertical levels
Nx = length(x_faces) - 1
Ny = length(y_faces) - 1

debug_print("Configured vertical discretization with ", Nz, " levels (", length(z_faces), " faces)")

z_faces = MutableVerticalDiscretization(z_faces)

grid = LatitudeLongitudeGrid(arch;
    size      = (64, 48, 20),                       # <<< small!
    longitude = (minimum(xc), maximum(xc)),         # uniform spacing
    latitude  = (minimum(yc), maximum(yc)),         # uniform spacing
    z         = (-200.0, 0.0),                      # 20 levels to 200 m
    halo      = (6, 6, 3))                          # modest halos

debug_print("Constructed LatitudeLongitudeGrid with size (Nx=", Nx, ", Ny=", Ny, ", Nz=", Nz, "), halo=(7,7,7)")

# ### Load and set the Bathymetry 
# bottom_height = - ds["Depth"][:, :]
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
# debug_print("Applied bathymetry; active cells map ready (min depth ", minimum(bottom_height), ", max depth ", maximum(bottom_height), ")")

start_date = Date(2017, 1, 1)
# start_date = Date(2012, 1, 1)
end_date   = Date(2020, 12, 30)
# end_date   = Date(2016, 12, 30)

const POINTS_PER_DEG = 12  # GLORYS 1/12°

snap_floor(x) = floor(Int, x * POINTS_PER_DEG) / POINTS_PER_DEG
snap_ceil(x)  = ceil(Int,  x * POINTS_PER_DEG) / POINTS_PER_DEG

lon_min = snap_floor(x_faces[1] - 2)
lon_max = snap_ceil( x_faces[end] + 2)
lat_min = snap_floor(y_faces[1] - 2)
lat_max = snap_ceil( y_faces[end] + 2)

bbox_snapped = ClimaOcean.DataWrangling.BoundingBox(
    longitude = (lon_min, lon_max),
    latitude  = (lat_min, lat_max)
)
debug_print("Snapped bounding box lon=(", lon_min, ", ", lon_max, ") lat=(", lat_min, ", ", lat_max, ")")

dir = "./data"
# dir = "/leonardo_scratch/large/userexternal/naladrah/cluster_med/data"
dataset = GLORYSDaily()

u_meta = Metadata(:u_velocity;  dataset, dir,
                  bounding_box=bbox_snapped,
                  start_date=start_date, end_date=end_date)
v_meta = Metadata(:v_velocity;  dataset, dir,
                  bounding_box=bbox_snapped,
                  start_date=start_date, end_date=end_date)
T_meta = Metadata(:temperature; dataset, dir,
                  bounding_box=bbox_snapped,
                  start_date=start_date, end_date=end_date)
S_meta = Metadata(:salinity;    dataset, dir,
                  bounding_box=bbox_snapped,
                  start_date=start_date, end_date=end_date)

path = ClimaOcean.DataWrangling.metadata_path(u_meta[1])
if !isfile(path)
    error("Data needs to be downloaded on the login node! run `download_glorys_data.jl`.")
end
debug_print("Using metadata stored at ", path)

@info "bbox_snapped" lon_min lon_max lat_min lat_max

u_out = FieldTimeSeries(u_meta, grid; inpainting=NearestNeighborInpainting(130),time_indices_in_memory=30)
v_out = FieldTimeSeries(v_meta, grid; inpainting=NearestNeighborInpainting(130),time_indices_in_memory=30)

T_out = FieldTimeSeries(T_meta, grid; inpainting=NearestNeighborInpainting(130), time_indices_in_memory=30)
S_out = FieldTimeSeries(S_meta, grid; inpainting=NearestNeighborInpainting(130), time_indices_in_memory=30)
debug_print("Instantiated FieldTimeSeries for u, v, T, S (time_indices_in_memory = 10)")

const velocity_open_scheme = PerturbationAdvection(inflow_timescale=
1hours, outflow_timescale=6hours)

# ------------------------------------------------------------
# discrete-form boundary function that pins the missing j explicitly, 
# and pulls from the series with the proper 4-index signature (i, j, k, Time(clock.time)).
# --- Choose boundary j-indices explicitly
const j_south = 1
const j_north = Ny  # = size(grid, 2)
debug_print("Boundary indices pinned at j_south=", j_south, ", j_north=", j_north)

# --- Helper to wrap a FieldTimeSeries as a discrete boundary function
# side ∈ (:south, :north)
# CHECK THIS
@inline bc_from_series(series, side::Symbol) = begin
    j = side === :south ? 1 : size(series, 2)      # robust for Center (Ny) and v-Face (Ny+1)
    (i, k, grid, clock, fields...) -> series[i, j, k, OTime(clock.time)]
end

u_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(u_out, :south); discrete_form = true),
    # north = ValueBoundaryCondition(bc_from_series(u_out, :north); discrete_form = true)
)

v_bcs = (;
    south = OpenBoundaryCondition(bc_from_series(v_out, :south);
                                  discrete_form = true,
                                  scheme = velocity_open_scheme),
    # north = OpenBoundaryCondition(bc_from_series(v_out, :north);
                                #   discrete_form = true,
                                #   scheme = velocity_open_scheme)
)

T_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(T_out, :south); discrete_form = true),
    north = ValueBoundaryCondition(bc_from_series(T_out, :north); discrete_form = true)
)

S_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(S_out, :south); discrete_form = true),
    north = ValueBoundaryCondition(bc_from_series(S_out, :north); discrete_form = true)
)
# ------------------------------------------------------------

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order=5)

ocean = ocean_simulation(grid; 
                         momentum_advection, 
                         tracer_advection,
                         # TODO: Uncomment below...
                         # timestepper,
                         free_surface=ImplicitFreeSurface(),
                         lateral_boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs))

# Initializing the model
# ------------------------------------------------------------
const t_ic = 0.0
T_ic = T_out[OTime(t_ic)]
S_ic = S_out[OTime(t_ic)]
set!(ocean.model, T=T_ic, S=S_ic)
debug_print("Initial conditions applied from GLORYS snapshot at t=", prettytime(t_ic))
# ------------------------------------------------------------

atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true, dir=dir)

# This uses a quite simple ocean albedo model (latitude dependent) and 
# an ocean emissivity of 0.97. It is all customizable
radiation = Radiation()

# The coupled model! (we have no sea-ice so we do not add it)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# The coupled simulation:
Δt = 1minutes
stop_time = 10days
simulation = Simulation(coupled_model; Δt, stop_time)
debug_print("Simulation configured with Δt=", prettytime(Δt), ", stop_time=", prettytime(stop_time))

function progress(sim) 
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers

    @info @sprintf("Time: %s, Iteration %d, Δt %s\n
                    max(vel): (%.2e, %.2e, %.2e), max(T, S): %.2f, %.2f\n
                    min(vel): (%.2e, %.2e, %.2e), min(T, S): %.2f, %.2f\n
                    -----------------------------------------------------\n",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(abs, T), maximum(abs, S),
                   minimum(abs, u), minimum(abs, v), minimum(abs, w),
                   minimum(abs, T), minimum(abs, S))
end

using Oceananigans.Diagnostics: AdvectiveCFL
advective_cfl = AdvectiveCFL(Δt)

simulation.callbacks[:cfl_monitor] = Callback(IterationInterval(10)) do sim
    ocean_model = sim.model.ocean.model
    current_cfl = advective_cfl(ocean_model)
    @info "Advective CFL" iteration = sim.model.clock.iteration cfl = current_cfl
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

#Versione con uscite ogni 24 ore
simulation.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                       schedule = TimeInterval(1hours),
                                                       indices = (:, :, grid.Nz),
                                                       overwrite_existing = true,
                                                       filename = "med_surface_fields.jld2")

η = ocean.model.free_surface.η
simulation.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η,),
                                                      schedule = TimeInterval(1hours),
                                                      overwrite_existing = true,
                                                      filename = "freesurface_field.jld2")

ocean.output_writers[:checkpointer] = Checkpointer(ocean.model, 
						   schedule = IterationInterval(86400),
						   overwrite_existing = true,
						   prefix = "mediterranean")

## Running the Simulation
debug_print("Starting simulation run...")
run!(simulation)

# simulation.Δt = 1minutes
# simulation.stop_time = 365days

# run!(simulation)

# Record a video
#
# Let's read the data and record a video of the Mediterranean Sea's surface
# (1) Zonal velocity (u)
# (2) Meridional velocity (v)
# (3) Temperature (T)
# (4) Salinity (S)

u_series = FieldTimeSeries("med_surface_fields.jld2", "u"; backend=OnDisk())
v_series = FieldTimeSeries("med_surface_fields.jld2", "v"; backend=OnDisk())
T_series = FieldTimeSeries("med_surface_fields.jld2", "T"; backend=OnDisk())
S_series = FieldTimeSeries("med_surface_fields.jld2", "S"; backend=OnDisk())
η_series = FieldTimeSeries("freesurface_field.jld2", "η"; backend=OnDisk())
iter = Observable(1)

u = @lift(u_series[$iter])
v = @lift(v_series[$iter])
T = @lift(T_series[$iter])
S = @lift(S_series[$iter])
η = @lift(η_series[$iter])

using Oceananigans.Fields: interior

surface_slice(field; k = 1) = Array(interior(field)[:, :, k])

u_slice = @lift(surface_slice($u))
v_slice = @lift(surface_slice($v))
T_slice = @lift(surface_slice($T))
S_slice = @lift(surface_slice($S))
η_slice = @lift(surface_slice($η)) # k = 1 since free surface is 2‑D

fig = Figure()
ax  = Axis(fig[1, 1], title = "surface zonal velocity ms⁻¹")
heatmap!(ax, u_slice)
ax  = Axis(fig[1, 2], title = "surface meridional velocity ms⁻¹")
heatmap!(ax, v_slice)
ax  = Axis(fig[2, 1], title = "surface temperature ᵒC")
heatmap!(ax, T_slice)
ax  = Axis(fig[2, 2], title = "surface salinity psu")
heatmap!(ax, S_slice)
ax = Axis(fig[3, 1], title = "free surface η (m)")
heatmap!(ax, η_slice)

CairoMakie.record(fig, "mediterranean_video.mp4", 1:length(u_series.times); framerate = 5) do i
    @info "recording iteration $i"
    iter[] = i    
end
