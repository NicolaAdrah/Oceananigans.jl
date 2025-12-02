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

# -----------------------------------------------
# PARAMETERS OF THE SIMULATION

# With implicit nudging ON
# When in = 6hours, out  = 1hours,  NaN on b RHS starts at 5.2.. days
# When in = 3days,  out  = 1days,   NaN at 1.361 days
# When in = 6hours, out  = 12hours, NaN at 14.167 hours
# When no specifying,               NaN at 14.... hours

# With implicit nudging OFF
# in = 6hours, out = 1hours, NaN at 14.667 hours
# in = 3days, out = 1days, NaN at 14.167 hours

const velocity_open_scheme = PerturbationAdvection(inflow_timescale = 6hours,
                                                   outflow_timescale = 1hours)
Δt = 1minutes
stop_time = 15days

const OTime = Oceananigans.Units.Time
const ENABLE_ADRIATIC_DEBUG = false
# -----------------------------------------------


function debug_print(args...)
    ENABLE_ADRIATIC_DEBUG || return nothing
    println("[AdriaticDebug] ", join(string.(args), ""))
    return nothing
end
# # For debugging
# using Logging
# ENV["JULIA_DEBUG"] = "ClimaOcean.DataWrangling"
# global_logger(SimpleLogger(stderr, Logging.Debug))


# --------- Modification on ClimaOcean functions
import ClimaOcean.DataWrangling: propagate_horizontally!, NearestNeighborInpainting, propagating
import ClimaOcean.DataWrangling: _nan_mask!, _propagate_field!, _substitute_values!, _fill_nans!
using KernelAbstractions: @kernel, @index

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

skip_x = 7
skip_y = 7
skip_z = 4

x_faces = [xc..., xc[end] + (xc[end] - xc[end-1])]
x_faces = x_faces[1:skip_x:end]
y_faces = [yc..., yc[end] + (yc[end] - yc[end-1])]
y_faces = y_faces[1:skip_y:end]
debug_print("Loaded horizontal grid with ", length(x_faces) - 1, " x-cells and ", length(y_faces) - 1, " y-cells")

z_faces = [-228.9898, -217.6341, -206.7042, -196.1894, -186.0793, -176.3635, 
    -167.032, -158.075, -149.4827, -141.2458, -133.3548, -125.8009, -118.575, 
    -111.6685, -105.0728, -98.77982, -92.78125, -87.06921, -81.63593, 
    -76.47385, -71.57558, -66.93387, -62.54169, -58.39215, -54.47855, 
    -50.79433, -47.33311, -44.08865, -41.05489, -38.2259, -35.59592, 
    -33.15932, -30.91062, -28.8445, -26.95574, -25.2393, -23.69024, 
    -22.30376, -21.0752, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 
    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

z_faces = z_faces[1:skip_z:end]
push!(z_faces, 0.0)  # ensure surface face at 0 m
Nz = length(z_faces) - 1 # 140 vertical levels
Nx = length(x_faces) - 1
Ny = length(y_faces) - 1

debug_print("Configured vertical discretization with ", Nz, " levels (", length(z_faces), " faces)")

z_faces = MutableVerticalDiscretization(z_faces)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = y_faces,
                             longitude = x_faces,
                             z = z_faces,
                             halo = (7, 7, 7))
debug_print("Constructed LatitudeLongitudeGrid with size (Nx=", Nx, ", Ny=", Ny, ", Nz=", Nz, "), halo=(7,7,7)")

# ### Load and set the Bathymetry 
bottom_height = - ds["Depth"][:, :]
bottom_height = bottom_height[1:skip_x:end, 1:skip_y:end][1:Nx,1:Ny]
bottom_height[:,end] .= 0.0  # coastal boundary at north
bottom_height[1,:] .= 0.0  # coastal boundary at west
bottom_height[end,:] .= 0.0  # coastal boundary at east

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
debug_print("Applied bathymetry; active cells map ready (min depth ", minimum(bottom_height), ", max depth ", maximum(bottom_height), ")")

start_date = Date(2019, 1, 1)
# start_date = Date(2012, 1, 1)
# end_date   = Date(2020, 12, 30)
end_date   = Date(2019, 12, 30)

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

u_out = FieldTimeSeries(u_meta, grid; inpainting=NearestNeighborInpainting(200), time_indices_in_memory=10)
v_out = FieldTimeSeries(v_meta, grid; inpainting=NearestNeighborInpainting(200), time_indices_in_memory=10)

T_out = FieldTimeSeries(T_meta, grid; inpainting=NearestNeighborInpainting(200), time_indices_in_memory=10)
S_out = FieldTimeSeries(S_meta, grid; inpainting=NearestNeighborInpainting(200), time_indices_in_memory=10)

# ------
# Plotting inpainting effect example
# surface_slice(field;k) = Array(interior(field)[:,:,k])
# iter = Observable(1)

# u_raw  = @lift(T_out[$iter])
# u_slice_raw = @lift(surface_slice($u_raw; k = grid.Nz))

# fig = Figure(resolution = (800, 400))
# ax  = Axis(fig[1, 1], title = "T after inpainting 500 (GLORYS → grid)",
#                     xlabel = "i", ylabel = "j")
# hm  = heatmap!(ax, u_slice_raw; nan_color = :black)
# Colorbar(fig[1, 2], hm)

# CairoMakie.record(fig, "inpainting_exp/T_after_inpainting_500_raw.mp4", 1:length(T_out.times); framerate = 5) do i
#     @info "recording frame $i"
#     iter[] = i
# end
# ------

# ------
# 3D plot
# using GLMakie
# out = S_out
# title_out = "Salinity"
# nt = length(out.times)

# zero_point = 0.0
# Tmin = minimum(out)
# Tmax = maximum(out)

# @info "Global T range:" Tmin Tmax

# volume_iter  = Observable(1)
# volume_field = @lift(out[$volume_iter])
# volume_data  = @lift Array(interior($volume_field))

# fig = Figure(resolution = (900, 700))

# ax = Axis3(fig[1, 1];
#     title  = title_out,
#     xlabel = "i",
#     ylabel = "j",
#     zlabel = "k (depth)"
# )

# vol = volume!(ax, volume_data;
#     colormap   = :thermal,
#     nan_color  = :black,
#     colorrange = (Tmin, Tmax)
# )

# Colorbar(fig[1, 2], vol; label="Salinity (psu)")

# record(fig, "inpainting_exp/S_after_inpainting_volume.mp4", 1:nt; framerate = 5) do i
#     @info "Recording 3D frame $i / $nt"
#     volume_iter[] = i
# end

# ------


debug_print("Instantiated FieldTimeSeries for u, v, T, S (time_indices_in_memory = 10)")

# ------------------------------------------------------------
# discrete-form boundary function that pins the missing j explicitly, 
# and pulls from the series with the proper 4-index signature (i, j, k, Time(clock.time)).
# --- Choose boundary j-indices explicitly
const j_south = 1
const j_north = Ny
debug_print("Boundary indices pinned at j_south=", j_south, ", j_north=", j_north)

# --- Helper to wrap a FieldTimeSeries as a discrete boundary function
# side ∈ (:south, :north)
@inline bc_from_series(series, side::Symbol) = begin
    j = side === :south ? 1 : size(series, 2)      # robust for Center (Ny) and v-Face (Ny+1)
    (i, k, grid, clock, fields...) -> series[i, j, k, OTime(clock.time)]
end


u_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(u_out, :south); discrete_form = true)
)

v_bcs = (;
    south = OpenBoundaryCondition(bc_from_series(v_out, :south);
                                  discrete_form = true,
                                  scheme = velocity_open_scheme)
)

T_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(T_out, :south); discrete_form = true)
)

S_bcs = (;
    south = ValueBoundaryCondition(bc_from_series(S_out, :south); discrete_form = true)
)
# ------------------------------------------------------------

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order=5)

ocean = ocean_simulation(grid; 
                         momentum_advection, 
                         tracer_advection,
                         free_surface=ImplicitFreeSurface(),
                         lateral_boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs))

# Initializing the model
# --------------------- Initial Conditions -----------------------
const t_ic = 0.0
T_ic = T_out[OTime(t_ic)]
S_ic = S_out[OTime(t_ic)]

set!(ocean.model; T=T_ic, S=S_ic)

debug_print("Initial conditions applied from GLORYS snapshot at t=", prettytime(t_ic))
# ------------------------------------------------------------

# atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true, dir=dir)

jra = RepeatYearJRA55()

atmosphere = JRA55PrescribedAtmosphere(
    arch;
    dataset = jra,
    start_date = first_date(jra, :river_freshwater_flux),   # 1990-01-01T00
    end_date   = last_date(jra,  :river_freshwater_flux),   # 1990-12-31T00
    backend = JRA55NetCDFBackend(100),
    include_rivers_and_icebergs = true,
    dir = dir,
)

radiation = Radiation()

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)


simulation = Simulation(coupled_model; Δt, stop_time)
debug_print("Simulation configured with Δt=", prettytime(Δt), ", stop_time=", prettytime(stop_time))

using Oceananigans.BoundaryConditions: Open, BoundaryCondition

function update_orlanski_history!(sim)
    ocean_model = sim.model.ocean.model

    # We are using Orlanski / PerturbationAdvection on v_south
    v_field = ocean_model.velocities.v.data
    bc_south = ocean_model.velocities.v.boundary_conditions.south

    # Check that this is indeed an Open(PerturbationAdvection) BC
    if bc_south isa BoundaryCondition{<:Open{<:PerturbationAdvection}}
        pa = bc_south.classification.scheme

        if pa.previous === nothing
            pa.previous = similar(v_field)
        end

        # true "know" := last time step; here we store current state
        copyto!(pa.previous, v_field)
    end

    return nothing
end

simulation.callbacks[:orlanski_history] = Callback(update_orlanski_history!, IterationInterval(1))

function progress(sim) 
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers

    @info @sprintf("Time: %s, Iteration %d, Δt %s, max(vel): (%.2e, %.2e, %.2e), max(T, S): %.2f, %.2f\n
                                                   min(vel): (%.2e, %.2e, %.2e), min(T, S): %.2f, %.2f\n",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(abs, T), maximum(abs, S),
                   minimum(abs, u), minimum(abs, v), minimum(abs, w),
                   minimum(T), minimum(S))
    if any(!isfinite, parent(ocean.model.velocities.u.data))
        inds = findall(!isfinite, parent(ocean.model.velocities.u.data))
        @warn "Non-finite u" first_index = first(inds)
    end
    if any(!isfinite, parent(ocean.model.velocities.v.data))
        inds = findall(!isfinite, parent(ocean.model.velocities.v.data))
        @warn "Non-finite v" first_index = first(inds)
    end
    if any(!isfinite, parent(ocean.model.tracers.T.data))
        inds = findall(!isfinite, parent(ocean.model.tracers.T.data))
        @warn "Non-finite T" first_index = first(inds)
    end
    if any(!isfinite, parent(ocean.model.tracers.S.data))
        inds = findall(!isfinite, parent(ocean.model.tracers.S.data))
        @warn "Non-finite S" first_index = first(inds)
    end
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

#Versione con uscite ogni 24 ore
simulation.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                       schedule = TimeInterval(1hours),
                                                       indices = (:, :, grid.Nz),
                                                       overwrite_existing = true,
                                                       filename = "med_surface_fields.jld2")

using Oceananigans.Diagnostics: AdvectiveCFL
advective_cfl = AdvectiveCFL(Δt)

simulation.callbacks[:cfl_monitor] = Callback(IterationInterval(10)) do sim
    ocean_model = sim.model.ocean.model
    current_cfl = advective_cfl(ocean_model)
    @info "Advective CFL" iteration = sim.model.clock.iteration cfl = current_cfl
end

η = ocean.model.free_surface.η
simulation.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η,),
                                                      schedule = TimeInterval(24hours),
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

# # Record a video
# #
# # Let's read the data and record a video of the Mediterranean Sea's surface
# # (1) Zonal velocity (u)
# # (2) Meridional velocity (v)
# # (3) Temperature (T)
# # (4) Salinity (S)

u_series = FieldTimeSeries("med_surface_fields.jld2", "u"; backend=OnDisk())
v_series = FieldTimeSeries("med_surface_fields.jld2", "v"; backend=OnDisk())
T_series = FieldTimeSeries("med_surface_fields.jld2", "T"; backend=OnDisk())
S_series = FieldTimeSeries("med_surface_fields.jld2", "S"; backend=OnDisk())
η_series = FieldTimeSeries("freesurface_field.jld2", "η"; backend=OnDisk())
iter = Observable(1)

umin = minimum(u_series)
umax = maximum(u_series)

vmin = minimum(v_series)
vmax = maximum(v_series)

Tmin = minimum(T_series)
Tmax = maximum(T_series)

Smin = minimum(S_series)
Smax = maximum(S_series)

ηmin = minimum(η_series)
ηmax = maximum(η_series)

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
# --- u ---
ax_u = Axis(fig[1, 1], title = "surface zonal velocity ms⁻¹")
hm_u = heatmap!(ax_u, u_slice; colorrange = (umin, umax))
Colorbar(fig[1, 2], hm_u)

# --- v ---
ax_v = Axis(fig[1, 3], title = "surface meridional velocity ms⁻¹")
hm_v = heatmap!(ax_v, v_slice; colorrange = (vmin, vmax))
Colorbar(fig[1, 4], hm_v)

# --- T ---
ax_T = Axis(fig[2, 1], title = "surface temperature ᵒC")
hm_T = heatmap!(ax_T, T_slice; colorrange = (Tmin, Tmax))
Colorbar(fig[2, 2], hm_T)

# --- S ---
ax_S = Axis(fig[2, 3], title = "surface salinity psu")
hm_S = heatmap!(ax_S, S_slice; colorrange = (Smin, Smax))
Colorbar(fig[2, 4], hm_S)

# --- η ---
ax_eta = Axis(fig[3, 1], title = "free surface η (m)")
hm_eta = heatmap!(ax_eta, η_slice; colorrange = (ηmin, ηmax))
Colorbar(fig[3, 2], hm_eta)

CairoMakie.record(fig, "mediterranean_video_im_nudging_NaNs.mp4", 1:length(u_series.times); framerate = 5) do i
    @info "recording iteration $i"
    iter[] = i    
end
