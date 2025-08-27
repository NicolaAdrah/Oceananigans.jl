using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Grids: architecture
using Oceananigans.Models

using KernelAbstractions: @kernel, @index

# Fade coefficient from Fortran code:
const δ = 0.006

wt = time_ns()

z = MutableVerticalDiscretization((-10, 0))

grid = RectilinearGrid(size = (50, 50, 1),     
                          x = (0, 500kilometers),
                          y = (0, 500kilometers),
                          topology = (Bounded, Bounded, Bounded),
                          z = z)

spacing_x = spacing_y = 10kilometers

free_surface = ImplicitFreeSurface() 
                      
##### 
##### Build boundary conditions
#####

using Oceananigans.BoundaryConditions: Open
import Oceananigans.BoundaryConditions: getbc, update_boundary_condition!

struct OrlanskiBoundary{U, U1}
    uᴮ :: U
    u1 :: U1
end

OrlanskiBoundaryCondition = BoundaryCondition{<:Open, <:OrlanskiBoundary}

# Current-field allocation
uᵂ  = Field{Face, Center, Center}(grid)
uᴱ  = Field{Face, Center, Center}(grid)
vᴺ  = Field{Center, Face, Center}(grid)
vˢ  = Field{Center, Face, Center}(grid)

# History-field allocation
u₁ᵂ = Field{Face, Center, Center}(grid)
u₁ᴱ = Field{Face, Center, Center}(grid)
v₁ᴺ = Field{Center, Face, Center}(grid)
v₁ˢ = Field{Center, Face, Center}(grid)

# Initialize the boundary fields with zeros or appropriate initial conditions
fill!(uᵂ, 0)
fill!(uᴱ, 0)
fill!(vᴺ, 0)
fill!(v₁ˢ, 0)
fill!(u₁ᵂ, 0)
fill!(u₁ᴱ, 0)
fill!(v₁ᴺ, 0)
fill!(v₁ˢ, 0)

u_west  = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ))
u_east  = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ))
v_north = OpenBoundaryCondition(OrlanskiBoundary(vᴺ, v₁ᴺ))
v_south = OpenBoundaryCondition(OrlanskiBoundary(vˢ, v₁ˢ))

# West/East (x‑faces): Oceananigans will call getbc(bc, j, k, …)
@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) =
  bc.condition.uᴮ[1, j, k]

# South/North (y‑faces): Oceananigans will call getbc(bc, i, k, …)
@inline getbc(bc::OrlanskiBoundaryCondition, i, k, args...) =
  bc.condition.uᴮ[i, 1, k]

u_bcs = FieldBoundaryConditions(west=u_west, east=u_east)
v_bcs = FieldBoundaryConditions(north=v_north, south=v_south)

# Helper: fade step
@inline function orlanski_step(current, estimate)
  current + δ * (estimate - current)
end

@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  j, k = @index(Global, NTuple)
  Nx   = size(grid, 1)
  Ny   = size(grid, 2)

	@inbounds begin

		# 1) compute differences (interior gradients)
		Δuₓ = uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k]
		Δuₜ = uⁿ⁺¹[2, j, k] - u₁[2, j, k]
		
		# CHECK THIS IMP HERE
    Δuᵧ = j < Ny ? uⁿ⁺¹[2, j+1, k] - uⁿ⁺¹[2, j, k] : uⁿ⁺¹[2, j, k] - uⁿ⁺¹[2, j-1, k]

		max_speed = - sqrt(9.80665 * grid.Lz)

    # 2) raw phase speed estimate - handle potential division by zero
    speed_x = abs(Δuₓ * Δt) > 1e-20 ? - (Δuₜ / Δuₓ) * (1 / (1 - (Δuᵧ/Δuₓ * spacing_x / spacing_y)^2)) *
            (spacing_x / Δt) : 0.0
    
    speed_y = abs(Δuᵧ * Δt) > 1e-20 ? - (Δuₜ / Δuᵧ) * (1 / (1 - (Δuₓ/Δuᵧ * spacing_y / spacing_x)^2)) *
            (spacing_y / Δt) : 0.0

    # 5) clamp speeds to [0, max_speed]
    speed_x = clamp(isnan(speed_x) ? 0.0 : speed_x, 0, max_speed)
    speed_y = clamp(isnan(speed_y) ? 0.0 : speed_y, 0, max_speed)

		# 4) nondimensional Courant number
		cₓ = max_speed * Δt / spacing_x
		cᵧ = max_speed * Δt / spacing_y

		# 5) Orlanski update
    neighbor_j = j + (speed_y > 0 ? 1 : -1)
		uᴮ[1, j, k] = (uᴮ[1, j, k] - cₓ * uⁿ⁺¹[2, j, k] - cᵧ * uⁿ⁺¹[2, neighbor_j, k]) / (1 - cₓ - (speed_y > 0 ? cᵧ : -cᵧ))
    # uᴮ[1, j, k] = orlanski_step(uᴮ[1, j, k], estimate)
	end
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  j, k = @index(Global, NTuple)
  Nx   = size(grid, 1)
  Ny   = size(grid, 2)

  @inbounds begin

		# 1) compute differences
    # TODO should I shift the indices here by extra -1 ?
		Δuₓ = uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx-1, j, k]
		Δuₜ = uⁿ⁺¹[Nx, j, k] - u₁[Nx, j, k]

		# CHECK THIS IMP HERE
		Δuᵧ = j < Ny ? uⁿ⁺¹[Nx, j+1, k] - uⁿ⁺¹[Nx, j, k] : uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx, j-1, k]
		
		max_speed = - sqrt(9.80665 * grid.Lz)

    # 2) raw phase speed estimate - handle potential division by zero
    speed_x = abs(Δuₓ * Δt) > 1e-20 ? - (Δuₜ / Δuₓ) * (1 / (1 - (Δuᵧ/Δuₓ * spacing_x / spacing_y)^2)) *
            (spacing_x / Δt) : 0.0
    
    speed_y = abs(Δuᵧ * Δt) > 1e-20 ? - (Δuₜ / Δuᵧ) * (1 / (1 - (Δuₓ/Δuᵧ * spacing_y / spacing_x)^2)) *
            (spacing_y / Δt) : 0.0

    # 5) clamp speeds to [0, max_speed]
    speed_x = clamp(isnan(speed_x) ? 0.0 : speed_x, 0, max_speed)
    speed_y = clamp(isnan(speed_y) ? 0.0 : speed_y, 0, max_speed)

		# 4) nondimensional Courant number
		cₓ = max_speed * Δt / spacing_x
		cᵧ = max_speed * Δt / spacing_y

    neighbor_j = j + (speed_y > 0 ? 1 : -1)
		# 5) Orlanski update
		uᴮ[Nx+1, j, k] = (uᴮ[Nx+1, j, k] - cₓ * uⁿ⁺¹[Nx, j, k] - cᵧ * uⁿ⁺¹[Nx, neighbor_j, k]) / (1 - cₓ - (speed_y > 0 ? cᵧ : -cᵧ))
		# uᴮ[Nx+1, j, k] = orlanski_step(uᴮ[Nx+1, j, k], estimate)
	end
end

@kernel function _update_south_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  i, k = @index(Global, NTuple)
  Nx = size(grid, 1)

  @inbounds begin
    # 1) compute differences in y
    Δuᵧ = uⁿ⁺¹[i, 3, k]   - uⁿ⁺¹[i, 2, k]
    Δuₜ = uⁿ⁺¹[i, 2, k]   - u₁[i, 2, k]

    # CHECK THIS IMP HERE
    Δuₓ = i < Nx ? uⁿ⁺¹[i+1, 2, k] - uⁿ⁺¹[i, 2, k] : uⁿ⁺¹[i, 2, k] - uⁿ⁺¹[i-1, 2, k]

		max_speed = - sqrt(9.80665 * grid.Lz)

    # 2) raw phase speed estimate - handle potential division by zero
    speed_x = abs(Δuₓ * Δt) > 1e-20 ? - (Δuₜ / Δuₓ) * (1 / (1 - (Δuᵧ/Δuₓ * spacing_x / spacing_y)^2)) *
            (spacing_x / Δt) : 0.0
    
    speed_y = abs(Δuᵧ * Δt) > 1e-20 ? - (Δuₜ / Δuᵧ) * (1 / (1 - (Δuₓ/Δuᵧ * spacing_y / spacing_x)^2)) *
            (spacing_y / Δt) : 0.0

    # 5) clamp speeds to [0, max_speed]
    speed_x = clamp(isnan(speed_x) ? 0.0 : speed_x, 0, max_speed)
    speed_y = clamp(isnan(speed_y) ? 0.0 : speed_y, 0, max_speed)

		# 4) nondimensional Courant number
		cₓ = max_speed * Δt / spacing_x
		cᵧ = max_speed * Δt / spacing_y

    neighbor_i = i + (speed_x > 0 ? 1 : -1)
    # 4) Orlanski update at j=1
    uᴮ[i, 1, k] = (uᴮ[i, 1, k] - cₓ * uⁿ⁺¹[i, 2, k] - cᵧ * uⁿ⁺¹[neighbor_i, 2, k]) / (1 - cₓ - (speed_y > 0 ? cᵧ : -cᵧ))
    # uᴮ[i, 1, k] = orlanski_step(uᴮ[i, 1, k], estimate)
  end
end

@kernel function _update_north_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  i, k = @index(Global, NTuple)
  Nx = size(grid, 1)
  Ny = size(grid, 2)

  @inbounds begin
    # 1) compute gradients in y and time
    Δuᵧ = uⁿ⁺¹[i, Ny,   k] - uⁿ⁺¹[i, Ny-1, k]
    Δuₜ = uⁿ⁺¹[i, Ny,   k] - u₁[i, Ny,   k]

    # 2) compute lateral gradient in x
    Δuₓ = i < Nx ? uⁿ⁺¹[i, Ny, k] - uⁿ⁺¹[i-1, Ny, k] : uⁿ⁺¹[i+1, Ny, k] - uⁿ⁺¹[i, Ny, k]

    max_speed = - sqrt(9.80665 * grid.Lz)

    # 2) raw phase speed estimate - handle potential division by zero
    speed_x = abs(Δuₓ * Δt) > 1e-20 ? - (Δuₜ / Δuₓ) * (1 / (1 - (Δuᵧ/Δuₓ * spacing_x / spacing_y)^2)) *
            (spacing_x / Δt) : 0.0
    
    speed_y = abs(Δuᵧ * Δt) > 1e-20 ? - (Δuₜ / Δuᵧ) * (1 / (1 - (Δuₓ/Δuᵧ * spacing_y / spacing_x)^2)) *
            (spacing_y / Δt) : 0.0

    # 5) clamp speeds to [0, max_speed]
    speed_x = clamp(isnan(speed_x) ? 0.0 : speed_x, 0, max_speed)
    speed_y = clamp(isnan(speed_y) ? 0.0 : speed_y, 0, max_speed)

		# 4) nondimensional Courant number
		cₓ = max_speed * Δt / spacing_x
		cᵧ = max_speed * Δt / spacing_y

    neighbor_i = i + (speed_x > 0 ? 1 : -1)

    # 5) Orlanski update at north boundary (j = Ny+1)
		uᴮ[i, Ny+1, k] = (uᴮ[i, Ny+1, k] - cₓ * uⁿ⁺¹[i, Ny, k] - cᵧ * uⁿ⁺¹[neighbor_i, Ny, k]) / (1 - cₓ - (speed_y > 0 ? cᵧ : -cᵧ))
    # uᴮ[i, Ny+1, k] = orlanski_step(uᴮ[i, Ny+1, k], estimate)
  end
end


function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    
    launch!(architecture(grid), grid, :yz,  _update_west_bc, uᴮ, grid, u, u₁, Δt)
    
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    
    launch!(architecture(grid), grid, :yz,  _update_east_bc, uᴮ, grid, u, u₁, Δt)
    
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:south}, v, model)
  vᴮ   = bc.condition.uᴮ
  v₁   = bc.condition.u1
  v    = model.velocities.v
  grid = model.grid
  Δt   = model.clock.last_Δt > 1e10 ? zero(grid) : model.clock.last_Δt

  launch!(architecture(grid), grid, :xz, _update_south_bc, vᴮ, grid, v, v₁, Δt)
  return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:north}, v, model)
  vᴮ   = bc.condition.uᴮ
  v₁   = bc.condition.u1
  v    = model.velocities.v
  grid = model.grid
  Δt   = model.clock.last_Δt > 1e10 ? zero(grid) : model.clock.last_Δt

  launch!(architecture(grid), grid, :xz, _update_north_bc, vᴮ, grid, v, v₁, Δt)
  return nothing
end

function update_boundary_condition!(bcs::FieldBoundaryConditions, vel, model)
  update_boundary_condition!(bcs.west,  Val(:west),  vel, model)
  update_boundary_condition!(bcs.east,  Val(:east),  vel, model)
  update_boundary_condition!(bcs.north, Val(:north), vel, model)
  update_boundary_condition!(bcs.south, Val(:south), vel, model)

  # still conserve volume only in x:
  # impose_volume_conservation!(bcs.west, bcs.east, model)

  return nothing
end


using Oceananigans.Operators

@kernel function _impose_volume_conservation!(uw, ue, grid)
    j = @index(Global, Linear)

    Uc = 0
    for k in 1:grid.Nz
        Uc += @inbounds uw[1, j, k] * Δzᶠᶜᶜ(1, j, k, grid) + ue[1, j, k] * Δzᶠᶜᶜ(grid.Nx+1, j, k, grid)
    end

    for k in 1:grid.Nz
        @inbounds ue[1, j, k] = ue[1, j, k] - Uc / grid.Lz 
        @inbounds uw[1, j, k] = uw[1, j, k] - Uc / grid.Lz 
    end
end

impose_volume_conservation!(u_west, u_east, model) = nothing

function impose_volume_conservation!(u_west::OrlanskiBoundaryCondition, u_east::OrlanskiBoundaryCondition, model)
    grid = model.grid
    launch!(architecture(grid), grid, (grid.Ny, ),  _impose_volume_conservation!, u_west.condition.uᴮ, u_east.condition.uᴮ, grid)
    return nothing
end

# Function to initialize boundary history fields using a kernel to safely copy data
@kernel function _initialize_history_field!(u₁, u, i_offset)
    i, j, k = @index(Global, NTuple)
    
    # Only copy interior points that are needed for the boundary calculation
    if i >= 2 && i <= size(u, 1)
        @inbounds u₁[i, j, k] = u[i, j, k]
    end
end

function initialize_boundary_history!(model)
    u = model.velocities.u
    grid = model.grid
    
    # Use kernel to safely initialize west boundary history
    launch!(architecture(grid), grid, :xyz, _initialize_history_field!, u₁ᵂ, u, 0)
    
    # Use kernel to safely initialize east boundary history
    launch!(architecture(grid), grid, :xyz, _initialize_history_field!, u₁ᴱ, u, 0)
    
    return nothing
end


#####
##### Build the model
#####

model = HydrostaticFreeSurfaceModel(; grid,
                                      timestepper = :SplitRungeKutta3,
                                      free_surface,
                                    #   vertical_coordinate = ZStar(),
                                      boundary_conditions = (; u=u_bcs, v=v_bcs))

#####
##### Set initial conditions
#####

Rx = 250kilometers
Ry = 250kilometers
σ  = 50kilometers

function gaussian_bump(x, y, z=0.0)
  x0, y0, σ = Rx, Ry, 50kilometers
  return 0.1 * exp(-((x - x0)^2 + (y - y0)^2) / σ^2)
end

set!(model, η = gaussian_bump)

initialize_boundary_history!(model)

simulation = Simulation(model, Δt=0.1minutes, stop_time=1days)

#####
##### Attach an output writer and run!
#####

u, v, w = model.velocities
η = model.free_surface.η

output_dir = "validation/open_boundaries/hydro_bc_output_2D"

if !isdir(output_dir)
    mkpath(output_dir)
end

simulation.output_writers[:total_velocities] = JLD2Writer(model, (; u, v, w),
                                                          schedule = TimeInterval(10minutes),
                                                          filename = joinpath(output_dir, "hydrostatic_open_boundaries.jld2"),
                                                          overwrite_existing = true)

simulation.output_writers[:free_surface] = JLD2Writer(model, (; η),
                                                      schedule = TimeInterval(10minutes),
                                                      filename = joinpath(output_dir, "hydrostatic_open_boundaries_free_surface.jld2"),
                                                      overwrite_existing = true)

run!(simulation)

#####
##### Visualize the output
#####

using GLMakie, ColorSchemes

u = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "u")
v = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "v")
w = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "w")
η = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries_free_surface.jld2"), "η")

Nt = length(u.times)

fig = Figure(size = (1000, 500))
axu = Axis(fig[1, 1], title = "u-velocity")
axv = Axis(fig[1, 2], title = "v-velocity")
axw = Axis(fig[2, 1], title = "w-velocity")
axη = Axis(fig[2, 2], title = "free surface")

n = Observable(1)

# ────────────────────────────────
# 1) Make a bigger figure up‐front
fig = Figure(size = (1000, 800), backgroundcolor = :white)    # 1200×1000 pixels
ax  = Axis(fig[1, 1], xlabel="x (m)", ylabel="y (m)",
           title="u‐velocity at t=$(round(u.times[1], digits=2))", backgroundcolor = :white)

# Build your initial heatmap just like before
xs      = range(grid.Lx/grid.Nx/2, step=grid.Lx/grid.Nx, length=grid.Nx)
ys      = range(grid.Ly/grid.Ny/2, step=grid.Ly/grid.Ny, length=grid.Ny)
first   = interior(u[1])[:, :, 1]
hm      = heatmap!(ax, xs, ys, first)
Colorbar(fig[1, 2], hm)

# ────────────────────────────────
# 2) Record with an explicit resolution override
fps        = 30
movie_path = joinpath(output_dir, "u_surface_evolution.mp4")

record(fig, movie_path, 1:Nt;
       framerate = fps,
       resolution = (1920, 1080)) do i    # <-- override to Full HD
  data    = interior(u[i])[:, :, 1]
  hm[3][] = data                         # update the color‐matrix
  ax.title = "u‐velocity at t=$(round(u.times[i], digits=2))"
end

@info (time_ns() - wt) / 1e9