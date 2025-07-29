using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Grids: architecture
using Oceananigans.Models

using KernelAbstractions: @kernel, @index

wt = time_ns()

z = MutableVerticalDiscretization((-10, 0))

grid = RectilinearGrid(size = (50, 10),     
                          x = (0, 500kilometers),
                        #   y = (0, 500kilometers),
                          topology = (Bounded, Flat, Bounded),
                          z = z)

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
uᵂ  = Field{Nothing, Nothing, Center}(grid)
uᴱ  = Field{Nothing, Nothing, Center}(grid)
uᴺ  = Field{Nothing, Nothing, Center}(grid)
uˢ  = Field{Nothing, Nothing, Center}(grid)

# History-field allocation
u₁ᵂ = Field{Nothing, Nothing, Center}(grid)
u₁ᴱ = Field{Nothing, Nothing, Center}(grid)
u₁ᴺ = Field{Nothing, Nothing, Center}(grid)
u₁ˢ = Field{Nothing, Nothing, Center}(grid)

# Initialize the boundary fields with zeros or appropriate initial conditions
fill!(uᵂ, 0)
fill!(uᴱ, 0)
fill!(uᴺ, 0)
fill!(uˢ, 0)
fill!(u₁ᵂ, 0)
fill!(u₁ᴱ, 0)
fill!(u₁ᴺ, 0)
fill!(u₁ˢ, 0)

u_west  = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ))
u_east  = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ))
# u_north = OpenBoundaryCondition(OrlanskiBoundary(uᴺ, u₁ᴺ))
# u_south = OpenBoundaryCondition(OrlanskiBoundary(uˢ, u₁ˢ))

@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]

# CHECK HERE
# @inline getbc(bc::OrlanskiBoundaryCondition, i, j, k, args...) = bc.condition.uᴮ[i, 1, k]

# u_bcs = FieldBoundaryConditions(west=u_west, east=u_east, north=u_north, south=u_south)
u_bcs = FieldBoundaryConditions(west=u_west, east=u_east)

@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  j, k = @index(Global, NTuple)

    @inbounds begin
        # 0) Error 
        # δ = 0.006

        # 1) compute differences
        Δuₓ = uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k]

        Δuₜ = uⁿ⁺¹[2, j, k] - u₁[2, j, k]

        max_speed = - sqrt(9.80665 * grid.Lz)

        # 2) raw phase speed estimate - handle potential division by zero
        speed = if abs(Δuₓ * Δt) > 1e-20
            - (Δuₜ * Δxᶠᶜᶜ(1, j, k, grid)) / (Δuₓ * Δt) 
        else
            0.0  # Default to zero if gradient is too small
        end
        
        # Handle NaN values
        if isnan(speed)
            speed = 0.0  # Default to zero for first time step
        end

        # 3) Following Orlanski's conditions, c_x should be between 0 and max_speed
        if speed < 0.0
            speed = 0.0
        elseif speed > max_speed
            speed = max_speed
        end

        # 4) nondimensional Courant number
        c = speed * Δt / Δxᶠᶜᶜ(1, j, k, grid)

        # 5) Orlanski update
        uᴮ[1, j, k] = (uᴮ[1, j, k] - c * uⁿ⁺¹[2, j, k]) / (1 - c)
        
        # Store the boundary value we just computed
        u₁[2, j, k] = uᴮ[1, j, k]
    end
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
  j, k = @index(Global, NTuple)
  Nx   = size(grid, 1)

    @inbounds begin
        # 0) Error 
        # δ = 0.006

        # 1) compute differences
        Δuₓ = uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx-1, j, k]

        Δuₜ = uⁿ⁺¹[Nx, j, k] - u₁[Nx, j, k]

        max_speed = - sqrt(9.80665 * grid.Lz)

        # 2) raw phase speed estimate - handle potential division by zero
        speed = if abs(Δuₓ * Δt) > 1e-20
            # @warn "computing c_x"
            - (Δuₜ * Δxᶠᶜᶜ(Nx+1, j, k, grid)) / (Δuₓ * Δt) 
        else
            # @warn "computing c_x = 0.0"
            0.0  # Default to zero if gradient is too small
        end
        
        # Handle NaN values
        if isnan(speed)
            # @warn "there is NaN"
            speed = 0.0  # Default to zero for first time step
        end

        # 3) Following Orlanski's conditions, c_x should be between 0 and max_speed
        if speed < 0.0 
            # @warn "speed < 0.0"
            speed = 0.0
        elseif speed > max_speed
            # @warn "speed > max_speed"
            speed = max_speed
        end
        
        # 4) nondimensional Courant number
        c = speed * Δt / Δxᶠᶜᶜ(Nx+1, j, k, grid)

        # 5) Orlanski update
        uᴮ[Nx+1, j, k] = (uᴮ[Nx+1, j, k] - c * uⁿ⁺¹[Nx, j, k]) / (1 - c)

        # Store the boundary value we just computed
        u₁[Nx, j, k] = uᴮ[Nx+1, j, k]  
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

function update_boundary_condition!(bcs::FieldBoundaryConditions, u, model)
    update_boundary_condition!(bcs.west, Val(:west), u, model)
    update_boundary_condition!(bcs.east, Val(:east), u, model)
    
    impose_volume_conservation!(bcs.west, bcs.east, model)
    
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
                                      free_surface,
                                    #   vertical_coordinate = ZStar(),
                                      boundary_conditions = (; u=u_bcs))

#####
##### Set initial conditions
#####

Rx = 250kilometers
Ry = 250kilometers
σ  = 50kilometers

gaussian_bump(x, z) = 0.1 * exp(-((x - Rx)^2 / σ^2))

set!(model, η = gaussian_bump)

initialize_boundary_history!(model)

simulation = Simulation(model, Δt=0.1minutes, stop_time=1days)

#####
##### Attach an output writer and run!
#####

u, v, w = model.velocities
η = model.free_surface.η

output_dir = "validation/open_boundaries/hydro_bc_output"

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

using GLMakie

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

un = @lift(interior(u[$n], :, 1, 10))
vn = @lift(interior(v[$n], :, 1, 10))
wn = @lift(interior(w[$n], :, 1, 11))
ηn = @lift(interior(η[$n], :, 1, 1))

lines!(axu, un)
lines!(axv, vn)
lines!(axw, wn)
lines!(axη, ηn)

ylims!(axw, (-1e-5, 1e-5))
ylims!(axη, (-0.1, 0.2))

record(fig, joinpath(output_dir, "cu_hydro_bc.mp4"), 1:Nt) do i
    @info "doing iteration $i of $Nt"
    n[] = i
end

@info (time_ns() - wt) / 1e9