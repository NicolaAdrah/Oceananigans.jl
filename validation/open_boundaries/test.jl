using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Models
using Oceananigans.Grids: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators

using KernelAbstractions: @kernel, @index

wt = time_ns()

# Set up the domain - fixed to include Nz dimension
grid = RectilinearGrid(size = (50, 50, 1),     # Must include all 3 dimensions (Nx, Ny, Nz)
                       x = (0, 500kilometers),
                       y = (0, 500kilometers),
                       topology = (Bounded, Bounded, Bounded),
                       z = (-10, 0))

free_surface = ExplicitFreeSurface()
                      
##### 
##### Build boundary conditions
#####

using Oceananigans.BoundaryConditions: Open
import Oceananigans.BoundaryConditions: getbc, update_boundary_condition!

struct OrlanskiBoundary{U, U1, U2, C}
    uᴮ :: U   # ghost‐cell field (the boundary values)
    u1 :: U1  # one‐step‐lagged interior values
    u2 :: U2  # two‐step‐lagged interior values (not used in final)
    c :: C    # phase‐speed field (can vary in space)
end

OrlanskiBoundaryCondition = BoundaryCondition{<:Open, <:OrlanskiBoundary}

uᵂ  = Field{Nothing, Nothing, Center}(grid)   # current ghost‐cell values
uᴱ  = Field{Nothing, Nothing, Center}(grid)   # current ghost‐cell values
uᴺ  = Field{Nothing, Nothing, Center}(grid)   # current ghost‐cell values north
uˢ  = Field{Nothing, Nothing, Center}(grid)   # current ghost‐cell values south

u₁ᵂ = Field{Face, Nothing, Center}(grid)      # u at previous time
u₁ᴱ = Field{Face, Nothing, Center}(grid)      # u at previous time
u₁ᴺ = Field{Nothing, Face, Center}(grid)      # u at previous time north
u₁ˢ = Field{Nothing, Face, Center}(grid)      # u at previous time south

u₂ᵂ = Field{Face, Nothing, Center}(grid)      # u two steps back
u₂ᴱ = Field{Face, Nothing, Center}(grid)      # u two steps back
u₂ᴺ = Field{Nothing, Face, Center}(grid)      # u two steps back north
u₂ˢ = Field{Nothing, Face, Center}(grid)      # u two steps back south

# Phase speed fields
cʷ = Field{Nothing, Nothing, Center}(grid)    # phase speed fields
cᵉ = Field{Nothing, Nothing, Center}(grid)    # phase speed fields
cⁿ = Field{Nothing, Nothing, Center}(grid)    # phase speed fields north
cˢ = Field{Nothing, Nothing, Center}(grid)    # phase speed fields south

c0 = sqrt(9.80655 * grid.Lz)                  # √(gH)

set!(cʷ, -c0)
set!(cᵉ, c0)
set!(cⁿ, c0)
set!(cˢ, -c0)

# Wrap our struct in an OpenBoundaryCondition so Oceananigans knows it's a radiation‐type.
u_west = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ, u₂ᵂ, cʷ))
u_east = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ, u₂ᴱ, cᵉ))
u_north = OpenBoundaryCondition(OrlanskiBoundary(uᴺ, u₁ᴺ, u₂ᴺ, cⁿ))
u_south = OpenBoundaryCondition(OrlanskiBoundary(uˢ, u₁ˢ, u₂ˢ, cˢ))

# Define getbc to fetch the boundary value from our ghost‐cell field uᴮ.
@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]  # For west/east boundaries
@inline getbc(bc::OrlanskiBoundaryCondition, i, k, args...) = bc.condition.uᴮ[i, 1, k]  # For north/south boundaries

u_bcs = FieldBoundaryConditions(west=u_west, east=u_east, north=u_north, south=u_south)

# Improved boundary condition update kernels
@kernel function _update_north_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    i, k = @index(Global, NTuple)
    Ny = size(grid, 2)
    
    @inbounds begin
        # Calculate gradients
        dVdt = (uⁿ⁺¹[i, Ny, k] - u₁[i, Ny, k])
        dVdy = (uⁿ⁺¹[i, Ny, k] - uⁿ⁺¹[i, Ny-1, k])
        
        # Calculate horizontal gradients for cross-terms
        dVdx_east = i < grid.Nx ? (uⁿ⁺¹[i+1, Ny, k] - uⁿ⁺¹[i, Ny, k]) : 0.0
        dVdx_west = i > 1 ? (uⁿ⁺¹[i, Ny, k] - uⁿ⁺¹[i-1, Ny, k]) : 0.0
        
        # Determine which gradient to use based on flow direction
        dVdx = dVdt > 0 ? dVdx_west : dVdx_east
        
        # Constants for the radiation calculation
        eps = 1e-20
        cff = max(dVdy^2 + dVdx^2, eps)
        
        # Compute final boundary value
        if dVdt * dVdy < 0.0  # Outgoing wave
            # Apply standard Orlanski with damping
            Ce = dVdt * dVdy
            Cx = min(cff, max(dVdt * dVdx, -cff))
            
            numerator = (cff * uᴮ[i, Ny+1, k] + Ce * uⁿ⁺¹[i, Ny, k] - 
                        max(Cx, 0.0) * dVdx_west - min(Cx, 0.0) * dVdx_east)
            denominator = cff + Ce
            
            uᴮ[i, Ny+1, k] = numerator / denominator
        else  # Incoming wave or zero gradient
            # Apply stronger damping
            uᴮ[i, Ny+1, k] = uⁿ⁺¹[i, Ny, k]
        end
        
    end
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    j, k = @index(Global, NTuple)
    Nx = size(grid, 1)
    
    @inbounds begin
        # Calculate gradients
        dVdt = (uⁿ⁺¹[Nx, j, k] - u₁[Nx, j, k])
        dVdx = (uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx-1, j, k])
        
        # Calculate vertical gradients for cross-terms
        dVdy_north = j < grid.Ny ? (uⁿ⁺¹[Nx, j+1, k] - uⁿ⁺¹[Nx, j, k]) : 0.0
        dVdy_south = j > 1 ? (uⁿ⁺¹[Nx, j, k] - uⁿ⁺¹[Nx, j-1, k]) : 0.0
        
        # Determine which gradient to use based on flow direction
        dVdy = dVdt > 0 ? dVdy_south : dVdy_north
        
        # Constants for the radiation calculation
        eps = 1e-20
        cff = max(dVdx^2 + dVdy^2, eps)
        
        # Compute final boundary value
        if dVdt * dVdx < 0.0  # Outgoing wave
            # Apply standard Orlanski with damping
            Ce = dVdt * dVdx
            Cy = min(cff, max(dVdt * dVdy, -cff))
            
            numerator = (cff * uᴮ[Nx+1, j, k] + Ce * uⁿ⁺¹[Nx, j, k] - 
                        max(Cy, 0.0) * dVdy_south - min(Cy, 0.0) * dVdy_north)
            denominator = cff + Ce
            
            uᴮ[Nx+1, j, k] = numerator / denominator
        else  # Incoming wave or zero gradient
            # Apply stronger damping
            uᴮ[Nx+1, j, k] = uⁿ⁺¹[Nx, j, k]
        end

    end
end

# Reusing your original west and south BC kernels for completeness
@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    j, k = @index(Global, NTuple)

    @inbounds begin
        # Calculate time derivative
        dVdt = (uⁿ⁺¹[2, j, k] - u₁[2, j, k])
        
        # Calculate spatial derivatives
        dVdx = (uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k])
 
        # Gradient for cross-shore variation
        grad_south = j > 1 ? (uⁿ⁺¹[2, j-1, k] - uⁿ⁺¹[2, j, k]) : 0.0
        grad_north = j < grid.Ny ? (uⁿ⁺¹[2, j+1, k] - uⁿ⁺¹[2, j, k]) : 0.0
        
        # Upwinding logic
        if (dVdt * dVdx) < 0.0
            dVdt = 0.0  # No incoming waves
        end
        
        # Choose gradient based on wave direction
        dVdy = 0.0
        if (dVdt * (grad_south + grad_north)) > 0.0
            dVdy = grad_south
        else
            dVdy = grad_north
        end
        
        # Calculate radiation coefficients
        eps = 1e-10
        cff = max(dVdx * dVdx + dVdy * dVdy, eps)
        Cx = min(cff, max(dVdt * dVdx, -cff))
        Ce = dVdt * dVdy
        
        # Radiation boundary condition
        numerator = cff * uᴮ[1, j, k] + Ce * uⁿ⁺¹[2, j, k] - 
                   max(Cx, 0.0) * grad_south - min(Cx, 0.0) * grad_north
        denominator = cff + Ce
        
        uᴮ[1, j, k] = numerator / denominator
        
    end
end

@kernel function _update_south_bc(uᴮ, grid, uⁿ⁺¹, u₁, u₂, Δt, c)
    i, k = @index(Global, NTuple)
    
    @inbounds begin
        # Calculate time derivative
        dVdt = (uⁿ⁺¹[i, 2, k] - u₁[i, 2, k])
        
        # Calculate spatial derivatives
        dVdy = (uⁿ⁺¹[i, 3, k] - uⁿ⁺¹[i, 2, k])
 
        # Gradient for cross-shore variation
        grad_west = i > 1 ? (uⁿ⁺¹[i-1, 2, k] - uⁿ⁺¹[i, 2, k]) : 0.0
        grad_east = i < grid.Nx ? (uⁿ⁺¹[i+1, 2, k] - uⁿ⁺¹[i, 2, k]) : 0.0
        
        # Upwinding logic
        if (dVdt * dVdy) < 0.0
            dVdt = 0.0  # No incoming waves
        end
        
        # Choose gradient based on wave direction
        dVdx = 0.0
        if (dVdt * (grad_west + grad_east)) > 0.0
            dVdx = grad_west
        else
            dVdx = grad_east
        end
        
        # Calculate radiation coefficients
        eps = 1e-10
        cff = max(dVdy * dVdy + dVdx * dVdx, eps)
        Cx = min(cff, max(dVdt * dVdx, -cff))
        Ce = dVdt * dVdy
        
        # Apply damping factor
        damping_factor = 0.0
        
        # Radiation boundary condition
        numerator = cff * uᴮ[i, 1, k] + Ce * uⁿ⁺¹[i, 2, k] - 
                   max(Cx, 0.0) * grad_west - min(Cx, 0.0) * grad_east
        denominator = cff + Ce
        
        boundary_value = numerator / denominator
        uᴮ[i, 1, k] = (1.0 - damping_factor) * boundary_value
        
    end
end

# Update functions for boundary conditions
function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    grid = model.grid
    
    # Use actual time step from model
    Δt = model.clock.last_Δt
    if Δt == 0 || Δt > 1e10
        Δt = 0.1 * 60.0  # fallback to 0.1 minutes in seconds
    end
    
    launch!(architecture(grid), grid, :yz, _update_west_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    grid = model.grid
    
    # Use actual time step from model
    Δt = model.clock.last_Δt
    if Δt == 0 || Δt > 1e10
        Δt = 0.1 * 60.0  # fallback to 0.1 minutes in seconds
    end
    
    launch!(architecture(grid), grid, :yz, _update_east_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:north}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    grid = model.grid
    
    # Use actual time step from model
    Δt = model.clock.last_Δt
    if Δt == 0 || Δt > 1e10
        Δt = 0.1 * 60.0  # fallback to 0.1 minutes in seconds
    end
    
    launch!(architecture(grid), grid, :xz, _update_north_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:south}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u₂ = bc.condition.u2
    c = bc.condition.c
    grid = model.grid
    
    # Use actual time step from model
    Δt = model.clock.last_Δt
    if Δt == 0 || Δt > 1e10
        Δt = 0.1 * 60.0  # fallback to 0.1 minutes in seconds
    end
    
    launch!(architecture(grid), grid, :xz, _update_south_bc, uᴮ, grid, u, u₁, u₂, Δt, c)
    return nothing
end

# Optional: volume conservation function
@kernel function _impose_volume_conservation!(uw, ue, un, us, grid)
    j, k = @index(Global, NTuple)

    # Calculate net flux through all boundaries
    net_flux_x = 0.0
    net_flux_y = 0.0
    
    # Sum fluxes in x-direction
    for i in 1:grid.Nx
        net_flux_x += uw[1, j, k] * Δzᶠᶜᶜ(1, j, k, grid) - 
                      ue[1, j, k] * Δzᶠᶜᶜ(grid.Nx+1, j, k, grid)
    end
    
    # Sum fluxes in y-direction
    for i in 1:grid.Nx
        net_flux_y += us[i, 1, k] * Δzᶠᶜᶜ(i, 1, k, grid) - 
                      un[i, 1, k] * Δzᶠᶜᶜ(i, grid.Ny+1, k, grid)
    end
    
    # Total net flux
    net_flux = net_flux_x + net_flux_y
    
    # Distribute correction evenly among all boundaries
    correction = net_flux / (2 * (grid.Nx + grid.Ny) * grid.Lz)
    
    # Apply corrections
    uw[1, j, k] = uw[1, j, k] - correction
    ue[1, j, k] = ue[1, j, k] + correction
    us[j, 1, k] = us[j, 1, k] - correction
    un[j, 1, k] = un[j, 1, k] + correction
end

function impose_volume_conservation!(u_west, u_east, u_north, u_south, model)
    grid = model.grid
    launch!(architecture(grid), grid, :yz, _impose_volume_conservation!, 
            u_west.condition.uᴮ, u_east.condition.uᴮ, 
            u_north.condition.uᴮ, u_south.condition.uᴮ, grid)
    return nothing
end

#####
##### Build the model
#####

model = HydrostaticFreeSurfaceModel(; grid,
                                      free_surface,
                                      boundary_conditions = (; u=u_bcs))

#####
##### Set initial conditions
#####

Rx = 250kilometers
Ry = 250kilometers
σ  = 50kilometers

# Modified to use both x and y coordinates for a centered bump
gaussian_bump(x, y, z) = 0.1 * exp(-((x - Rx)^2 + (y - Ry)^2) / σ^2)

set!(model, η = gaussian_bump)

simulation = Simulation(model, Δt=0.1minutes, stop_time=3days)

#####
##### Attach an output writer and run!
#####

u, v, w = model.velocities
η = model.free_surface.η

# Create output directory
output_dir = "validation/open_boundaries/hydro_bc_output_test"
isdir(output_dir) || mkdir(output_dir)

simulation.output_writers[:total_velocities] = JLD2Writer(model, (; u, v, w),
                                                         schedule = TimeInterval(10minutes),
                                                         filename = joinpath(output_dir, "velocities.jld2"),
                                                         overwrite_existing = true)

simulation.output_writers[:free_surface] = JLD2Writer(model, (; η),
                                                     schedule = TimeInterval(10minutes),
                                                     filename = joinpath(output_dir, "free_surface.jld2"),
                                                     overwrite_existing = true)

# Run the simulation
@info "Starting simulation..."
run!(simulation)
@info "Simulation complete!"

#####
##### Visualize the output
#####

using GLMakie

# Load the data from the JLD2 files
u = FieldTimeSeries(joinpath(output_dir, "velocities.jld2"), "u")
v = FieldTimeSeries(joinpath(output_dir, "velocities.jld2"), "v")
w = FieldTimeSeries(joinpath(output_dir, "velocities.jld2"), "w")
η = FieldTimeSeries(joinpath(output_dir, "free_surface.jld2"), "η")

Nt = length(u.times)

# Create a figure with the same layout as your original code
fig = Figure(size = (1000, 1000))
axu = Axis(fig[1, 1], title = "u-velocity")
axv = Axis(fig[1, 2], title = "v-velocity")
axw = Axis(fig[2, 1], title = "w-velocity")
axη = Axis(fig[2, 2], title = "free surface")

n = Observable(1)

# Create observables for each plot - using the same slices as your original code
un = @lift(interior(u[$n], :, 1, 1))  # x-slice of u at y=1, z=1
vn = @lift(interior(v[$n], :, 1, 1))  # x-slice of v at y=1, z=1
wn = @lift(interior(w[$n], :, 1, 1))  # x-slice of w at y=1, z=1
ηn = @lift(interior(η[$n], :, 1, 1))  # x-slice of η at y=1, z=1

# Create line plots
lines!(axu, un)
lines!(axv, vn)
lines!(axw, wn)
lines!(axη, ηn)

# Set the same y-limits as your original code
ylims!(axu, (-2e-1, 2e-1))
ylims!(axv, (-2e-1, 2e-1))
ylims!(axw, (-1e-5, 1e-5))
ylims!(axη, (0.0, 0.15))

# Create the animation using the same filename as your original code
@info "Creating animation..."
record(fig, joinpath(output_dir, "bc_01.mp4"), 1:Nt) do i 
    @info "doing iteration $i of $Nt"
    n[] = i
end

@info "Animation saved to $(joinpath(output_dir, "2dbc_01.mp4"))"

# Also create a 2D visualization to better observe boundary behavior
fig2 = Figure(size = (1000, 800))
ax2d = Axis(fig2[1, 1], 
          xlabel = "x (km)", 
          ylabel = "y (km)",
          title = "Free Surface 2D View",
          aspect = DataAspect())

# Convert coordinates to km for better visualization
x_coords = range(0, 500, length=grid.Nx)  # 500 km domain
y_coords = range(0, 500, length=grid.Ny)  # 500 km domain

# Create a heatmap of the free surface
η_plot = @lift interior(η[$n], :, :, 1)
heatmap!(ax2d, x_coords, y_coords, η_plot, colormap = :balance, 
         colorrange = (-0.1, 0.1))

Colorbar(fig2[1, 2], colormap = :balance, limits = (-0.1, 0.1),
         label = "Free Surface Elevation (m)")

# Create the 2D animation
record(fig2, joinpath(output_dir, "free_surface_2d.mp4"), 1:Nt, framerate = 15) do i
    n[] = i
    ax2d.title = "Free Surface 2D - Time: $(round(u.times[i]/60, digits=1)) minutes"
end

@info "2D visualization saved to $(joinpath(output_dir, "free_surface_2d.mp4"))"