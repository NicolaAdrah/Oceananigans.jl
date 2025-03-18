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
                          # y = (0, 500kilometers),
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

uᵂ  = Field{Nothing, Nothing, Center}(grid)
uᴱ  = Field{Nothing, Nothing, Center}(grid)
u₁ᵂ = Field{Nothing, Nothing, Center}(grid)
u₁ᴱ = Field{Nothing, Nothing, Center}(grid)

u_west = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ))
u_east = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ))

@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]

u_bcs = FieldBoundaryConditions(west=u_west, east=u_east)

@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁)
    j, k = @index(Global, NTuple)

    Δut = @inbounds uⁿ⁺¹[2, j, k] -   u₁[1, j, k]
    Δux = @inbounds uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k]

    r = ifelse(Δux == 0, zero(grid), Δut / Δux)

    @inbounds uᴮ[1, j, k] = (uᴮ[1, j, k] + r * uⁿ⁺¹[2, j, k]) / (1 + r)
    @inbounds u₁[1, j, k] = uⁿ⁺¹[2, j, k]

    if isnan(uᴮ[1, j, k]) || isnan(r) 
        @show uᴮ[1, j, k], r, Δut, Δux
    end
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁)
    j, k = @index(Global, NTuple)

    Δut = @inbounds uⁿ⁺¹[grid.Nx, j, k] - u₁[1, j, k]
    Δux = @inbounds uⁿ⁺¹[grid.Nx, j, k] - uⁿ⁺¹[grid.Nx-1, j, k]

    r = ifelse(Δux == 0, zero(grid), Δut / Δux)

    @inbounds uᴮ[1, j, k] = (uᴮ[1, j, k] + r * uⁿ⁺¹[grid.Nx, j, k]) / (1 + r)
    @inbounds u₁[1, j, k] =  uⁿ⁺¹[grid.Nx, j, k]
end


function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid

    launch!(architecture(grid), grid, :yz,  _update_west_bc, uᴮ, grid, u, u₁)
    
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid

    launch!(architecture(grid), grid, :yz,  _update_east_bc, uᴮ, grid, u, u₁)
    
    return nothing
end

#####
##### Build the model
#####

model = HydrostaticFreeSurfaceModel(; grid,
                                      free_surface,
                                      vertical_coordinate = ZStar(),
                                      boundary_conditions = (; u=u_bcs))

#####
##### Set initial conditions
#####

Rx = 250kilometers
Ry = 250kilometers
σ  = 50kilometers

gaussian_bump(x, z) = 0.1 * exp(-((x - Rx)^2 / σ^2))

set!(model, η = gaussian_bump)

simulation = Simulation(model, Δt=1minutes, stop_time=1days)

#####
##### Attach an output writer and run!
#####

u, v, w = model.velocities
η = model.free_surface.η

simulation.output_writers[:total_velocities] = JLD2Writer(model, (; u, v, w),
                                                          schedule = TimeInterval(10minutes),
                                                          filename = "hydrostatic_open_boundaries.jld2",
                                                          overwrite_existing = true)

simulation.output_writers[:free_surface] = JLD2Writer(model, (; η),
                                                      schedule = TimeInterval(10minutes),
                                                      filename = "hydrostatic_open_boundaries_free_surface.jld2",
                                                      overwrite_existing = true)

run!(simulation)

#####
##### Visualize the output
#####

using GLMakie

u = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "u")
v = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "v")
w = FieldTimeSeries("hydrostatic_open_boundaries.jld2", "w")
η = FieldTimeSeries("hydrostatic_open_boundaries_free_surface.jld2", "η")

Nt = length(u.times)

fig = Figure(size = (1000, 500))
axu = Axis(fig[1, 1], title = "u-velocity")
axv = Axis(fig[1, 2], title = "v-velocity")
axw = Axis(fig[2, 1], title = "w-velocity")
axη = Axis(fig[2, 2], title = "free surface")

n = Observable(1)

un = @lift(interior(u[$n], :, 1, 10))
vn = @lift(interior(v[$n], :, 1, 10))
wn = @lift(interior(w[$n], :, 1, 5))
ηn = @lift(interior(η[$n], :, 1, 1))

lines!(axu, un)
lines!(axv, vn)
lines!(axw, wn)
lines!(axη, ηn)

record(fig, "hydrostatic_open_boundaries.mp4", 1:Nt) do i 
    @info "doing iteration $i of $Nt"
    n[] = i
end

@info (time_ns() - wt) / 1e9
