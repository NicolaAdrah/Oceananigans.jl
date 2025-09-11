include("perturbation_advection_open_boundary_matching_scheme.jl")
using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: WENO
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Grids: Face, Center, nodes
using Oceananigans.Fields: Field
using Oceananigans: Simulation, run!, set!, FieldTimeSeries, PrescribedVelocityFields

output_dir = "validation/open_boundaries/PerturbationAdvection_OGS/output"
time_int   = 30minutes

grid = RectilinearGrid(CPU();
    size     = (50, 10),
    x        = (0, 500kilometers),
    topology = (Bounded, Flat, Bounded),
    z        = (-10, 0)
)

uB_east = Field{Nothing, Center, Center}(grid);
uB_west = Field{Nothing, Center, Center}(grid);

u_bcs = FieldBoundaryConditions(
    east = PerturbationAdvectionOpenBoundaryCondition(uB_east),
    west = PerturbationAdvectionOpenBoundaryCondition(uB_west)
)

# η_bcs = FieldBoundaryConditions(
#     east = PerturbationAdvectionOpenBoundaryCondition(uB_east),
#     west = PerturbationAdvectionOpenBoundaryCondition(uB_west)
# )

model = HydrostaticFreeSurfaceModel(;
    grid,
    free_surface = ImplicitFreeSurface(),
    boundary_conditions = (; u = u_bcs),
    # velocities        = PrescribedVelocityFields(u = 1.0),  # constant eastward flow
    # tracers           = (:c,),
    # tracer_advection  = WENO(),
)

x0 = 250kilometers; σx = 50kilometers
η₀(x, z) = 0.1 * exp(-((x - x0)^2) / (2σx^2))
ϕ(x, z)    = exp(-((x - x0)^2) / (2σx^2))

# set!(model; c = ϕ)
set!(model; η = η₀)

simulation = Simulation(model; Δt = 5minutes, stop_time = 5days)

# c = model.tracers.c
# simulation.output_writers[:tracer] = JLD2Writer(model, (; c,),
#     schedule            = TimeInterval(time_int),
#     filename            = joinpath(output_dir, "tracer.jld2"),
#     overwrite_existing  = true
# )

η = model.free_surface.η
simulation.output_writers[:tracer] = JLD2Writer(model, (; η,),
    schedule            = TimeInterval(time_int),
    filename            = joinpath(output_dir, "free_surface.jld2"),
    overwrite_existing  = true
)


@info "Running..." 
run!(simulation)
@info "Done."

using GLMakie
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: nodes
using Oceananigans.Fields: interior

# c_tr = FieldTimeSeries(joinpath(output_dir, "tracer.jld2"), "c")
# Nt   = length(c_tr.times)
η_tr = FieldTimeSeries(joinpath(output_dir, "free_surface.jld2"), "η")
Nt   = length(η_tr.times)

xC, _, zC = nodes(model.grid, Center(), Center(), Center())
Nx = length(xC)

n      = Observable(1)
# title_obs = @lift("Tracer c — t = $(round(c_tr.times[$n]/hour, digits=2)) h")
title_obs = @lift("Free surface η — t = $(round(η_tr.times[$n]/hour, digits=2)) h")

# 2D tracer slice (x–z, at y=1 since it's Flat)
# c_slice = @lift(Array(interior(c_tr[$n]))[:, 1, :])
η_slice = @lift(Array(interior(η_tr[$n]))[:, 1, :])

# c_line  = @lift(interior(c_tr[$n], :, 1, 1))
η_line  = @lift(interior(η_tr[$n], :, 1, 1))

fig = Figure(resolution = (1000, 800))

ax1 = Axis(fig[1, 1],
           xlabel = "x (km)", ylabel = "z (m)",
           title  = title_obs)
# hm = heatmap!(ax1, xC ./ kilometer, zC, c_slice)
# Colorbar(fig[1, 2], hm, label = "c")

# ax2 = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "c",
#            title = "Tracer bump at y=1, z=1")
# lines!(ax2, xC ./ kilometer, c_line)
ax2 = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "c",
           title = "Free Surface bump at y=1, z=1")
lines!(ax2, xC ./ kilometer, η_line)

mp4file = joinpath(output_dir, "free_surface.mp4")
record(fig, mp4file, 1:Nt; framerate = 12) do i
    @info "frame $i / $Nt"
    n[] = i
end

println("Saved $mp4file")