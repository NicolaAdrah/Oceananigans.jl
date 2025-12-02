using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvection
using Oceananigans.Units
using Oceananigans.Advection: WENO
using Oceananigans.OutputWriters: JLD2Writer, TimeInterval
using Oceananigans.Grids: Face, Center, nodes
using Oceananigans.Fields: Field
using Oceananigans: Simulation, run!, set!, FieldTimeSeries, PrescribedVelocityFields
using Oceananigans.Diagnostics: AdvectiveCFL
using Printf

output_dir = "validation/open_boundaries/PerturbationAdvection_OGS/output"
time_int   = 30minutes

grid = RectilinearGrid(CPU();
    size     = (50, 10),
    x        = (0, 500kilometers),
    topology = (Bounded, Flat, Bounded),
    z        = (-10, 0)
)

uB_east = Field{Nothing, Nothing, Center}(grid);
uB_west = Field{Nothing, Nothing, Center}(grid);

u_bcs = FieldBoundaryConditions(
    east = OpenBoundaryCondition(uB_east; scheme = PerturbationAdvection(inflow_timescale = 2hours,
                                                   outflow_timescale = 8hours)),
    west = OpenBoundaryCondition(uB_west; scheme = PerturbationAdvection(inflow_timescale = 2hours,
                                                   outflow_timescale = 8hours))
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
    tracers           = (:c,),
    tracer_advection  = WENO(),
)


x0 = 250kilometers; σx = 50kilometers
η₀(x, z) = 0.1 * exp(-((x - x0)^2) / (2σx^2))
ϕ(x, z)    = exp(-((x - x0)^2) / (2σx^2))

# set!(model; c = ϕ)
set!(model; η = η₀)

simulation = Simulation(model; Δt = 5minutes, stop_time = 10days)

# c = model.tracers.c
# simulation.output_writers[:tracer] = JLD2Writer(model, (; c,),
#     schedule            = TimeInterval(time_int),
#     filename            = joinpath(output_dir, "tracer.jld2"),
#     overwrite_existing  = true
# )

η = model.free_surface.η
simulation.output_writers[:free_surface] = JLD2Writer(model, (; η,),
    schedule            = TimeInterval(time_int),
    filename            = joinpath(output_dir, "free_surface.jld2"),
    overwrite_existing  = true
)

u = model.velocities.u
simulation.output_writers[:velocities] = JLD2Writer(model, (; u,),
    schedule            = TimeInterval(time_int),
    filename            = joinpath(output_dir, "velocities.jld2"),
    overwrite_existing  = true
)

β = model.free_surface.barotropic_volume_flux.u
simulation.output_writers[:barotropic_volume_flux] = JLD2Writer(model, (; β,),
    schedule            = TimeInterval(time_int),
    filename            = joinpath(output_dir, "barotropic_volume_flux.jld2"),
    overwrite_existing  = true
)

advective_cfl = AdvectiveCFL(simulation.Δt)
simulation.callbacks[:cfl_monitor] = Callback(IterationInterval(10)) do sim
    current_cfl = advective_cfl(sim.model)
    @info "Advective CFL" iteration = sim.model.clock.iteration cfl = current_cfl
end

@info "Running..." 
function progress(sim) 
    u, v, w = sim.model.velocities
    # T, S = sim.model.tracers

    @info @sprintf("Time: %s, Iteration %d, Δt %s, max(vel): (%.2e, %.2e, %.2e)\n",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))
                #    maximum(abs, T), maximum(abs, S),    )
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
run!(simulation)
@info "Done."

using GLMakie
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: nodes
using Oceananigans.Fields: interior

η = FieldTimeSeries(joinpath(output_dir, "free_surface.jld2"), "η")
u = FieldTimeSeries(joinpath(output_dir, "velocities.jld2"), "u")
β = FieldTimeSeries(joinpath(output_dir, "barotropic_volume_flux.jld2"), "β")

Nt   = length(η.times)

fig = Figure(resolution = (1000, 500))
axη = Axis(fig[1, 1], title = "free surface")
axu = Axis(fig[1, 2], title = "velocities")
axβ = Axis(fig[1, 3], title = "barotropic volume flux")


n = Observable(1)
ηn = @lift(interior(η[$n], :, 1, 1))
un = @lift(interior(u[$n], :, 1, 1))
βn = @lift(interior(β[$n], :, 1, 1))

lines!(axη, ηn); lines!(axu, un); lines!(axβ, βn);
ylims!(axη, (-0.1, 0.2)); ylims!(axu, (-0.07, 0.07)); ylims!(axβ, (-0.5, 0.5));

hlines!(axu, [0.045,-0.045]; color = :red, linestyle = :dash)
hlines!(axβ, [0.45,-0.45]; color = :red, linestyle = :dash)

mp4file = joinpath(output_dir, "perturbation_advection.mp4")
record(fig, mp4file, 1:Nt) do i
    @info "frame $i / $Nt"
    n[] = i
end

println("Saved $mp4file")
