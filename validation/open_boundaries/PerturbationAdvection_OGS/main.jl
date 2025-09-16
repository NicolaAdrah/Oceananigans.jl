using Oceananigans
using Oceananigans.BoundaryConditions: PerturbationAdvection
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

uB_east = Field{Nothing, Nothing, Center}(grid);
uB_west = Field{Nothing, Nothing, Center}(grid);

u_bcs = FieldBoundaryConditions(
    east = OpenBoundaryCondition(uB_east; scheme = PerturbationAdvection()),
    west = OpenBoundaryCondition(uB_west; scheme = PerturbationAdvection())
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
simulation.output_writers[:free_surface] = JLD2Writer(model, (; η,),
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

η = FieldTimeSeries(joinpath(output_dir, "free_surface.jld2"), "η")

Nt   = length(η.times)

fig = Figure(resolution = (1000, 500))
axη = Axis(fig[1, 1], title = "free surface")

n = Observable(1)
ηn = @lift(interior(η[$n], :, 1, 1))

lines!(axη, ηn)
ylims!(axη, (-0.1, 0.2))

mp4file = joinpath(output_dir, "free_surface.mp4")
record(fig, mp4file, 1:Nt) do i
    @info "frame $i / $Nt"
    n[] = i
end

println("Saved $mp4file")
