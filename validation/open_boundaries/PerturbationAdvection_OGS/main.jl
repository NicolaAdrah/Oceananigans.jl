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

uB_east = Field{Nothing, Nothing, Center}(grid);  set!(uB_east, 1.0)   # 1 m/s at east boundary
uB_west = Field{Nothing, Nothing, Center}(grid);  set!(uB_west, 1.0)   # 1 m/s at west boundary

u_bcs = FieldBoundaryConditions(
    east = PerturbationAdvectionOpenBoundaryCondition(uB_east),
    west = PerturbationAdvectionOpenBoundaryCondition(uB_west)
)

model = HydrostaticFreeSurfaceModel(;
    grid,
    velocities        = PrescribedVelocityFields(u = 1.0),  # constant eastward flow
    tracers           = (:c,),
    tracer_advection  = WENO(),
)

x0 = 250kilometers; σx = 50kilometers
ϕ(x, z)    = exp(-((x - x0)^2) / (2σx^2))

set!(model; c = ϕ)

simulation = Simulation(model; Δt = 5minutes, stop_time = 5days)

c = model.tracers.c
simulation.output_writers[:tracer] = JLD2Writer(model, (; c,),
    schedule            = TimeInterval(time_int),
    filename            = joinpath(output_dir, "tracer.jld2"),
    overwrite_existing  = true
)

@info "Running..." 
run!(simulation)
@info "Done."

using GLMakie
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: nodes
using Oceananigans.Fields: interior

c_tr = FieldTimeSeries(joinpath(output_dir, "tracer.jld2"), "c")
Nt   = length(c_tr.times)

xC, _, zC = nodes(model.grid, Center(), Center(), Center())
Nx = length(xC)

n      = Observable(1)
title_obs = @lift("Tracer c — t = $(round(c_tr.times[$n]/hour, digits=2)) h")

# 2D tracer slice (x–z, at y=1 since it's Flat)
c_slice = @lift(Array(interior(c_tr[$n]))[:, 1, :])

c_line  = @lift(interior(c_tr[$n], :, 1, 1))

fig = Figure(resolution = (1000, 800))

ax1 = Axis(fig[1, 1],
           xlabel = "x (km)", ylabel = "z (m)",
           title  = title_obs)
hm = heatmap!(ax1, xC ./ kilometer, zC, c_slice)
Colorbar(fig[1, 2], hm, label = "c")

ax2 = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "c",
           title = "Tracer bump at y=1, z=1")
lines!(ax2, xC ./ kilometer, c_line)

mp4file = joinpath(output_dir, "cu_hydro_bc_tracer.mp4")
record(fig, mp4file, 1:Nt; framerate = 12) do i
    @info "frame $i / $Nt"
    n[] = i
end

println("Saved $mp4file")
