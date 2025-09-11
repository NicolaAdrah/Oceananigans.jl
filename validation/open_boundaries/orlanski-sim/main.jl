# main.jl

include("src/BCs.jl")
include("src/Setup.jl")
include("src/IOandViz.jl")

using .BCs
using .Setup
using .IOAndViz
using Oceananigans
using Oceananigans.Units

wt = time_ns()

# Build the model & simulation
state = Setup.build_model()
model = state.model

simulation = Simulation(model; Δt = 0.1minutes, stop_time = 5days)

# Attach writers and run
outdir = IOAndViz.attach_writers_velocities!(simulation, time_int=10minutes)
# outdir = IOAndViz.attach_writers_tracer!(simulation, time_int=10minutes)

run!(simulation)

# Make movie
IOAndViz.make_movie_velocities(outdir)
# IOAndViz.make_movie_tracer(outdir)
# IOAndViz.make_tracer_heatmap(outdir)
# IOAndViz.make_debug_tracer_and_u(outdir)

@info (time_ns() - wt) / 1e9