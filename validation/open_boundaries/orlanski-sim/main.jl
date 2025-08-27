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
outdir = IOAndViz.attach_writers!(simulation, time_int=10minutes)

# using Statistics
# function print_progress(sim)
#     c = sim.model.tracers.c
#     mass = sum(interior(c))
#     @info "it=$(iteration(sim))  t=$(time(sim)/hours) h,  tracer mass≈$(round(mass, sigdigits=4))"
# end
# add_callback!(simulation, print_progress, IterationInterval(50))

run!(simulation)

# Make movie
IOAndViz.make_movie(outdir)
IOAndViz.make_tracer_heatmap(outdir)
IOAndViz.make_debug_tracer_and_u(outdir)

@info (time_ns() - wt) / 1e9