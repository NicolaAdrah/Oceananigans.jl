module IOAndViz

using Oceananigans
using Oceananigans.OutputWriters: JLD2Writer
using GLMakie
using Statistics
using Oceananigans.Units
using Oceananigans.AbstractOperations
using Oceananigans.Grids: znodes

function attach_writers!(simulation; output_dir = "validation/open_boundaries/hydro_bc_output", time_int=10minutes)
    if !isdir(output_dir); mkpath(output_dir); end

    model = simulation.model
    u, v, w = model.velocities
    η = model.free_surface.η
    c = model.tracers.c

    simulation.output_writers[:total_velocities] = JLD2Writer(model, (; u, v, w),
        schedule = TimeInterval(time_int),
        filename = joinpath(output_dir, "hydrostatic_open_boundaries.jld2"),
        overwrite_existing = true)

    simulation.output_writers[:free_surface] = JLD2Writer(model, (; η),
        schedule = TimeInterval(time_int),
        filename = joinpath(output_dir, "hydrostatic_open_boundaries_free_surface.jld2"),
        overwrite_existing = true)

    simulation.output_writers[:tracer] = JLD2Writer(model, (; c,),
        schedule = TimeInterval(time_int),
        filename = joinpath(output_dir, "tracer.jld2"),
        overwrite_existing = true)


    return output_dir
end

function make_movie(output_dir)
    u = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "u")
    v = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "v")
    w = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "w")
    η = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries_free_surface.jld2"), "η")
    c_tr = FieldTimeSeries(joinpath(output_dir, "tracer.jld2"), "c")  # --- NEW ---

    Nt = length(u.times)

    fig = Figure(size = (1000, 500))
    axu = Axis(fig[1, 1], title = "u-velocity")
    axv = Axis(fig[1, 2], title = "v-velocity")
    axw = Axis(fig[2, 1], title = "w-velocity")
    axη = Axis(fig[2, 2], title = "free surface")
    axc = Axis(fig[3, 1], title = "tracer bump")


    n = Observable(1)
    un = @lift(interior(u[$n], :, 1, 10))
    vn = @lift(interior(v[$n], :, 1, 10))
    wn = @lift(interior(w[$n], :, 1, 11))
    ηn = @lift(interior(η[$n], :, 1, 1))
    ctn = @lift(interior(c_tr[$n], :, 1, 1))
    lines!(axu, un); lines!(axv, vn); lines!(axw, wn); lines!(axη, ηn); lines!(axc, ctn);
    ylims!(axw, (-1e-5, 1e-5)); ylims!(axη, (-0.1, 0.2)); ylims!(axu, (-0.1, 0.1))

    record(fig, joinpath(output_dir, "cu_hydro_bc.mp4"), 1:Nt) do i
        @info "frame $i / $Nt"
        n[] = i
    end

    return joinpath(output_dir, "cu_hydro_bc.mp4")
end

function make_tracer_heatmap(output_dir)
    ts = FieldTimeSeries(joinpath(output_dir, "tracer.jld2"), "c")
    Nt = length(ts.times)

    fig = Figure(resolution = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "i-index", ylabel = "k-index", title = "Tracer c (slice at y=1)")
    n = Observable(1)
    cn = @lift(Array(interior(ts[$n], :, 1, :)))
    hm = heatmap!(ax, cn)
    Colorbar(fig[1, 2], hm)

    record(fig, joinpath(output_dir, "tracer.mp4"), 1:Nt) do i
        @info "tracer frame $i / $Nt"
        n[] = i
    end
    return joinpath(output_dir, "tracer.mp4")
end

# --- check if flow & tracer evolve ---
function make_debug_tracer_and_u(output_dir)
    u   = FieldTimeSeries(joinpath(output_dir, "hydrostatic_open_boundaries.jld2"), "u")
    cts = FieldTimeSeries(joinpath(output_dir, "tracer.jld2"), "c")

    Nt = min(length(u.times), length(cts.times))
    times = u.times[1:Nt] ./ 3600

    # Max |u| at y=1 (over x,z)
    umax = [maximum(abs.(Array(interior(u[i], :, 1, :)))) for i in 1:Nt]

    # Tracer center-of-mass in x averaged over z at y=1
    comx = Float64[]
    for i in 1:Nt
        Ci = Array(interior(cts[i], :, 1, :))  # (Nx, Nz)
        Ci[.!isfinite.(Ci)] .= 0
        # Sum over z -> profile across x
        cx = sum(Ci, dims=2)[:]
        if sum(cx) == 0
            push!(comx, NaN)
        else
            idx = 1:length(cx)
            push!(comx, sum(idx .* cx) / sum(cx))
        end
    end

    # Plot
    fig = Figure(resolution=(900, 400))
    ax1 = Axis(fig[1,1], xlabel="time [h]", ylabel="max|u| [m/s]", title="Flow speed diagnostic")
    lines!(ax1, times, umax)
    ax2 = Axis(fig[1,2], xlabel="time [h]", ylabel="x-COM [index]", title="Tracer center-of-mass (x)")
    lines!(ax2, times, comx)
    save(joinpath(output_dir, "debug_tracer_u.png"), fig)
    return joinpath(output_dir, "debug_tracer_u.png")
end

export attach_writers!, make_movie, make_tracer_heatmap, make_debug_tracer_and_u

end # module


