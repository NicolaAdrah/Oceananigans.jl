# TODO Nicola: "Add evolution plots for velocity and free surface"
using GLMakie
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: nodes, Center, Face
using Oceananigans.Fields: interior

# Output directory and file paths
output_dir = "validation/open_boundaries/PerturbationAdvection_OGS/output"
η_file = joinpath(output_dir, "free_surface.jld2")

# Load data
η_ts = FieldTimeSeries(η_file, "η")
xc = nodes(η_ts.grid, Center(), Center(), Center())[1]
times = η_ts.times

# Grid for Hovmöller
# η_ts[n] is a Field, interior(η_ts[n], :, 1, 1) is a Vector
Nt = length(times)
Nx = length(xc)
η_data = zeros(Nx, Nt)

for n in 1:Nt
    η_data[:, n] .= interior(η_ts[n], :, 1, 1)
end

# --- Hovmöller Diagram ---
fig_hov = Figure(size = (800, 400), fontsize = 18)
ax = Axis(fig_hov[1, 1],
    xlabel = "x (km)",
    ylabel = "Time (days)",
    title = "Hovmöller Diagram of Free Surface (η)")

hm = heatmap!(ax, xc / 1kilometer, times / 1day, η_data, colormap = :balance)
Colorbar(fig_hov[1, 2], hm, label = "η (m)")

save(joinpath(output_dir, "hovmoller_eta.png"), fig_hov)

# --- Snapshot Panels ---
# Select 4 representative times
indices = round.(Int, range(1, Nt, length=4))
fig_snap = Figure(size = (800, 600), fontsize = 18)

for (i, n) in enumerate(indices)
    ax = Axis(fig_snap[i, 1],
        ylabel = "η (m)",
        title = "t = $(round(times[n]/1day, digits=1)) days")
    
    if i < 4
        hidexdecorations!(ax, grid = false)
    else
        ax.xlabel = "x (km)"
    end
    
    lines!(ax, xc / 1kilometer, interior(η_ts[n], :, 1, 1), color = :black, linewidth = 2)
    ylims!(ax, (-0.1, 0.2))
end

save(joinpath(output_dir, "snapshots_eta.png"), fig_snap)

# --- Evolution Plots ---
# TODO Nicola: "Add time-evolution plots for peak free surface and velocity"
u_file = joinpath(output_dir, "velocities.jld2")
u_ts = FieldTimeSeries(u_file, "u")
ux = nodes(u_ts.grid, Face(), Center(), Center())[1]

η_max = [maximum(abs, interior(η_ts[n], :, 1, 1)) for n in 1:Nt]
u_max = [maximum(abs, interior(u_ts[n], :, 1, 1)) for n in 1:Nt]

fig_evol = Figure(size = (800, 600), fontsize = 18)
ax_eta = Axis(fig_evol[1, 1], 
    ylabel = "|η| max (m)", 
    title = "Evolution of Peak Free Surface")
ax_u = Axis(fig_evol[2, 1], 
    xlabel = "Time (days)", 
    ylabel = "|u| max (m/s)", 
    title = "Evolution of Peak Velocity")

lines!(ax_eta, times / 1day, η_max, color = :blue, linewidth = 2)
lines!(ax_u, times / 1day, u_max, color = :red, linewidth = 2)

save(joinpath(output_dir, "evolution_v_eta.png"), fig_evol)

println("Thesis plots saved to $output_dir")
