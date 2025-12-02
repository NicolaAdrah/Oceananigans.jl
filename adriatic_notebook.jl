### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 33e1eba9-2938-4a7d-bf6f-59aa6fe73e29
begin
	using Pkg
	Pkg.add(["PythonCall","CondaPkg"])
end

# ╔═╡ ed077140-2f5c-4f49-979b-6143b4fc9e8f
Pkg.add("CairoMakie")

# ╔═╡ 5d4abdb6-acaf-44b5-a869-8747129512c8
begin
    using CondaPkg

    # 1. Ensure there is a Python in the Conda environment, and pin it
    #    to a version TensorFlow actually supports (3.10 or 3.11 are safe).
    CondaPkg.add("python"; version=">=3.10,<3.12")

    # 2. Install TensorFlow via pip into this environment.
    #    The constraint >=2.17,<2.21 matches what your error was asking for,
    #    but keeps us on a stable release.
    CondaPkg.add_pip("tensorflow"; version=">=2.17,<2.21")

    # 3. Install the Copernicus Marine Python toolbox (the PyPI package is
    #    called 'copernicusmarine').
    CondaPkg.add_pip("copernicusmarine")

    # 4. Force resolution in case CondaPkg.toml was edited outside Julia.
    CondaPkg.resolve(force=true)

    # Optional: show what is installed for sanity
    CondaPkg.status()
end	

# ╔═╡ 4ecbdf3a-c7ee-11f0-0716-81cbedbb1b74
begin
	using CairoMakie
	using Oceananigans
	using Oceananigans.BoundaryConditions: PerturbationAdvection
	using Oceananigans.Advection: WENO
	using Oceananigans.OutputWriters: JLD2Writer, TimeInterval, Checkpointer
	using Oceananigans.Grids: Face, Center, nodes, MutableVerticalDiscretization
	using Oceananigans.Fields: Field
	using Oceananigans: Simulation, run!, set!, FieldTimeSeries, PrescribedVelocityFields, architecture, launch!
	using Oceananigans: fill_halo_regions!
	using ClimaOcean
	using Oceananigans.Units
	using Printf
	using NCDatasets
	using Dates
	using CopernicusMarine
end

# ╔═╡ Cell order:
# ╠═33e1eba9-2938-4a7d-bf6f-59aa6fe73e29
# ╠═5d4abdb6-acaf-44b5-a869-8747129512c8
# ╠═ed077140-2f5c-4f49-979b-6143b4fc9e8f
# ╠═4ecbdf3a-c7ee-11f0-0716-81cbedbb1b74
