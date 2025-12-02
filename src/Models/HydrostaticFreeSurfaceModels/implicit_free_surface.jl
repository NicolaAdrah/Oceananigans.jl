using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: device
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Solvers: solve!
using Oceananigans.Utils: prettysummary
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Adapt

struct ImplicitFreeSurface{E, G, B, I, M, S} <: AbstractFreeSurface{E, G}
    η :: E
    gravitational_acceleration :: G
    barotropic_volume_flux :: B
    implicit_step_solver :: I
    solver_method :: M
    solver_settings :: S
end

Base.show(io::IO, fs::ImplicitFreeSurface) =
    isnothing(fs.η) ?
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ solver_method: ", fs.solver_method, "\n", # TODO: implement summary for solvers
              "└─ settings: ", isempty(fs.solver_settings) ? "Default" : fs.solver_settings) :
    print(io, "ImplicitFreeSurface with ", fs.solver_method, "\n",
              "├─ grid: ", summary(fs.η.grid), "\n",
              "├─ η: ", summary(fs.η), "\n",
              "├─ gravitational_acceleration: ", prettysummary(fs.gravitational_acceleration), "\n",
              "├─ implicit_step_solver: ", nameof(typeof(fs.implicit_step_solver)), "\n", # TODO: implement summary for solvers
              "└─ settings: ", fs.solver_settings)

"""
    ImplicitFreeSurface(; solver_method = :Default,
                        gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                        solver_settings...)

Return an implicit free-surface solver. The implicit free-surface equation is

```math
\\left [ 𝛁_h ⋅ (H 𝛁_h) - \\frac{1}{g Δt^2} \\right ] η^{n+1} = \\frac{𝛁_h ⋅ 𝐐_⋆}{g Δt} - \\frac{η^{n}}{g Δt^2} ,
```

where ``η^n`` is the free-surface elevation at the ``n``-th time step, ``H`` is depth, ``g`` is
the gravitational acceleration, ``Δt`` is the time step, ``𝛁_h`` is the horizontal gradient operator,
and ``𝐐_⋆`` is the barotropic volume flux associated with the predictor velocity field ``𝐮_⋆``, i.e.,

```math
𝐐_⋆ = \\int_{-H}^0 𝐮_⋆ \\, 𝖽 z ,
```

where

```math
𝐮_⋆ = 𝐮^n + \\int_{t_n}^{t_{n+1}} 𝐆ᵤ \\, 𝖽t .
```

This equation can be solved, in general, using the [`ConjugateGradientSolver`](@ref) but
other solvers can be invoked in special cases.

If ``H`` is constant, we divide through out to obtain

```math
\\left ( ∇^2_h - \\frac{1}{g H Δt^2} \\right ) η^{n+1}  = \\frac{1}{g H Δt} \\left ( 𝛁_h ⋅ 𝐐_⋆ - \\frac{η^{n}}{Δt} \\right ) .
```

Thus, for constant ``H`` and on grids with regular spacing in ``x`` and ``y`` directions, the free
surface can be obtained using the [`FFTBasedPoissonSolver`](@ref).

`solver_method` can be either of:
* `:FastFourierTransform` for [`FFTBasedPoissonSolver`](@ref)
* `:PreconditionedConjugateGradient` for [`ConjugateGradientSolver`](@ref)

By default, if the grid has regular spacing in the horizontal directions then the `:FastFourierTransform` is chosen,
otherwise the `:PreconditionedConjugateGradient`.
"""
function ImplicitFreeSurface(;
    solver_method = :Default,
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    solver_settings...)

    return ImplicitFreeSurface(nothing, gravitational_acceleration, nothing, nothing, solver_method, solver_settings)
end

Adapt.adapt_structure(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(Adapt.adapt(to, free_surface.η), free_surface.gravitational_acceleration,
                        nothing, nothing, nothing, nothing)

on_architecture(to, free_surface::ImplicitFreeSurface) =
    ImplicitFreeSurface(on_architecture(to, free_surface.η),
                        on_architecture(to, free_surface.gravitational_acceleration),
                        on_architecture(to, free_surface.barotropic_volume_flux),
                        on_architecture(to, free_surface.implicit_step_solver),
                        on_architecture(to, free_surface.solver_methods),
                        on_architecture(to, free_surface.solver_settings))

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::ImplicitFreeSurface{Nothing}, velocities, grid)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    gravitational_acceleration = convert(eltype(grid), free_surface.gravitational_acceleration)

    # Initialize barotropic volume fluxes
    barotropic_x_volume_flux = Field{Face, Center, Nothing}(grid)
    barotropic_y_volume_flux = Field{Center, Face, Nothing}(grid)
    barotropic_volume_flux = (u=barotropic_x_volume_flux, v=barotropic_y_volume_flux)

    user_solver_method = free_surface.solver_method # could be = :Default
    solver = build_implicit_step_solver(Val(user_solver_method), grid, free_surface.solver_settings, gravitational_acceleration)
    solver_method = nameof(typeof(solver))

    return ImplicitFreeSurface(η,
                               gravitational_acceleration,
                               barotropic_volume_flux,
                               solver,
                               solver_method,
                               free_surface.solver_settings)
end

build_implicit_step_solver(::Val{:Default}, grid::XYRegularRG, settings, gravitational_acceleration) =
    build_implicit_step_solver(Val(:FastFourierTransform), grid, settings, gravitational_acceleration)

build_implicit_step_solver(::Val{:Default}, grid, settings, gravitational_acceleration) =
    build_implicit_step_solver(Val(:PreconditionedConjugateGradient), grid, settings, gravitational_acceleration)

@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::ImplicitFreeSurface) = 0

"""
Implicitly step forward η.
"""
function step_free_surface!(free_surface::ImplicitFreeSurface, model, timestepper, Δt)
    η      = free_surface.η
    g      = free_surface.gravitational_acceleration
    rhs    = free_surface.implicit_step_solver.right_hand_side
    ∫ᶻQ    = free_surface.barotropic_volume_flux
    solver = free_surface.implicit_step_solver
    arch   = model.architecture

    fill_halo_regions!(model.velocities, model.clock, fields(model))

    # Compute right hand side of implicit free surface equation
    @apply_regionally local_compute_integrated_volume_flux!(∫ᶻQ, model.velocities, arch)
    fill_halo_regions!(∫ᶻQ)

    compute_implicit_free_surface_right_hand_side!(rhs, solver, g, Δt, ∫ᶻQ, η)

    # TODO NICOLA
    # To check the data passed to the solver is finite
    @debug begin
        @warn all(isfinite, parent(model.velocities.v.data))
        @warn all(isfinite, parent(model.velocities.u.data)) 
    end
    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    solve!(η, solver, rhs, g, Δt)

    @debug "Implicit step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η)

    return nothing
end

function step_free_surface!(free_surface::ImplicitFreeSurface, model, timestepper::SplitRungeKutta3TimeStepper, Δt)
    parent(free_surface.η) .= parent(timestepper.Ψ⁻.η)
    step_free_surface!(free_surface, model, nothing, Δt)
    return nothing
end

function local_compute_integrated_volume_flux!(∫ᶻQ, velocities, arch)

    foreach(mask_immersed_field!, velocities)

    # Compute barotropic volume flux. Blocking.
    compute_vertically_integrated_volume_flux!(∫ᶻQ, velocities)

    return nothing
end
