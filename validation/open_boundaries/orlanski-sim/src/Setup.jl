module Setup

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Grids: architecture
using Oceananigans.BoundaryConditions: Open, GradientBoundaryCondition
using Oceananigans.Advection
using Oceananigans.TurbulenceClosures
using ..BCs

# ---------- Grid + free surface ----------
function build_grid()
    RectilinearGrid(CPU(), size = (50, 10), x = (0, 500kilometers), topology = (Bounded, Flat, Bounded), z = (-10, 0))
end

function build_free_surface(typefreesurface = "Implicit")
    if     typefreesurface == "Implicit"
        ImplicitFreeSurface()
    elseif typefreesurface == "Explicit"
        ExplicitFreeSurface()
    else
        error("Unknown free surface type: $typefreesurface")
    end
end

# build_free_surface() = ImplicitFreeSurface()

# ---------- Boundary fields ----------
function allocate_boundary_fields(grid)
    uᵂ  = Field{Nothing, Nothing, Center}(grid)
    uᴱ  = Field{Nothing, Nothing, Center}(grid)
    u₁ᵂ = Field{Nothing, Nothing, Center}(grid)
    u₁ᴱ = Field{Nothing, Nothing, Center}(grid)
    fill!.((uᵂ, uᴱ, u₁ᵂ, u₁ᴱ), 0)
    return uᵂ, uᴱ, u₁ᵂ, u₁ᴱ
end

function build_u_bcs(uᵂ, uᴱ, u₁ᵂ, u₁ᴱ)
    u_west  = OpenBoundaryCondition(OrlanskiBoundary(uᵂ, u₁ᵂ))
    u_east  = OpenBoundaryCondition(OrlanskiBoundary(uᴱ, u₁ᴱ))
    FieldBoundaryConditions(west=u_west, east=u_east)
end

function build_c_bcs()
    c_west = OpenBoundaryCondition(GradientBoundaryCondition(0.0))
    c_east = OpenBoundaryCondition(GradientBoundaryCondition(0.0))
    FieldBoundaryConditions(west=c_west, east=c_east)
end

# ---------- Initial conditions ----------
const Rx = 250kilometers
const σ  =  50kilometers
const σc =  100kilometers

_gaussian_bump_η(x, z) = 0.1 * exp(-((x - Rx)^2 / σ^2))
_gaussian_bump_c(x, z) = exp(-((x - Rx)^2 / σc^2))

η₀(x, z) = _gaussian_bump_η(x, z)
c₀(x, z) = _gaussian_bump_c(x, z)

# ---------- Model ----------
function build_model()
    grid = build_grid()
    free_surface = build_free_surface("Implicit")
    uᵂ, uᴱ, u₁ᵂ, u₁ᴱ = allocate_boundary_fields(grid)
    u_bcs = build_u_bcs(uᵂ, uᴱ, u₁ᵂ, u₁ᴱ)
    c_bcs = build_c_bcs()
    model = HydrostaticFreeSurfaceModel(; grid,
        free_surface,
        # timestepper = :SplitRungeKutta3,
        boundary_conditions = (; u = u_bcs, c = c_bcs),
        tracers = :c,
        tracer_advection = WENO(),
        closure = ScalarDiffusivity(κ=(; c = 1e-6))
    )

    # set ICs
    set!(model; η = η₀, c = c₀)

    # initialize Orlanski history fields
    # BCs.initialize_boundary_history!(model, u₁ᵂ, u₁ᴱ)

    return (; model, grid, uᵂ, uᴱ, u₁ᵂ, u₁ᴱ)
end

export build_model, η₀

end # module