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

# Tracer boundary conditions: zero-gradient (open)
function build_c_bcs()
    c_west = OpenBoundaryCondition(GradientBoundaryCondition(0.0))
    c_east = OpenBoundaryCondition(GradientBoundaryCondition(0.0))
    FieldBoundaryConditions(west=c_west, east=c_east)
end

# ---------- Initial conditions ----------
const Rx = 250kilometers
const σ  =  50kilometers
const σc =  10kilometers

_gaussian_bump(x, z) = 0.1 * exp(-((x - Rx)^2 / σ^2))
η₀(x, z) = _gaussian_bump(x, z)

# Tracer initial condition: Gaussian centered at Rx with narrower width
c0(x, z) = exp(-((x - 0.3*Rx)^2 / σc^2)) * exp(-(z+5)^2 / (2*2^2))

# ---------- Model ----------
function build_model()
    grid = build_grid()
    free_surface = build_free_surface("Implicit")
    uᵂ, uᴱ, u₁ᵂ, u₁ᴱ = allocate_boundary_fields(grid)
    u_bcs = build_u_bcs(uᵂ, uᴱ, u₁ᵂ, u₁ᴱ)
    c_bcs = build_c_bcs()
    closure = ScalarDiffusivity(κ=(; c = 1e-6))
    model = HydrostaticFreeSurfaceModel(; grid,
        free_surface,
        tracers = (:c,),
        tracer_advection = WENO(),
        boundary_conditions = (; u = u_bcs, c = c_bcs),
        closure = closure
    )

    # set ICs
    set!(model; η = η₀, c = c0)

    # initialize Orlanski history fields
    BCs.initialize_boundary_history!(model, u₁ᵂ, u₁ᴱ)

    return (; model, grid, uᵂ, uᴱ, u₁ᵂ, u₁ᴱ)
end

export build_model, η₀, c0

end # module