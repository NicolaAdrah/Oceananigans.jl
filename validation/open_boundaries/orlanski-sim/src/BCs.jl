module BCs

using Oceananigans
using Oceananigans.Grids
using Oceananigans.BoundaryConditions: Open
import Oceananigans.BoundaryConditions: getbc, update_boundary_condition!
using Oceananigans.Grids: architecture
using Oceananigans.Utils
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators

"""
OrlanskiBoundary holds (current, history) boundary fields.
U and U1 are Field types bound to the model's grid.
"""
struct OrlanskiBoundary{U, U1}
    uᴮ :: U   # ghost-cell field at the boundary
    u1 :: U1  # last-step interior snapshot for phase speed
end

# For convenience when annotating a BoundaryCondition using Open
const OrlanskiBoundaryCondition = BoundaryCondition{<:Open, <:OrlanskiBoundary}

# =============
# GETBC (read)
# =============
# West/east u-BCs read from the ghost cell in the x-direction.
# Oceananigans calls `getbc` to retrieve the boundary value during halo filling.
@inline getbc(bc::OrlanskiBoundaryCondition, j, k, args...) = bc.condition.uᴮ[1, j, k]

# ======================
# WEST/EAST UPDATE KERNELS
# ======================
@kernel function _update_west_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
    j, k = @index(Global, NTuple)
    @inbounds begin
        Δuₓ = uⁿ⁺¹[3, j, k] - uⁿ⁺¹[2, j, k]
        Δuₜ = uⁿ⁺¹[2, j, k] - u₁[2, j, k]
        max_speed = - sqrt(9.80665 * grid.Lz)
        speed = if abs(Δuₓ * Δt) > 1e-20
            - (Δuₜ * Δxᶠᶜᶜ(1, j, k, grid)) / (Δuₓ * Δt)
        else
            0.0
        end
        if isnan(speed); speed = 0.0; end
        speed = clamp(speed, 0.0, max_speed)
        c = speed * Δt / Δxᶠᶜᶜ(1, j, k, grid)
        uᴮ[1, j, k] = (uᴮ[1, j, k] - c * uⁿ⁺¹[2, j, k]) / (1 - c)
    end
end

@kernel function _update_east_bc(uᴮ, grid, uⁿ⁺¹, u₁, Δt)
    j, k = @index(Global, NTuple)
    Nx   = size(grid, 1)
    @inbounds begin
        Δuₓ = uⁿ⁺¹[Nx,   j, k] - uⁿ⁺¹[Nx-1, j, k]
        Δuₜ = uⁿ⁺¹[Nx,   j, k] - u₁[Nx,     j, k]
        max_speed = - sqrt(9.80665 * grid.Lz)
        speed = if abs(Δuₓ * Δt) > 1e-20
            - (Δuₜ * Δxᶠᶜᶜ(Nx+1, j, k, grid)) / (Δuₓ * Δt)
        else
            0.0
        end
        if isnan(speed); speed = 0.0; end
        speed = clamp(speed, 0.0, max_speed)
        c = speed * Δt / Δxᶠᶜᶜ(Nx+1, j, k, grid)
        uᴮ[Nx+1, j, k] = (uᴮ[Nx+1, j, k] - c * uⁿ⁺¹[Nx, j, k]) / (1 - c)
    end
end

# ======================
# UPDATE BOUNDARIES (hooks)
# ======================
function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:west}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    launch!(architecture(grid), grid, :yz,  _update_west_bc, uᴮ, grid, u, u₁, Δt)
    return nothing
end

function update_boundary_condition!(bc::OrlanskiBoundaryCondition, ::Val{:east}, u, model)
    uᴮ = bc.condition.uᴮ
    u₁ = bc.condition.u1
    u  = model.velocities.u
    grid = model.grid
    Δt = ifelse(model.clock.last_Δt > 1e10, zero(grid), model.clock.last_Δt)
    launch!(architecture(grid), grid, :yz,  _update_east_bc, uᴮ, grid, u, u₁, Δt)
    return nothing
end

function update_boundary_condition!(bcs::FieldBoundaryConditions, u, model)
    update_boundary_condition!(bcs.west, Val(:west), u, model)
    update_boundary_condition!(bcs.east, Val(:east), u, model)
    impose_volume_conservation!(bcs.west, bcs.east, model)
    return nothing
end

# ======================
# VOLUME CONSERVATION
# ======================
@kernel function _impose_volume_conservation!(uw, ue, grid)
    j = @index(Global, Linear)
    Uc = 0.0
    for k in 1:grid.Nz
        Uc += @inbounds uw[1, j, k] * Δzᶠᶜᶜ(1, j, k, grid) + ue[1, j, k] * Δzᶠᶜᶜ(grid.Nx+1, j, k, grid)
    end
    for k in 1:grid.Nz
        @inbounds ue[1, j, k] -= Uc / grid.Lz
        @inbounds uw[1, j, k] -= Uc / grid.Lz
    end
end

impose_volume_conservation!(u_west, u_east, model) = nothing

function impose_volume_conservation!(u_west::OrlanskiBoundaryCondition, u_east::OrlanskiBoundaryCondition, model)
    grid = model.grid
    launch!(architecture(grid), grid, (grid.Ny, ),  _impose_volume_conservation!, u_west.condition.uᴮ, u_east.condition.uᴮ, grid)
    return nothing
end

# ======================
# HISTORY INIT HELPERS
# ======================
@kernel function _initialize_history_field!(u₁, u)
    i, j, k = @index(Global, NTuple)
    if i >= 2 && i <= size(u, 1)
        @inbounds u₁[i, j, k] = u[i, j, k]
    end
end

function initialize_boundary_history!(model, u₁ᵂ, u₁ᴱ)
    u = model.velocities.u
    grid = model.grid
    launch!(architecture(grid), grid, :xyz, _initialize_history_field!, u₁ᵂ, u)
    launch!(architecture(grid), grid, :xyz, _initialize_history_field!, u₁ᴱ, u)
    return nothing
end

export OrlanskiBoundary, OrlanskiBoundaryCondition,
       initialize_boundary_history!, impose_volume_conservation!

end # module