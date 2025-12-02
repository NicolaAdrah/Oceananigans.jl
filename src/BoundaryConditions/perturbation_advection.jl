using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans: defaults

# struct PerturbationAdvection{FT}
#     inflow_timescale :: FT
#    outflow_timescale :: FT
# end

"""
    PerturbationAdvection(FT = defaults.FloatType;
                          outflow_timescale = Inf,
                          inflow_timescale = 0)

Create a `PerturbationAdvection` scheme to be used with an `OpenBoundaryCondition`.
This scheme will nudge the boundary velocity to the OpenBoundaryCondition's exterior value `val`,
using a time-scale `inflow_timescale` for inflow and `outflow_timescale` for outflow.

For cases where we assume that the internal flow is a small perturbation from
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components.

We begin with the equation governing the fluid in the interior:
    ∂ₜu + u⋅∇u = −∇P + F,
and note that on the boundary the pressure gradient is zero.
We can then assume that the flow composes of mean (U⃗) and pertubation (u⃗′) components,
and considering the x-component of velocity, we can rewrite the equation as
    ∂ₜu₁ = -u₁∂₁u - u₂∂₂u₁ - u₃∂₃u₁ + F₁ ≈ - U₁∂₁u₁′ - U₂∂₂u₁′ - U₃∂₃u₁′ + F.

Simplify by assuming that U⃗ = Ux̂, an then take a numerical step to find u₁.

When the boundaries are filled the interior is at time tₙ₊₁ so we can take
a backwards euler step (in the case that the mean flow is boundary normal) on a right boundary:
    (Uⁿ⁺¹ - Uⁿ) / Δt + (u′ⁿ⁺¹ - u′ⁿ) / Δt = - Uⁿ⁺¹ (u′ⁿ⁺¹ᵢ - u′ⁿ⁺¹ᵢ₋₁) / Δx + Fᵤ.

This can not be solved for general forcing, but if we assume the dominant forcing is
relaxation to the mean velocity (i.e. u′→0) then Fᵤ = -u′ / τ then we can find u′ⁿ⁺¹:
    u′ⁿ⁺¹ = (uⁿ + Ũu′ⁿ⁺¹ᵢ₋₁ - Uⁿ⁺¹) / (1 + Ũ + Δt/τ),

where Ũ = U Δt / Δx, then uⁿ⁺¹ is:
    uⁿ⁺¹ = (uᵢⁿ + Ũuᵢ₋₁ⁿ⁺¹ + Uⁿ⁺¹τ̃) / (1 + τ̃ + Ũ)

where τ̃ = Δt/τ.

The same operation can be repeated for left boundaries.

The relaxation timescale ``τ`` can be set to different values depending on whether
``U`` points in or out of the domain (`inflow_timescale`/`outflow_timescale`). Since the
scheme is only valid when the flow is directed out of the domain the boundary condition
falls back to relaxation to the prescribed value. By default this happens instantly but
if the direction varies this may not be preferable. It is beneficial to relax the outflow
(i.e. non-zero `outflow_timescale`) to reduce the shock when the flow changes direction
to point into the domain.

The ideal value of the timescales probably depend on the grid spacing and details of the
boundary flow.
"""
mutable struct PerturbationAdvection{FT}
    inflow_timescale  :: FT
    outflow_timescale :: FT
    previous          :: Union{Nothing, AbstractArray}
end

function PerturbationAdvection(FT = defaults.FloatType;
                               outflow_timescale = Inf,
                               inflow_timescale  = 0)
    inflow_timescale  = convert(FT, inflow_timescale)
    outflow_timescale = convert(FT, outflow_timescale)
    return PerturbationAdvection{FT}(inflow_timescale, outflow_timescale, nothing)
end

Adapt.adapt_structure(to, pe::PerturbationAdvection) =
    PerturbationAdvection(adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale),
                          adapt(to, pe.previous))

const X_AXIS = :x
const Y_AXIS = :y
const Z_AXIS = :z

@inline function axis_from_indices(bry_idx, adj_idx)
    if bry_idx[1] != adj_idx[1]
        return X_AXIS
    elseif bry_idx[2] != adj_idx[2]
        return Y_AXIS
    else
        return Z_AXIS
    end
end

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

@inline function step_right_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                                      grid, u, clock, model_fields, loc, ΔX)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), 0, Δt)

    uᵢⁿ     = @inbounds getindex(u, iᴮ, jᴮ, kᴮ)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, iᴬ, jᴬ, kᴬ)
    normal_axis = axis_from_indices(boundary_indices, boundary_adjacent_indices)
    Ũ = dimensionless_phase_speed(u, uᵢⁿ, boundary_adjacent_indices, boundary_interior_indices,
                                   grid, loc, Δt, ΔX, normal_axis, 1)
    if Ũ isa AbstractFloat && isnan(Ũ)
        Ũ = zero(Ũ)
    end
    max_speed = convert(typeof(Ũ), 0.29)
    U = clamp(Ũ, zero(Ũ), max_speed)
    # pa = bc.classification.scheme
    # ū = getbc(bc, l, m, grid, clock, model_fields)
    # τ = isnothing(pa) ? Inf : (ū ≥ 0 ? pa.outflow_timescale : pa.inflow_timescale)

    # if τ == 0
        # uᵢⁿ⁺¹ = ū
    # else
        # τ̃ = isinf(τ) || Δt == 0 ? zero(Δt) : Δt / τ
        # numer = uᵢⁿ + U * uᵢ₋₁ⁿ⁺¹ + τ̃ * ū
        numer = uᵢⁿ + U * uᵢ₋₁ⁿ⁺¹
        # denom = 1 + τ̃ + U
        denom = 1 + U
        uᵢⁿ⁺¹ = denom == 0 ? uᵢⁿ : numer / denom
    # end
    @inbounds setindex!(u, uᵢⁿ⁺¹, iᴮ, jᴮ, kᴮ)

    return nothing
end

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    boundary_interior_indices = (i-2, j, k)

    Δx = Δxᶠᶜᶜ(i, j, k, grid)

    step_right_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                         grid, u, clock, model_fields, loc, Δx)

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    boundary_interior_indices = (3, j, k)
    
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                        grid, u, clock, model_fields, loc, Δx)

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, u, bc::PAOBC, loc::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    boundary_interior_indices = (i, j-2, k)

    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                         grid, u, clock, model_fields, loc, Δy)

    return nothing
end

const ORLANSKI_EPS = 1e-20
const RADIATION_2D = false 
const IMPLICIT_NUDGING = true
# let this function return dVdt and dVde to be used outside for the nudging.
@inline function orlanski_south_boundary!(
    u::AbstractArray,      # current time level (like kout)
    u_old::AbstractArray,  # previous time level (like know)
    i::Int, k::Int,
)
    Nx = size(u, 1)
    Ny = size(u, 2)

    j_halo = 1
    j_adj   = 2
    j_intr_1    = j_adj + 1  # = 3
    j_intr_2    = j_adj + 2  # = 4

    # Require enough neighbors in i and j
    # if i < 2 || i > Nx - 1 || j_intr_2 > Ny
    #     @warn "not enough grid points to apply Orlanski BC at (i=$i, j=$j_halo, k=$k). Skipping."
    #     return zero(eltype(u)), zero(eltype(u))  # no-op, no nudging info
    # end

    dVdt = zero(eltype(u))
    dVde = zero(eltype(u))
    @inbounds begin
        # ROMS:
        # grad(i, j_adj)   = vbar(i, j_adj,   know) - vbar(i-1, j_adj,   know)
        # grad(i, j_adj+1) = vbar(i, j_adj+1, know) - vbar(i-1, j_adj+1, know)
        grad_j_adj_I_bw      = u_old[i,   j_adj, k] - u_old[i-1, j_adj, k]
        grad_j_adj_I_fw  = u_old[i+1, j_adj, k] - u_old[i,   j_adj, k]

        grad_j_intr1_I_bw        = u_old[i,   j_intr_1,  k] - u_old[i-1, j_intr_1,  k]
        grad_j_intr1_I_fw    = u_old[i+1, j_intr_1,  k] - u_old[i,   j_intr_1,  k]

        # ROMS:
        #   dVdt = vbar(i, j_adj+1, know) - vbar(i, j_adj+1, kout)
        #   dVde = vbar(i, j_adj+1, kout) - vbar(i, j_adj+2, kout)
        v_know_j_intr_1  = u_old[i, j_intr_1, k]
        v_kout_j_intr_1  = u[i,     j_intr_1, k]
        v_kout_j_intr_2  = u[i,     j_intr_2, k]

        dVdt = v_know_j_intr_1 - v_kout_j_intr_1
        dVde = v_kout_j_intr_1 - v_kout_j_intr_2

        if dVdt * dVde < 0
            dVdt = zero(dVdt)
        end

        # ROMS:
        # IF (dVdt*(grad(i,j_adj+1) + grad(i+1,j_adj+1))) > 0 THEN
        #   dVdx = grad(i,j_adj+1)
        # ELSE
        #   dVdx = grad(i+1,j_adj+1)
        s    = dVdt * (grad_j_intr1_I_bw + grad_j_intr1_I_fw)
        dVdx = s > 0 ? grad_j_intr1_I_bw : grad_j_intr1_I_fw

        # cff = MAX(dVdx*dVdx + dVde*dVde, eps)
        cff = dVdx * dVdx + dVde * dVde
        if cff < ORLANSKI_EPS
            cff = ORLANSKI_EPS
        end

        # In ROMS:
        #   Cx = MIN(cff, MAX(dVdt*dVdx, -cff))   (for RADIATION_2D)
        #   Ce = dVdt * dVde
        
        Cx = dVdt * dVdx

        # TODO put keyword to accept @static condition in the function call.
        @static if RADIATION_2D
        # println("using RADIATION_2D")
            Cx = min(cff, max(dVdt * dVdx, -cff))
        else
            Cx = 0.0
        end

        Ce = dVdt * dVde

        v_know_j_adj = u_old[i, j_adj, k]
        v_kout_j_intr_1   = v_kout_j_intr_1  # already defined

        denom = cff + Ce

        # To be as close as possible to ROMS, we *do not* add extra guards here.
        # If denom ≈ 0, the scheme is marginal; with your Δt and grid this
        # should not systematically happen.
        v_boundary = (  cff * v_know_j_adj
                       + Ce  * v_kout_j_intr_1
                       - max(Cx, zero(Cx)) * grad_j_adj_I_bw
                       - min(Cx, zero(Cx)) * grad_j_adj_I_fw
                      ) / denom

        u[i, j_halo, k] = v_boundary
    end

    return dVdt, dVde
end

@inline function step_left_boundary!(bc::PAOBC,
                                     l, m,
                                     boundary_indices,
                                     boundary_adjacent_indices,
                                     boundary_interior_indices,
                                     grid, u, clock, model_fields, loc, ΔX)

    iB, jB, kB = boundary_indices
    normal_axis = axis_from_indices(boundary_indices, boundary_adjacent_indices)

    if normal_axis === Y_AXIS && jB == 1
        pa = bc.classification.scheme  # ::PerturbationAdvection

        # 1. Ensure we have a previous time level
        if pa.previous === nothing
            pa.previous = similar(u)
            copyto!(pa.previous, u)
        end
        u_old = pa.previous

        # 2. Pure radiation update at (iB, kB), and get dVdt, dVde
        dVdt, dVde = orlanski_south_boundary!(u, u_old, iB, kB)

        # 3. Check for nudging
        inflow_τ  = pa.inflow_timescale
        outflow_τ = pa.outflow_timescale

        if isinf(inflow_τ) && isinf(outflow_τ)
            return nothing
        end

        # 4. Decide inflow vs outflow based on the sign of dVdt*dVde
        σ  = dVdt * dVde
        τ  = σ < 0 ? inflow_τ : outflow_τ

        # If τ<=0 or infinite, no nudging
        if τ <= 0 || isinf(τ)
            return nothing
        end

        # 5. External boundary value this should looks like (ROMS BOUNDARY(ng)%vbar_south(i))
        ū = getbc(bc, l, m, grid, clock, model_fields)

        # 6. Time step
        Δt = clock.last_stage_Δt
        Δt = ifelse(isinf(Δt), zero(Δt), Δt)

        # 7. nudging
        @static if IMPLICIT_NUDGING
            # "Implicit" nudging like ROMS' IMPLICIT_NUDGING:
            #   phi = Δt / (τ + Δt)
            #   u_new = (1 - phi) * u_rad + phi * ū
            if Δt > 0
                φ   = Δt / (τ + Δt)
                @inbounds begin
                    u_rad = u[iB, jB, kB]
                    u_new = (one(φ) - φ) * u_rad + φ * ū
                    u[iB, jB, kB] = u_new
                end
            end            
        else
            τ̃ = Δt / τ
            if τ̃ > 0
                @inbounds begin
                    u_rad          = u[iB, jB, kB]       # after radiation step
                    u_old_boundary = u_old[iB, jB, kB]   # previous time-level

                    u_new = u_rad + τ̃ * (ū - u_old_boundary)
                    u[iB, jB, kB] = u_new
                end
            end
        end
        return nothing
    end
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PAOBC, loc::Tuple{Any, Face, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)
    boundary_interior_indices = (i, 3, k)

    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                        grid, u, clock, model_fields, loc, Δy)

    return nothing
end


@inline function _fill_top_halo!(i, j, grid, u, bc::PAOBC, loc::Tuple{Any, Any, Face}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)
    boundary_interior_indices = (i, j, k-2)

    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    step_right_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                         grid, u, clock, model_fields, loc, Δz)

    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, u, bc::PAOBC, loc::Tuple{Any, Any, Face}, clock, model_fields)
    boundary_indices = (i, j, 1)
    boundary_adjacent_indices = (i, j, 2)
    boundary_interior_indices = (i, j, 3)

    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    step_left_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                        grid, u, clock, model_fields, loc, Δz)

    return nothing
end
