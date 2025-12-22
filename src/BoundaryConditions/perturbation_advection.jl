# TODO Nicola
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
   @warn "Accessing step_right_boundary!"
   iᴮ, jᴮ, kᴮ = boundary_indices
   iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
   Δt = clock.last_stage_Δt
   Δt = ifelse(isinf(Δt), 0, Δt)

   uᵢⁿ     = @inbounds getindex(u, iᴮ, jᴮ, kᴮ)
   uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, iᴬ, jᴬ, kᴬ)
   normal_axis = axis_from_indices(boundary_indices, boundary_adjacent_indices)
   Ũ = dimensionless_phase_speed(u, uᵢⁿ, boundary_adjacent_indices, boundary_interior_indices,
                                  grid, loc, Δt, ΔX, normal_axis, 1)
   if Ũ isa AbstractFloat && !isfinite(Ũ)
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
       uᵢⁿ⁺¹ = abs(denom) < 100 * eps(eltype(denom)) ? uᵢⁿ : numer / denom
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

# TODO Nicola: debugging and diagnostics
const ORLANSKI_EPS = 1e-20
const RADIATION_2D = true
const IMPLICIT_NUDGING = false
const ENABLE_ORLANSKI_DEBUG = false
const ORLANSKI_DEBUG_START = 0
const ORLANSKI_DEBUG_EVERY = 1
const ORLANSKI_MAX_CFL = 0.3
# TODO Nicola: Enhanced diagnostics flags
const ENABLE_DETAILED_DIAGNOSTICS = false  # Print detailed value-by-value diagnostics (disabled after fixing corner cell issue)
const CHECK_IMMERSED_BOUNDARY = true       # Check if cells are inside immersed boundaries
const SAFE_MODE = false                     # If true, use fallback values instead of erroring
const LOG_CORNER_CELL_WARNINGS = false     # If true, log warnings for corner cells (can be noisy)
const LOG_ONLY_ON_ERROR = true             # Only log detailed diagnostics when actual non-finite values detected  

@inline function maybe_log_orlanski(iter, i, k; stage=:radiation, kwargs...)
    ENABLE_ORLANSKI_DEBUG || return nothing
    iter >= ORLANSKI_DEBUG_START || return nothing
    (iter - ORLANSKI_DEBUG_START) % ORLANSKI_DEBUG_EVERY == 0 || return nothing
    @info "[OrlanskiDebug]" stage=stage iteration=iter i=i k=k kwargs...
    return nothing
end

# TODO Nicola: Helper function to validate array bounds
@inline function validate_orlanski_bounds(u::AbstractArray, i::Int, j::Int, k::Int, 
                                         j_adj::Int, j_intr_1::Int, j_intr_2::Int)
    Nx, Ny, Nz = size(u)
    
    # Check i bounds (need i-1, i, i+1)
    if i < 2 || i > Nx - 1
        return false, "i=$i out of valid range [2, $(Nx-1)]"
    end
    
    # Check j bounds for all accessed indices
    if j < 1 || j > Ny
        return false, "j_halo=$j out of valid range [1, $Ny]"
    end
    if j_adj < 1 || j_adj > Ny
        return false, "j_adj=$j_adj out of valid range [1, $Ny]"
    end
    if j_intr_1 < 1 || j_intr_1 > Ny
        return false, "j_intr_1=$j_intr_1 out of valid range [1, $Ny]"
    end
    if j_intr_2 < 1 || j_intr_2 > Ny
        return false, "j_intr_2=$j_intr_2 out of valid range [1, $Ny]"
    end
    
    # Check k bounds
    if k < 1 || k > Nz
        return false, "k=$k out of valid range [1, $Nz]"
    end
    
    return true, "All bounds valid"
end

# TODO Nicola: Helper function to check if cell is in immersed boundary
@inline function is_immersed_cell(grid, i::Int, j::Int, k::Int)
    # Check if grid has immersed boundary information
    if hasproperty(grid, :immersed_boundary)
        ib = grid.immersed_boundary
        if hasproperty(ib, :mask)
            # If mask is 0, the cell is inside the immersed boundary
            return ib.mask[i, j, k] == 0
        end
    end
    return false
end

# let this function return dVdt and dVde to be used outside for the nudging.
@inline function orlanski_south_boundary!(
   u::AbstractArray,      # current time level (like kout)
   u_old::AbstractArray,  # previous time level (like know)
   i::Int, k::Int, j::Int; iter::Union{Nothing, Int}=nothing,
)
   Nx = size(u, 1)
   Ny = size(u, 2)
 
   j_halo   = j
   j_adj    = j + 1
   j_intr_1 = j + 2
   j_intr_2 = j + 3
 
   # TODO Nicola: Validate array bounds
   bounds_valid, bounds_msg = validate_orlanski_bounds(u, i, j, k, j_adj, j_intr_1, j_intr_2)
   if !bounds_valid
       @static if LOG_CORNER_CELL_WARNINGS
           @warn "[OrlanskiBounds] Insufficient neighbors for Orlanski BC - using fallback" i=i j=j k=k message=bounds_msg Nx=Nx Ny=Ny iter=iter
       end
       # For corner cells or cells without sufficient neighbors, use simple fallback
       # Option 1: Keep current value (do nothing)
       # Option 2: Copy from adjacent interior cell
       # Option 3: Set to zero
       # We'll keep the current value for now
       return zero(eltype(u)), zero(eltype(u)), u[i, j_halo, k]
   end
   
   # Require enough neighbors in i and j
   if i < 2 || i > Nx - 1
        # @warn "not enough grid points to apply Orlanski BC at (i=$i, j=$j_halo, k=$k). Skipping."
        return zero(eltype(u)), zero(eltype(u)), u[i, j_halo, k]
   end
 
   # TODO Nicola: Check for immersed boundary cells - we'll need grid for this
   # This will be checked in step_left_boundary! where we have access to grid
 
   dVdt = zero(eltype(u))
   dVde = zero(eltype(u))
   @inbounds begin
    # =========================================================================
    # TODO Nicola: The following block has been commented out and replaced 
    # with corrected indices matching ROMS implementation.
    # =========================================================================
    #    # ROMS:
    #    # grad(i, j_adj)   = vbar(i, j_adj,   know) - vbar(i-1, j_adj,   know)
    #    # grad(i, j_adj+1) = vbar(i, j_adj+1, know) - vbar(i-1, j_adj+1, know)
    #    grad_j_adj_I_bw      = u_old[i,   j_adj, k] - u_old[i-1, j_adj, k]
    #    grad_j_adj_I_fw  = u_old[i+1, j_adj, k] - u_old[i,   j_adj, k]
    #
    #    grad_j_intr1_I_bw        = u_old[i,   j_intr_1,  k] - u_old[i-1, j_intr_1,  k]
    #    grad_j_intr1_I_fw    = u_old[i+1, j_intr_1,  k] - u_old[i,   j_intr_1,  k]
    #
    #    # ROMS:
    #    #   dVdt = vbar(i, j_adj+1, know) - vbar(i, j_adj+1, kout)
    #    #   dVde = vbar(i, j_adj+1, kout) - vbar(i, j_adj+2, kout)
    #    v_know_j_intr_1  = u_old[i, j_intr_1, k]
    #    v_kout_j_intr_1  = u[i,     j_intr_1, k]
    #    v_kout_j_intr_2  = u[i,     j_intr_2, k]
    #
    #    dVdt = v_know_j_intr_1 - v_kout_j_intr_1
    #    dVde = v_kout_j_intr_1 - v_kout_j_intr_2
    #
    #    if dVdt * dVde < 0
    #        dVdt = zero(dVdt)
    #    end
    #
    #    # ROMS:
    #    # IF (dVdt*(grad(i,j_adj+1) + grad(i+1,j_adj+1))) > 0 THEN
    #    #   dVdx = grad(i,j_adj+1)
    #    # ELSE
    #    #   dVdx = grad(i+1,j_adj+1)
    #    s    = dVdt * (grad_j_intr1_I_bw + grad_j_intr1_I_fw)
    #    dVdx = s > 0 ? grad_j_intr1_I_bw : grad_j_intr1_I_fw
    #
    #    # cff = MAX(dVdx*dVdx + dVde*dVde, eps)
    #    cff = dVdx * dVdx + dVde * dVde
    #
    #    # In ROMS:
    #    #   Cx = MIN(cff, MAX(dVdt*dVdx, -cff))   (for RADIATION_2D)
    #    #   Ce = dVdt * dVde
    #   
    #    # TODO Nicola put keyword to accept @static condition in the function call.
    #    @static if RADIATION_2D
    #    # println("using RADIATION_2D")
    #        Cx = min(cff, max(dVdt * dVdx, -cff))
    #    else
    #     #  println("not using RADIATION_2D")
    #        Cx = 0.0
    #    end
    #
    #    Ce = dVdt * dVde
    #
    #    v_know_j_adj = u_old[i, j_adj, k]
    #     vals = (v_know_j_adj, v_kout_j_intr_1, v_kout_j_intr_2,
    #             grad_j_adj_I_bw, grad_j_adj_I_fw,
    #             grad_j_intr1_I_bw, grad_j_intr1_I_fw)
    #
    #     if any(!isfinite, vals)
    #         error("Non-finite value encountered in Orlanski boundary condition computation at (i=$i, j=$j_halo, k=$k). Values: $vals")
    #     end
    #
    #    denom = cff + Ce
    #    if denom < ORLANSKI_EPS
    #         denom = ORLANSKI_EPS
    #    end
    #
    #    # To be as close as possible to ROMS, we *do not* add extra guards here.
    #    # If denom ≈ 0, the scheme is marginal; with your Δt and grid this
    #    # should not systematically happen.
    # #    @info "cff=$cff, Ce=$Ce, Cx=$Cx, denom=$denom"
    #    v_boundary = (  cff * v_know_j_adj
    #                   + Ce  * v_kout_j_intr_1
    #                   - max(Cx, zero(Cx)) * grad_j_adj_I_bw
    #                   - min(Cx, zero(Cx)) * grad_j_adj_I_fw
    #                  ) / denom
    #
    #     u[i, j_halo, k] = v_boundary
    #
    # #    if any(!isfinite, (v_know_j_adj, v_kout_j_intr_1, v_kout_j_intr_2,
    # #                grad_j_adj_I_bw, grad_j_adj_I_fw,
    # #                grad_j_intr1_I_bw, grad_j_intr1_I_fw, cff, Ce, Cx))
    # #         @warn "[OrlanskiZeros] injected zero at (i=$i, j=$j, k=$k)"
    # #         v_boundary = zero(eltype(u))  # or getbc(bc, l, m, grid, clock, model_fields)
    # #     end
    # #     u[i, j_halo, k] = v_boundary
    #
    #    if iter !== nothing
    #        maybe_log_orlanski(iter, i, k;
    #                           stage=:radiation,
    #                           dVdt=dVdt, dVde=dVde, Cx=Cx, Ce=Ce, cff=cff, denom=denom,
    #                           v_boundary=v_boundary,
    #                           grad_bw=grad_j_adj_I_bw, grad_fw=grad_j_adj_I_fw,
    #                           v_adj=v_know_j_adj,
    #                           v_intr1_old=v_know_j_intr_1,
    #                           v_intr1_new=v_kout_j_intr_1,
    #                           v_intr2_new=v_kout_j_intr_2)
    #    end

       # =========================================================================
       # Corrected Implementation 
       # =========================================================================
   
       grad_j_halo_I_bw = u_old[i,   j_halo, k] - u_old[i-1, j_halo, k]
       grad_j_halo_I_fw = u_old[i+1, j_halo, k] - u_old[i,   j_halo, k]
   
       grad_j_adj_I_bw  = u_old[i,   j_adj, k] - u_old[i-1, j_adj, k]
       grad_j_adj_I_fw  = u_old[i+1, j_adj, k] - u_old[i,   j_adj, k]
   
       # ROMS Phase speed estimation at Jstr+1 (j_adj):
       # dVdt = vbar(i, Jstr+1, know) - vbar(i, Jstr+1, kout)
       v_know_j_adj = u_old[i, j_adj, k]
       v_kout_j_adj = u[i,     j_adj, k]
       
       # dVde = vbar(i, Jstr+1, kout) - vbar(i, Jstr+2, kout)
       v_kout_j_intr_1 = u[i, j_intr_1, k] # (j_adj + 1)
   
       dVdt = v_know_j_adj - v_kout_j_adj
       dVde = v_kout_j_adj - v_kout_j_intr_1
   
       if dVdt * dVde < 0
           dVdt = zero(dVdt)
       end
   
       # dVdx = grad(i, Jstr+1) ..
       s    = dVdt * (grad_j_adj_I_bw + grad_j_adj_I_fw)
       dVdx = s > 0 ? grad_j_adj_I_bw : grad_j_adj_I_fw
   
       cff = dVdx * dVdx + dVde * dVde
   
       @static if RADIATION_2D
           Cx = min(cff, max(dVdt * dVdx, -cff))
       else
           Cx = 0.0
       end
   
       Ce = dVdt * dVde
   
       v_know_j_halo = u_old[i, j_halo, k]
       
       # TODO Nicola: Enhanced diagnostics - check each value individually
       @static if ENABLE_DETAILED_DIAGNOSTICS
           # Always run detailed diagnostics
           # First, check all raw input values from arrays
           raw_vals = (
               u_old_i_jh = u_old[i, j_halo, k],
               u_old_im1_jh = u_old[i-1, j_halo, k],
               u_old_ip1_jh = u_old[i+1, j_halo, k],
               u_old_i_jadj = u_old[i, j_adj, k],
               u_old_im1_jadj = u_old[i-1, j_adj, k],
               u_old_ip1_jadj = u_old[i+1, j_adj, k],
               u_i_jadj = u[i, j_adj, k],
               u_i_jintr1 = u[i, j_intr_1, k],
           )
           
           for (name, val) in pairs(raw_vals)
               if !isfinite(val)
                   @warn "[OrlanskiNonFinite] Non-finite raw input value detected" location=(i,j,k) iteration=iter field=name value=val
                   @static if SAFE_MODE
                       return zero(eltype(u)), zero(eltype(u)), zero(eltype(u))
                   else
                       error("Non-finite INPUT value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
           end
           
           # Check computed gradients
           grad_vals = (
               grad_j_halo_I_bw = grad_j_halo_I_bw,
               grad_j_halo_I_fw = grad_j_halo_I_fw,
               grad_j_adj_I_bw = grad_j_adj_I_bw,
               grad_j_adj_I_fw = grad_j_adj_I_fw,
           )
           
           for (name, val) in pairs(grad_vals)
               if !isfinite(val)
                   @warn "[OrlanskiNonFinite] Non-finite gradient detected" location=(i,j,k) iteration=iter field=name value=val
                   @static if SAFE_MODE
                       return zero(eltype(u)), zero(eltype(u)), zero(eltype(u))
                   else
                       error("Non-finite GRADIENT in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
           end
           
           # Check intermediate phase speed values
           phase_vals = (
               v_know_j_adj = v_know_j_adj,
               v_kout_j_adj = v_kout_j_adj,
               v_kout_j_intr_1 = v_kout_j_intr_1,
               v_know_j_halo = v_know_j_halo,
               dVdt = dVdt,
               dVde = dVde,
               s = s,
               dVdx = dVdx,
           )
           
           for (name, val) in pairs(phase_vals)
               if !isfinite(val)
                   @warn "[OrlanskiNonFinite] Non-finite phase speed value detected" location=(i,j,k) iteration=iter field=name value=val
                   @static if SAFE_MODE
                       return zero(eltype(u)), zero(eltype(u)), zero(eltype(u))
                   else
                       error("Non-finite PHASE SPEED value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
           end
           
           # Check final computation values
           final_vals = (
               cff = cff,
               Cx = Cx,
               Ce = Ce,
           )
           
           for (name, val) in pairs(final_vals)
               if !isfinite(val)
                   @warn "[OrlanskiNonFinite] Non-finite final computation value detected" location=(i,j,k) iteration=iter field=name value=val
                   @static if SAFE_MODE
                       return zero(eltype(u)), zero(eltype(u)), zero(eltype(u))
                   else
                       error("Non-finite FINAL value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
           end
       elseif LOG_ONLY_ON_ERROR
           # Quick check first, only run detailed diagnostics if there's a problem
           quick_check_vals = (v_know_j_halo, v_kout_j_adj, v_kout_j_intr_1,
                             grad_j_halo_I_bw, grad_j_halo_I_fw,
                             grad_j_adj_I_bw, grad_j_adj_I_fw,
                             dVdt, dVde, s, dVdx, cff, Cx, Ce)
           
           if any(!isfinite, quick_check_vals)
               # Found a problem - run detailed diagnostics
               raw_vals = (
                   u_old_i_jh = u_old[i, j_halo, k],
                   u_old_im1_jh = u_old[i-1, j_halo, k],
                   u_old_ip1_jh = u_old[i+1, j_halo, k],
                   u_old_i_jadj = u_old[i, j_adj, k],
                   u_old_im1_jadj = u_old[i-1, j_adj, k],
                   u_old_ip1_jadj = u_old[i+1, j_adj, k],
                   u_i_jadj = u[i, j_adj, k],
                   u_i_jintr1 = u[i, j_intr_1, k],
               )
               
               for (name, val) in pairs(raw_vals)
                   if !isfinite(val)
                       @warn "[OrlanskiNonFinite] Non-finite raw input value detected" location=(i,j,k) iteration=iter field=name value=val
                       error("Non-finite INPUT value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
               
               # Check gradients
               for (name, val) in pairs((grad_j_halo_I_bw=grad_j_halo_I_bw, grad_j_halo_I_fw=grad_j_halo_I_fw,
                                        grad_j_adj_I_bw=grad_j_adj_I_bw, grad_j_adj_I_fw=grad_j_adj_I_fw))
                   if !isfinite(val)
                       @warn "[OrlanskiNonFinite] Non-finite gradient detected" location=(i,j,k) iteration=iter field=name value=val
                       error("Non-finite GRADIENT in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
               
               # Check phase speeds
               for (name, val) in pairs((dVdt=dVdt, dVde=dVde, s=s, dVdx=dVdx))
                   if !isfinite(val)
                       @warn "[OrlanskiNonFinite] Non-finite phase speed value detected" location=(i,j,k) iteration=iter field=name value=val
                       error("Non-finite PHASE SPEED value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
               
               # Check final values
               for (name, val) in pairs((cff=cff, Cx=Cx, Ce=Ce))
                   if !isfinite(val)
                       @warn "[OrlanskiNonFinite] Non-finite final computation value detected" location=(i,j,k) iteration=iter field=name value=val
                       error("Non-finite FINAL value in Orlanski at (i=$i, j=$j_halo, k=$k): $name = $val")
                   end
               end
           end
       else
           # Legacy simple check
           vals = (v_know_j_halo, v_kout_j_adj, v_kout_j_intr_1,
                   grad_j_halo_I_bw, grad_j_halo_I_fw,
                   grad_j_adj_I_bw, grad_j_adj_I_fw)
       
           if any(!isfinite, vals)
                error("Non-finite value encountered in Orlanski (Corrected) at (i=$i, j=$j_halo, k=$k).")
           end
       end
   
       denom = cff + Ce
       if denom < ORLANSKI_EPS
           denom = ORLANSKI_EPS
       end
   
       # ROMS Update at Jstr (j_halo):
       # vbar(i,Jstr,kout) = ( cff * vbar(i,Jstr,know) + Ce * vbar(i,Jstr+1,kout)
       #                       - MAX(Cx,0)*grad(i,Jstr) - MIN(Cx,0)*grad(i+1,Jstr) ) / (cff+Ce)
       
       v_boundary = (  cff * v_know_j_halo
                     + Ce  * v_kout_j_adj
                     - max(Cx, zero(Cx)) * grad_j_halo_I_bw
                     - min(Cx, zero(Cx)) * grad_j_halo_I_fw
                    ) / denom
   
       # TODO Nicola: Final safety check on computed boundary value
       @static if ENABLE_DETAILED_DIAGNOSTICS
           if !isfinite(v_boundary)
               @warn "[OrlanskiNonFinite] Non-finite boundary value computed" location=(i,j_halo,k) iteration=iter v_boundary=v_boundary cff=cff Ce=Ce Cx=Cx denom=denom
               @static if SAFE_MODE
                   v_boundary = zero(eltype(u))
                   @info "[OrlanskiSafe] Replaced non-finite boundary value with zero" location=(i,j_halo,k)
               else
                   error("Non-finite BOUNDARY VALUE computed in Orlanski at (i=$i, j=$j_halo, k=$k): v_boundary = $v_boundary")
               end
           end
       end
   
       u[i, j_halo, k] = v_boundary
   
       if iter !== nothing
          maybe_log_orlanski(iter, i, k;
                             stage=:radiation,
                             dVdt=dVdt, dVde=dVde, Cx=Cx, Ce=Ce, cff=cff, denom=denom,
                             v_boundary=v_boundary)
       end
   end

   return dVdt, dVde, u[i, j_halo, k]
end

@inline function step_left_boundary!(bc::PAOBC,
                                    l, m,
                                    boundary_indices,
                                    boundary_adjacent_indices,
                                    boundary_interior_indices,
                                    grid, u, clock, model_fields, loc, ΔX)

   iB, jB, kB = boundary_indices
   normal_axis = axis_from_indices(boundary_indices, boundary_adjacent_indices)

   if normal_axis === Y_AXIS
       pa = bc.classification.scheme  # ::PerturbationAdvection

       # TODO Nicola: Check if this cell is in an immersed boundary
       @static if CHECK_IMMERSED_BOUNDARY
           if is_immersed_cell(grid, iB, jB, kB)
               @warn "[OrlanskiImmersed] Skipping Orlanski BC for immersed boundary cell" location=(iB,jB,kB) iteration=clock.iteration
               # Set to zero or keep current value
               u[iB, jB, kB] = zero(eltype(u))
               return nothing
           end
       end

       # 1. Ensure we have a previous time level
    #    Couldn't be that we are not updating correctly?
       if pa.previous === nothing
           @static if ENABLE_DETAILED_DIAGNOSTICS
               @info "[OrlanskiInit] Initializing pa.previous (first time)" iteration=clock.iteration
           end
           pa.previous = similar(u)
           copyto!(pa.previous, u)
       end
       u_old = pa.previous
       
       # TODO Nicola: Check for non-finite values in u_old
       @static if ENABLE_DETAILED_DIAGNOSTICS
           if !isfinite(u_old[iB, jB, kB])
               @warn "[OrlanskiNonFinite] Non-finite value in u_old (pa.previous)" location=(iB,jB,kB) iteration=clock.iteration value=u_old[iB, jB, kB]
               @static if SAFE_MODE
                   u_old[iB, jB, kB] = zero(eltype(u_old))
               else
                   error("Non-finite value in u_old at (i=$iB, j=$jB, k=$kB): $(u_old[iB, jB, kB])")
               end
           end
       end

       log_iter = clock.iteration

       # 2. Pure radiation update at (iB, kB), and get dVdt, dVde
       dVdt, dVde, u[iB, jB, kB] = orlanski_south_boundary!(u, u_old, iB, kB, jB; iter=log_iter)

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
               maybe_log_orlanski(log_iter, iB, kB;
                                  stage=:nudging,
                                  σ=σ, τ=τ, Δt=Δt, φ=φ,
                                  u_rad=u_rad, u_new=u_new, ū=ū)
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
               maybe_log_orlanski(log_iter, iB, kB;
                                  stage=:nudging,
                                  σ=σ, τ=τ, Δt=Δt, τ̃=τ̃,
                                  u_rad=u_rad, u_old_boundary=u_old_boundary, u_new=u_new, ū=ū)
           end
       end
    # Mask out inactive cells if a mask exists
    # val = @inbounds u[iB, jB, kB]
    # if hasproperty(grid, :active_cells_map)
    #     mask = grid.active_cells_map[1][iB, jB, kB]
    #     val *= mask
    # end

    # CFL clamp on the boundary value
    val = u[iB, jB, kB]
    # Δt = clock.last_stage_Δt # Already defined above
    cfl = abs(val) * Δt / ΔX
    if cfl > ORLANSKI_MAX_CFL && cfl > 0
        val *= ORLANSKI_MAX_CFL / cfl
        u[iB, jB, kB] = val
    end

       return nothing
   end
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PAOBC, loc::Tuple{Any, Face, Any}, clock, model_fields)
   # We must fill from the interior outwards.
   # The boundary is at j=1.
   # We also need to fill the ghost cells j=0, j=-1, ... down to 1-Hy.
   
   Hy = grid.Hy
   for j in 1:-1:(1-Hy)
       boundary_indices = (i, j, k)
       boundary_adjacent_indices = (i, j+1, k)
       boundary_interior_indices = (i, j+2, k)
    
       Δy = Δyᶜᶠᶜ(i, j, k, grid)
       step_left_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, boundary_interior_indices,
                           grid, u, clock, model_fields, loc, Δy)
   end

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
