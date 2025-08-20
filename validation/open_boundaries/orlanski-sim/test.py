# orlanski_movie_tracer.py
# Self-contained 1D SWE + Orlanski(u) + Passive tracer + MP4 output.
# Requires: numpy, matplotlib, ffmpeg installed.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

g = 9.80665

# -----------------------------
# Grid
# -----------------------------
class Grid:
    def __init__(self, Nx, Lx):
        self.Nx = Nx
        self.Lx = Lx
        self.dx = Lx / Nx

# -----------------------------
# Orlanski BC on u (unchanged)
# -----------------------------
def update_west_u_orlanski(u, u_hist, dx, dt, Lz):
    i1, i2 = 1, 2
    dux = u[i2] - u[i1]
    dut = u[i1] - u_hist[i1]
    max_speed = -np.sqrt(g * Lz)
    if abs(dux * dt) > 1e-20:
        speed = -(dut * dx) / (dux * dt)
    else:
        speed = 0.0
    if np.isnan(speed): speed = 0.0
    if speed < 0.0: speed = 0.0
    elif speed > max_speed: speed = max_speed
    c = speed * dt / dx
    u[0] = (u[0] - c * u[i1]) / (1.0 - c)

def update_east_u_orlanski(u, u_hist, dx, dt, Lz, Nx):
    iNm1, iN, iGhost = Nx - 1, Nx, Nx + 1
    dux = u[iN] - u[iNm1]
    dut = u[iN] - u_hist[iN]
    max_speed = -np.sqrt(g * Lz)
    if abs(dux * dt) > 1e-20:
        speed = -(dut * dx) / (dux * dt)
    else:
        speed = 0.0
    if np.isnan(speed): speed = 0.0
    if speed < 0.0: speed = 0.0
    elif speed > max_speed: speed = max_speed
    c = speed * dt / dx
    u[iGhost] = (u[iGhost] - c * u[iN]) / (1.0 - c)

# -----------------------------
# Other BCs
# -----------------------------
def eta_neumann_bc(eta):
    eta[0]  = eta[1]
    eta[-1] = eta[-2]

def tracer_neumann_bc(c):
    c[0]  = c[1]
    c[-1] = c[-2]

# Optional sponge to reduce box modes (kept mild)
def apply_sponge(u, eta, c, Nx, width, alpha, dt):
    if width <= 0 or alpha <= 0: 
        return
    coefL = alpha * dt * np.linspace(1.0, 0.2, width)
    u[1:1+width]   *= (1.0 - coefL)
    eta[1:1+width] *= (1.0 - coefL)
    c[1:1+width]   *= (1.0 - coefL)
    coefR = alpha * dt * np.linspace(0.2, 1.0, width)
    u[Nx-width+1:Nx+1]   *= (1.0 - coefR)
    eta[Nx-width+1:Nx+1] *= (1.0 - coefR)
    c[Nx-width+1:Nx+1]   *= (1.0 - coefR)

# -----------------------------
# Numerical helpers
# -----------------------------
def central_gradient(phi, dx):
    # centered derivative on interior (1..Nx)
    return (phi[2:] - phi[:-2]) / (2 * dx)

# Flux-form upwind tracer advection
def tracer_flux_upwind(u_face, c_left, c_right):
    # Godunov upwind: if u_face >= 0 use left state, else right state
    return np.where(u_face >= 0.0, u_face * c_left, u_face * c_right)

def tracer_tendency(c, u, dx, kappa):
    """
    Conservative flux-form upwind advection for tracer c with ghost cells.

    Arrays have length Nx+2 (ghosts at i=0 and i=Nx+1).
    - Faces are between i and i+1 for i = 0..Nx  (=> Nx+1 faces).
    - Interior cells are i = 1..Nx               (=> Nx cells).
    """
    # Face velocities u_{i+1/2} for i=0..Nx  -> length Nx+1
    u_face = 0.5 * (u[:-1] + u[1:])         # uses ghosts at both ends

    # Upwind states at faces (left/right cells around each face)
    c_L = c[:-1]                             # cells i=0..Nx     -> length Nx+1
    c_R = c[1:]                              # cells i=1..Nx+1   -> length Nx+1

    # Godunov upwind flux at each face
    F = np.where(u_face >= 0.0, u_face * c_L, u_face * c_R)  # length Nx+1

    # Flux divergence over interior cells 1..Nx -> length Nx
    divF = (F[1:] - F[:-1]) / dx             # faces 1..Nx minus 0..Nx-1

    # Assemble tendency
    tend = np.zeros_like(c)
    tend[1:-1] = -divF                       # matches length Nx

    # Optional diffusion (centered Laplacian on interior)
    if kappa > 0.0:
        lap = (c[2:] - 2*c[1:-1] + c[:-2]) / (dx * dx)
        tend[1:-1] += kappa * lap

    return tend


# -----------------------------
# RK2 (Heun) for SWE + tracer
# -----------------------------
def rk2_step(u, eta, c, H, grid: Grid, dt, kappa):
    # SWE tendencies
    du_dx   = central_gradient(u,   grid.dx)
    dη_dx   = central_gradient(eta, grid.dx)

    η_t = np.zeros_like(eta); η_t[1:-1] = - H * du_dx
    u_t = np.zeros_like(u);   u_t[1:-1] = - g * dη_dx

    # Tracer tendency
    c_t = tracer_tendency(c, u, grid.dx, kappa)

    # Predictor
    η_p = eta + dt * η_t
    u_p = u   + dt * u_t
    c_p = c   + dt * c_t

    # Predictor BCs (use u as history for Orlanski predictor stage)
    update_west_u_orlanski(u_p, u, grid.dx, dt, H)
    update_east_u_orlanski(u_p, u, grid.dx, dt, H, grid.Nx)
    eta_neumann_bc(η_p)
    tracer_neumann_bc(c_p)

    # Corrector tendencies
    du_dx_p = central_gradient(u_p,   grid.dx)
    dη_dx_p = central_gradient(η_p,   grid.dx)
    η_t_p   = np.zeros_like(eta); η_t_p[1:-1] = - H * du_dx_p
    u_t_p   = np.zeros_like(u);   u_t_p[1:-1] = - g * dη_dx_p
    c_t_p   = tracer_tendency(c_p, u_p, grid.dx, kappa)

    # Heun average
    η_new = eta + 0.5 * dt * (η_t + η_t_p)
    u_new = u   + 0.5 * dt * (u_t + u_t_p)
    c_new = c   + 0.5 * dt * (c_t + c_t_p)

    return u_new, η_new, c_new

# -----------------------------
# Main: run and save MP4
# -----------------------------
def main():
    # Grid & physics
    Nx = 400
    Lx = 500_000.0    # 500 km
    H  = 10.0         # depth (used in Orlanski speed cap)
    grid = Grid(Nx, Lx)

    celerity = np.sqrt(g * H)
    CFL = 0.3
    dt = CFL * grid.dx / celerity
    t_end = 3.5 * (Lx / celerity)   # multiple crossings

    # Ghosted arrays: i=0 and i=Nx+1 are ghosts
    u   = np.zeros(Nx + 2)
    eta = np.zeros(Nx + 2)
    c   = np.zeros(Nx + 2)   # passive tracer
    u_hist = u.copy()

    # Initial conditions
    x = (np.arange(Nx + 2) - 1) * grid.dx
    Rx = 250_000.0
    sigma = 50_000.0
    eta[:] = 0.1 * np.exp(-((x - Rx) ** 2) / (2.0 * sigma ** 2))
    # Tracer: co-located dye (narrower than η to visualize better)
    c_center = Rx - 0.6 * sigma
    c[:]   = np.exp(-((x - c_center) ** 2) / (2.0 * (0.6 * sigma) ** 2))


    eta_neumann_bc(eta)
    tracer_neumann_bc(c)

    # u[:] = 0.0
    # u[1:-1] = +celerity * eta[1:-1]
    update_west_u_orlanski(u, u, grid.dx, dt, H)   # use u as history just for init
    update_east_u_orlanski(u, u, grid.dx, dt, H, grid.Nx)
    u_hist[:] = u

    # Tracer diffusivity (small; tune 0.5–2 m^2/s)
    kappa = 2.0

    # Sponge settings (mild)
    sponge_w = max(3, int(0.05 * Nx))
    alpha = 0.0015  # s^-1

    # Plot + writer
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    line_eta, = ax1.plot(x * 1e-3, eta, label="η")
    line_u,   = ax1.plot(x * 1e-3, u,   label="u")
    ax1.set_ylabel("η, u")
    ax1.legend(loc="best")
    ax1.set_title("1D SWE + Orlanski(u) + Passive tracer")

    line_c,   = ax2.plot(x * 1e-3, c,   label="tracer c")
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("c")
    ax2.legend(loc="best")

    ax1.set_ylim(1.3 * eta.min(), 1.3 * eta.max())
    ax2.set_ylim(-0.1 * c.max(), 1.2 * c.max())

    writer = FFMpegWriter(fps=30, metadata=dict(artist="orlanski+tracer"), bitrate=2000)
    out_path = "orlanski_shallow_water_tracer.mp4"

    t = 0.0
    with writer.saving(fig, out_path, dpi=144):
        while t < t_end:
            # Advance
            u_new, eta_new, c_new = rk2_step(u, eta, c, H, grid, dt, kappa)

            # Final BCs with true history
            u[:] = u_new
            eta[:] = eta_new
            c[:] = c_new

            update_west_u_orlanski(u, u_hist, grid.dx, dt, H)
            update_east_u_orlanski(u, u_hist, grid.dx, dt, H, grid.Nx)
            eta_neumann_bc(eta)
            tracer_neumann_bc(c)

            # Sponge
            apply_sponge(u, eta, c, grid.Nx, sponge_w, alpha, dt)

            # Update history for next step
            u_hist[:] = u

            # Frame
            line_eta.set_ydata(eta)
            line_u.set_ydata(u)
            line_c.set_ydata(c)
            writer.grab_frame()

            t += dt

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
