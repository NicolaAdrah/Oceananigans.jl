using Pkg
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

const OTime = Oceananigans.Units.Time

import ClimaOcean.DataWrangling: restrict
import ClimaOcean.DataWrangling: NearestNeighborInpainting

ClimaOcean.DataWrangling.restrict(bbox_interfaces, interfaces, N) = begin
    Δ  = interfaces[2] - interfaces[1]
    rΔ = bbox_interfaces[2] - bbox_interfaces[1]
    ϵ  = rΔ / Δ
    rN = round(Int, ϵ * N)      # <- here it was Integer(...)
    return bbox_interfaces, rN
end

# ---------- End of modifications

arch = CPU()

ds = Dataset("data/mesh_mask2D_NA.nc")
# ds = Dataset("/leonardo_scratch/large/userexternal/naladrah/cluster_med/data/mesh_mask2D_NA.nc")
xc = ds["XC"][:]
yc = ds["YC"][:]

x_faces = [xc..., xc[end] + (xc[end] - xc[end-1])]
y_faces = [yc..., yc[end] + (yc[end] - yc[end-1])]

z_faces = [-228.9898, -217.6341, -206.7042, -196.1894, -186.0793, -176.3635, 
    -167.032, -158.075, -149.4827, -141.2458, -133.3548, -125.8009, -118.575, 
    -111.6685, -105.0728, -98.77982, -92.78125, -87.06921, -81.63593, 
    -76.47385, -71.57558, -66.93387, -62.54169, -58.39215, -54.47855, 
    -50.79433, -47.33311, -44.08865, -41.05489, -38.2259, -35.59592, 
    -33.15932, -30.91062, -28.8445, -26.95574, -25.2393, -23.69024, 
    -22.30376, -21.0752, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 
    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

Nz = length(z_faces) - 1 # 140 vertical levels
Nx = length(x_faces) - 1
Ny = length(y_faces) - 1

z_faces = MutableVerticalDiscretization(z_faces)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = y_faces,
                             longitude = x_faces,
                             z = z_faces,
                             halo = (7, 7, 7))

bottom_height = - ds["Depth"][:, :]
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
start_date = Date(2017, 1, 1)
end_date   = Date(2020, 12, 30)

const POINTS_PER_DEG = 12  # GLORYS 1/12°

snap_floor(x) = floor(Int, x * POINTS_PER_DEG) / POINTS_PER_DEG
snap_ceil(x)  = ceil(Int,  x * POINTS_PER_DEG) / POINTS_PER_DEG

lon_min = snap_floor(x_faces[1] - 2)
lon_max = snap_ceil( x_faces[end] + 2)
lat_min = snap_floor(y_faces[1] - 2)
lat_max = snap_ceil( y_faces[end] + 2)

bbox_snapped = ClimaOcean.DataWrangling.BoundingBox(
    longitude = (lon_min, lon_max),
    latitude  = (lat_min, lat_max)
)

dir = "./data"
dataset = GLORYSDaily()
S_meta = Metadata(:salinity;    dataset, dir,
                  bounding_box=bbox_snapped,
                  start_date=start_date, end_date=end_date)

S = CenterField(grid)

set!(S, S_meta[1], inpainting=NearestNeighborInpainting(200))

using CairoMakie

heatmap(interior(S, :, :, 59))

