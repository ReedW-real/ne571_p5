from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import sys
import os


# User-defined parameters ------------------------------------------------------

# Define node counts
"""
Node counts are defined as integers, where each integer represents the number of nodes in the x, y, and z directions.
The node counts are defined as follows:
nx = 3: Number of nodes in the x direction
ny = 3: Number of nodes in the y direction
nz = 3: Number of nodes in the z direction
"""
nx = 11
ny = 11
nz = 11


# Define materials
"""
Materials are defined as a 3D array of integers, where each integer represents a different material.
The materials are defined as follows:
0: Water
1: Fuel
2: Control Rod
"""
zones = np.zeros((nx, ny, nz), dtype=int)
zones[:, :, :] = 1

# checker board pattern of med and low enriched fuel (constant in z, checkerboard in x-y)
for i in range(1, nx-1):
    for j in range(1, ny-1):
        if (i+j) % 2 == 0:
            zones[i, j, :] = 1
        else:
            zones[i, j, :] = 0

# high enriched ring
'''
zones[1, :, :] = 2
zones[-2, :, :] = 2
zones[:, 1, :] = 2
zones[:, -2, :] = 2
zones[:, :, 1] = 2
zones[:, :, -2] = 2
'''


# outer reflector
zones[0, :, :] = 2
zones[-1, :, :] = 2
zones[:, 0, :] = 2
zones[:, -1, :] = 2
zones[:, :, 0] = 2
zones[:, :, -1] = 2






# Define widths
"""
Widths are defined as a 3D array of floats, where each float represents the width of a cell in the x, y, and z directions.
The widths are defined as follows:
cell = [x, y, z]

NOTE: The x, y, and z widths must sum to the same value in the other two dimensions across all cells.
"""
dimensions = np.zeros((nx, ny, nz, 3), dtype=float)
dimensions[:, :, :, :] = [8.426,8.426,15.27]


# Arbirary flags
print_sizes = True



# Useful information ------------------------------------------------------

# Suppress warnings
warnings.filterwarnings("ignore")

# Create directory for plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Print sizes of dimensions and widths
if print_sizes:
    # Print size of dimensions
    print("Size of node counts: ", sys.getsizeof(dimensions), " bytes")
    # Print size of widths tensor
    print("Size of widths tensor: ", sys.getsizeof(zones), " bytes")



# Define material properties ------------------------------------------------------
"""
Material properties are defined as a dictionary, where each key is a material name and each value is a dictionary of properties.

The indexes of the properties are as follows:
0: Water
1: Typical PWR Fuel
2: Benchmark Pu239
3: Control Rod

"""

# --- material properties ---
WITHDRAWN = False

'''
# energy group 1: fast
D_list_fast        = [1.13,   1.2627,   2.354]
SigR_list_fast     = [0.0494, 0.02619,  0.1416]
NuSigF_list_fast   = [0.0,    0.008476, 0.29016]
Sig12_list_fast    = [0.0494, 0.014123, 0.04320]

# energy group 2: thermal
D_list_thermal     = [0.16,   0.3543,    3.341]
SigA_list_thermal  = [0.0197, 0.1210,    0.09984]
NuSigF_list_thermal= [0.0,    0.18514,   0.2503392]
Sig21_zero         = [0.0,    0.0,       0.0]
'''

# withdrawn: 2.35, 3.40, 4.45, ref
if WITHDRAWN:
    # energy group 1: fast
    D_list_fast        = [1.43E+00, 1.44E+00, 1.44E+00, 1.40E+00]
    SigR_list_fast     = [2.56E-02, 2.54E-02, 2.54E-02, 2.63E-02]
    NuSigF_list_fast   = [5.64E-03, 6.97E-03, 8.21E-03, 5.87E-03]
    Sig12_list_fast    = [1.68E-02, 1.59E-02, 1.53E-02, 1.77E-02]

    # energy group 2: thermal
    D_list_thermal     = [3.75E-01, 3.75E-01, 3.75E-01, 3.22E-01]
    SigA_list_thermal  = [6.64E-02, 8.43E-02, 1.00E-01, 5.13E-02]
    NuSigF_list_thermal= [1.06E-01, 1.45E-01, 1.79E-01, 7.20E-02]
    Sig21_zero         = [0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00]

# not withdrawn: 2.35, 3.40, 4.45, ref
elif not WITHDRAWN:
    # energy group 1: fast
    D_list_fast        = [1.38E+00, 1.39E+00, 1.39E+00, 1.40E+00]
    SigR_list_fast     = [2.73E-02, 2.69E-02, 2.67E-02, 2.63E-02]
    NuSigF_list_fast   = [5.55E-03, 6.87E-03, 8.09E-03, 5.87E-03]
    Sig12_list_fast    = [1.46E-02, 1.38E-02, 1.31E-02, 1.77E-02]

    # energy group 2: thermal
    D_list_thermal     = [3.79E-01, 3.78E-01, 3.77E-01, 3.22E-01]
    SigA_list_thermal  = [9.59E-02, 1.15E-01, 1.32E-01, 5.13E-02]
    NuSigF_list_thermal= [1.07E-01, 1.46E-01, 1.82E-01, 7.20E-02]
    Sig21_zero         = [0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00]


# pack into nested dict
material_properties = {
    # energy group 1: fast
    0: {
        0: D_list_fast,
        1: SigR_list_fast,
        2: NuSigF_list_fast,
        3: Sig12_list_fast
    },
    # energy group 2: thermal
    1: {
        0: D_list_thermal, 
        1: SigA_list_thermal,
        2: NuSigF_list_thermal,
        3: Sig21_zero,
    },
}

# flux norm
kappa = 200 * 1.6022E-16  # [kJ]

# convergence tol
epsilon = 1e-5

# fission 
nu = 2.5



# Make functions ------------------------------------------------------

# 3-D to 1-D index mapping
def idx(i, j, k, nx, ny):
    return i + nx * (j + ny * k)




# Initialization ------------------------------------------------------

# Total number of nodes
N = nx * ny * nz  # total number of nodes

# Initialize arrays
A_fast    = np.zeros((N, N))
A_thermal = np.zeros((N, N))
S_fast    = np.zeros((N, N))
S_thermal = np.zeros((N, N))
Sig12_mat = np.zeros((N, N))



# --- build diffusion‐removal (A) and fission/scatter (S) matrices with per‐cell dx,dy,dz ---
from scipy.sparse import csr_matrix

# assume idx, N, zones and material_properties are already defined, and
# A_fast, A_thermal, S_fast, S_thermal, Sig12_mat = np.zeros((N,N)) etc.

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):

            # --- global index ---
            i = idx(ix, iy, iz, nx, ny)
            mat = zones[ix, iy, iz]

            # --- local cell widths ---
            dx = dimensions[ix, iy, iz, 0]
            dy = dimensions[ix, iy, iz, 1]
            dz = dimensions[ix, iy, iz, 2]

            # --- local material properties ---
            D_fast = material_properties[0][0][mat]
            D_thermal = material_properties[1][0][mat]
            SigR_fast = material_properties[0][1][mat]
            SigR_thermal = material_properties[1][1][mat]
            NuSigF_fast = material_properties[0][2][mat]
            NuSigF_thermal = material_properties[1][2][mat]
            Sig12_fast = material_properties[0][3][mat]
            Sig21_thermal = material_properties[1][3][mat]

            # --- initilatize central values ---
            A_fast_center    = 0.0
            A_thermal_center = 0.0



            # --- x‐direction ---
            # left boundary
            if ix == 0:
                A_fast_center += D_fast / dx**2
                A_thermal_center += D_thermal / dx**2
            # left face
            else:
                mat = zones[ix-1, iy, iz]
                D_fast_x = material_properties[0][0][mat]
                D_thermal_x = material_properties[1][0][mat]
                SigR_fast_x = material_properties[0][1][mat]
                SigR_thermal_x = material_properties[1][1][mat]
                NuSigF_fast_x = material_properties[0][2][mat]
                NuSigF_thermal_x = material_properties[1][2][mat]
                Sig12_fast_x = material_properties[0][3][mat]
                Sig21_thermal_x = material_properties[1][3][mat]
                dx_x = dimensions[ix-1, iy, iz, 0]
                i_x = idx(ix-1, iy, iz, nx, ny)

                A_fast_center += D_fast_x / (2*dx_x**2)
                A_thermal_center += D_thermal_x / (2*dx_x**2)
                A_fast[i, i_x] = -D_fast_x / (dx_x**2)
                A_thermal[i, i_x] = -D_thermal_x / (dx_x**2)

            # right boundary
            if ix == nx-1:
                A_fast_center += D_fast / dx**2
                A_thermal_center += D_thermal / dx**2
            # right face
            else:
                mat = zones[ix+1, iy, iz]
                D_fast_x = material_properties[0][0][mat]
                D_thermal_x = material_properties[1][0][mat]
                SigR_fast_x = material_properties[0][1][mat]
                SigR_thermal_x = material_properties[1][1][mat]
                NuSigF_fast_x = material_properties[0][2][mat]
                NuSigF_thermal_x = material_properties[1][2][mat]
                Sig12_fast_x = material_properties[0][3][mat]
                Sig21_thermal_x = material_properties[1][3][mat]
                dx_x = dimensions[ix+1, iy, iz, 0]
                i_x = idx(ix+1, iy, iz, nx, ny)

                A_fast_center += D_fast_x / (2*dx_x**2)
                A_thermal_center += D_thermal_x / (2*dx_x**2)
                A_fast[i, i_x] = -D_fast_x / (dx_x**2)
                A_thermal[i, i_x] = -D_thermal_x / (dx_x**2)



            # --- y‐direction ---
            # back boundary
            if iy == 0:
                A_fast_center += D_fast / dy**2
                A_thermal_center += D_thermal / dy**2
            # back face
            else:
                mat = zones[ix, iy-1, iz]
                D_fast_y = material_properties[0][0][mat]
                D_thermal_y = material_properties[1][0][mat]
                SigR_fast_y = material_properties[0][1][mat]
                SigR_thermal_y = material_properties[1][1][mat]
                NuSigF_fast_y = material_properties[0][2][mat]
                NuSigF_thermal_y = material_properties[1][2][mat]
                Sig12_fast_y = material_properties[0][3][mat]
                Sig21_thermal_y = material_properties[1][3][mat]
                dy_y = dimensions[ix, iy-1, iz, 0]
                i_y = idx(ix, iy-1, iz, nx, ny)

                A_fast_center += D_fast_y / (2*dy_y**2)
                A_thermal_center += D_thermal_y / (2*dy_y**2)
                A_fast[i, i_y] = -D_fast_y / (dy_y**2)
                A_thermal[i, i_y] = -D_thermal_y / (dy_y**2)

            # front boundary
            if iy == ny-1:
                A_fast_center += D_fast / dy**2
                A_thermal_center += D_thermal / dy**2
            # front face
            else:
                mat = zones[ix, iy+1, iz]
                D_fast_y = material_properties[0][0][mat]
                D_thermal_y = material_properties[1][0][mat]
                SigR_fast_y = material_properties[0][1][mat]
                SigR_thermal_y = material_properties[1][1][mat]
                NuSigF_fast_y = material_properties[0][2][mat]
                NuSigF_thermal_y = material_properties[1][2][mat]
                Sig12_fast_y = material_properties[0][3][mat]
                Sig21_thermal_y = material_properties[1][3][mat]
                dy_y = dimensions[ix, iy+1, iz, 0]
                i_y = idx(ix, iy+1, iz, nx, ny)

                A_fast_center += D_fast_y / (2*dy_y**2)
                A_thermal_center += D_thermal_y / (2*dy_y**2)
                A_fast[i, i_y] = -D_fast_y / (dy_y**2)
                A_thermal[i, i_y] = -D_thermal_y / (dy_y**2)



            # --- z‐direction ---
            # up boundary
            if iz == 0:
                A_fast_center += D_fast / dz**2
                A_thermal_center += D_thermal / dz**2
            # up face
            else:
                mat = zones[ix, iy, iz-1]
                D_fast_z = material_properties[0][0][mat]
                D_thermal_z = material_properties[1][0][mat]
                SigR_fast_z = material_properties[0][1][mat]
                SigR_thermal_z = material_properties[1][1][mat]
                NuSigF_fast_z = material_properties[0][2][mat]
                NuSigF_thermal_z = material_properties[1][2][mat]
                Sig12_fast_z = material_properties[0][3][mat]
                Sig21_thermal_z = material_properties[1][3][mat]
                dz_z = dimensions[ix, iy, iz-1, 0]
                i_z = idx(ix, iy, iz-1, nx, ny)

                A_fast_center += D_fast_z / (2*dz_z**2)
                A_thermal_center += D_thermal_z / (2*dz_z**2)
                A_fast[i, i_z] = -D_fast_z / (dz_z**2)
                A_thermal[i, i_z] = -D_thermal_z / (dz_z**2)

            # down boundary
            if iz == nz-1:
                A_fast_center += D_fast / dz**2
                A_thermal_center += D_thermal / dz**2
            # down face
            else:
                mat = zones[ix, iy, iz+1]
                D_fast_z = material_properties[0][0][mat]
                D_thermal_z = material_properties[1][0][mat]
                SigR_fast_z = material_properties[0][1][mat]
                SigR_thermal_z = material_properties[1][1][mat]
                NuSigF_fast_z = material_properties[0][2][mat]
                NuSigF_thermal_z = material_properties[1][2][mat]
                Sig12_fast_z = material_properties[0][3][mat]
                Sig21_thermal_z = material_properties[1][3][mat]
                dz_z = dimensions[ix, iy, iz+1, 0]
                i_z = idx(ix, iy, iz+1, nx, ny)

                A_fast_center += D_fast_z / (2*dz_z**2)
                A_thermal_center += D_thermal_z / (2*dz_z**2)
                A_fast[i, i_z] = -D_fast_z / (dz_z**2)
                A_thermal[i, i_z] = -D_thermal_z / (dz_z**2)
            
            A_fast[i, i] = 2*A_fast_center + SigR_fast
            A_thermal[i, i] = 2*A_thermal_center + SigR_thermal

            
            # --- fission and scattering terms ---
            # fast fission
            S_fast[i, i] = NuSigF_fast
            # thermal fission
            S_thermal[i, i] = NuSigF_thermal

            # scatter matrix
            Sig12_mat[i, i] = Sig12_fast


# convert to sparse if desired
A_fast    = csr_matrix(A_fast)
A_thermal = csr_matrix(A_thermal)
S_fast    = csr_matrix(S_fast)
S_thermal = csr_matrix(S_thermal)
Sig12_mat = csr_matrix(Sig12_mat)

# --- power iteration for two‐group k‐eigenvalue ---
# initial guess
phi1 = np.ones(N)     # fast flux
phi2 = np.ones(N)     # thermal flux
k    = 1.0
tol  = epsilon
diff = 1e9
it   = 0

start = time.time()
while diff > tol:
    it += 1
    # compute sources
    src1 = S_fast.dot(phi1) + S_thermal.dot(phi2)    # fission into fast
    src2 = Sig12_mat.dot(phi1)                       # scatter into thermal

    # solve for new fluxes
    phi1_new = spsolve(A_fast,    src1) / k
    phi2_new = spsolve(A_thermal, src2)

    # update k via ratio of new/old total fission source
    tot_old = src1.sum()
    src1_new = S_fast.dot(phi1_new) + S_thermal.dot(phi2_new)
    k_new   = k * (src1_new.sum() / tot_old)

    # convergence metric
    diff = np.linalg.norm(phi1_new - phi1) + np.linalg.norm(phi2_new - phi2)

    # prepare next iter
    phi1, phi2, k = phi1_new, phi2_new, k_new

elapsed = time.time() - start

# --- print results ---
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Converged in {it} iterations → keff = {k:.6f}")


# --- reshape to 3D and visualize central slice ---
phi1_3d = phi1.reshape((nx, ny, nz))
phi2_3d = phi2.reshape((nx, ny, nz))

# pick the middle z‐plane
mz = nz // 2

plt.figure()
plt.imshow(phi1_3d[:, :, mz], origin='lower', cmap='viridis', 
           extent=[0, nx, 0, ny])
plt.colorbar(label='Fast Flux')
plt.title(f'Fast Flux (z={mz})')
plt.xlabel('i'); plt.ylabel('j')

plt.figure()
plt.imshow(phi2_3d[:, :, mz], origin='lower', cmap='viridis',
           extent=[0, nx, 0, ny])
plt.colorbar(label='Thermal Flux')
plt.title(f'Thermal Flux (z={mz})')
plt.xlabel('i'); plt.ylabel('j')

plt.show()


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Midplane slice
z_mid = nz // 2
slice_2d = zones[:, :, z_mid]

# Custom color map: RGBA
cmap = ListedColormap([

    # material 0: highly enriched fuel (yellow)
    (1.0, 1.0, 0.0, 1.0),  # Yellow
    # material 1: med enriched fuel (orange)
    (1.0, 0.5, 0.0, 1.0),  # Orange
    # material 2: low enriched fuel (red)
    (1.0, 0.0, 0.0, 1.0),  # Red
    # material 4: reflector (blue)
    #(0.0, 0.0, 1.0, 0.3),  # Blue

])

# Plot the slice
plt.figure(figsize=(6, 6))
plt.imshow(slice_2d.T, origin='lower', cmap=cmap, extent=[0, nx, 0, ny])
plt.colorbar(label='Material ID', ticks=[0, 1, 2])
plt.clim(-0.5, 2.5)
plt.title(f'Core Material Layout (z = {z_mid})')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.tight_layout()
plt.show()
