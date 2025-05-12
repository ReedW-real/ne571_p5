##################################################################
### package imports ##############################################
##################################################################
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import warnings, time, sys, os
# supress da warning
warnings.filterwarnings("ignore")
##################################################################

def checker(nx, ny, nz):
    """ checker pattern with enriched ring and reflector """
    zones = np.zeros((nx, ny, nz), dtype=int)
    zones[:, :, :] = 1

    # checker board pattern of med and low enriched fuel (constant in z, checkerboard in x-y)
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            if (i+j) % 2 == 0:
                zones[i, j, :] = 1
            else:
                zones[i, j, :] = 0
    
    # high enriched ring
    zones[1, :, :], zones[-2, :, :] = 2, 2
    zones[:, 1, :], zones[:, -2, :] = 2, 2
    zones[:, :, 1], zones[:, :, -2] = 2, 2
    
    return zones


def solid_2p25(nx, ny, nz):
    """ core of just 2p25 enriched fuel """
    return np.zeros((nx, ny, nz), dtype=int)  
    
    
def solid_3p35(nx, ny, nz):
    """ core of just 3p35 enriched fuel """
    return np.ones((nx, ny, nz), dtype=int)  


def solid_3p35(nx, ny, nz):
    """ core of just 4p45 enriched fuel """
    zones = 1 + np.ones((nx, ny, nz), dtype=int)  
    
    return zones
    
    
def just_ref(nx, ny, nz):
    """ core of just 4p45 enriched fuel """ 
    return 2 + np.ones((nx, ny, nz), dtype=int) 


def add_reflector(zones):
    """ outer reflector """
    zones[0, :, :], zones[-1, :, :] = 3, 3
    zones[:, 0, :], zones[:, -1, :] = 3, 3
    zones[:, :, 0], zones[:, :, -1] = 3, 3
    return zones


def widths(nx, ny, nz, shape="tetragonal", WITHDRAWN=False):
    """
    Widths are a 3D array of floats, where each float represents the width of a cell in the x, y, and z directions
    cell = [x, y, z]
    NOTE: The x, y, and z widths must sum to the same value in the other two dimensions across all cells.
    """
    dimensions = np.zeros((nx, ny, nz, 3), dtype=float)
    
    dimensions[:, :, :, :] = [8.426, 8.426, 15.27]  if shape == "tetragonal" else dimensions # tetragonal a=b!=c
    dimensions[:, :, :, :] = [8.426, 8.426, 8.426]  if shape == "cubic" else dimensions # cubic a=b=c
    
    if not os.path.exists("plots"): os.makedirs("plots")
    print("Size of node counts: ", sys.getsizeof(dimensions), " bytes\nSize of widths tensor: ", sys.getsizeof(zones), " bytes")
    
    
    if WITHDRAWN:
        """ withdrawn: 2.35, 3.40, 4.45, ref """
        # group 1: fast
        D_list_fast        = [1.43E+00, 1.44E+00, 1.44E+00, 1.40E+00]
        SigR_list_fast     = [2.56E-02, 2.54E-02, 2.54E-02, 2.63E-02]
        NuSigF_list_fast   = [5.64E-03, 6.97E-03, 8.21E-03, 5.87E-03]
        Sig12_list_fast    = [1.68E-02, 1.59E-02, 1.53E-02, 1.77E-02]
        # group 2: thermal
        D_list_thermal     = [3.75E-01, 3.75E-01, 3.75E-01, 3.22E-01]
        SigA_list_thermal  = [6.64E-02, 8.43E-02, 1.00E-01, 5.13E-02]
        NuSigF_list_thermal= [1.06E-01, 1.45E-01, 1.79E-01, 7.20E-02]
        Sig21_zero         = [0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00]
    
    elif not WITHDRAWN:
        """ not withdrawn: 2.35, 3.40, 4.45, ref """
        # group 1: fast
        D_list_fast        = [1.38E+00, 1.39E+00, 1.39E+00, 1.40E+00]
        SigR_list_fast     = [2.73E-02, 2.69E-02, 2.67E-02, 2.63E-02]
        NuSigF_list_fast   = [5.55E-03, 6.87E-03, 8.09E-03, 5.87E-03]
        Sig12_list_fast    = [1.46E-02, 1.38E-02, 1.31E-02, 1.77E-02]
        # group 2: thermal
        D_list_thermal     = [3.79E-01, 3.78E-01, 3.77E-01, 3.22E-01]
        SigA_list_thermal  = [9.59E-02, 1.15E-01, 1.32E-01, 5.13E-02]
        NuSigF_list_thermal= [1.07E-01, 1.46E-01, 1.82E-01, 7.20E-02]
        Sig21_zero         = [0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00]
    
    # pack into nested dict
    material_properties = {
    # group 1: fast
        0: {0: D_list_fast, 1: SigR_list_fast, 2: NuSigF_list_fast, 3: Sig12_list_fast},
    # group 2: thermal
        1: {0: D_list_thermal, 1: SigA_list_thermal, 2: NuSigF_list_thermal, 3: Sig21_zero}
        }
    
    return dimensions, material_properties


def idx(i, j, k, nx, ny):
    """ 3-D to 1-D index mapping """
    return i + nx * (j + ny * k)


def coe1(D, d_):
    """ for coefficient in diffusion removal building"""
    return D / (d_**2)
    
    
def get_props(coordinate):
    """ reduces repetition of these collection of arguments """
    fast_props = [material_properties[0][i][coordinate] for i in range(4)]
    thermal_props = [material_properties[1][i][coordinate] for i in range(4)]
    D_f, SigR_f, NuSigF_f, Sig12_f = fast_props
    D_t, SigR_t, NuSigF_t, Sig21_t = thermal_props
    
    return D_f, D_t, SigR_f, SigR_t, NuSigF_f, NuSigF_t, Sig12_f, Sig21_t
    

def power_norm(fast_source, thermal_source, flux1, flux2, thermal_power=990e+06, wf=2e+08*1.602e-19, nu=2.5):
    """ 
    power normalization function
    
    in:
    - fast_source: fast source matrix
    - thermal_source: thermal source matrix
    - flux1: fast flux
    - flux2: thermal flux
    
    - thermal power is 990 MWth from NRC document
    - wf is energy per fission, units of [J / fission]
    - nu is the fission neutron yield assumed 2.5
    
    out:
    - normalized flux1, flux2
    """
    norm_factor = thermal_power / ((wf / nu) * np.sum(fast_source.dot(flux1) + thermal_source.dot(flux2)))
    return flux1*norm_factor, flux2*norm_factor
    

def builder(nx, ny, nz, dimensions, zones):
    """
    Builds coefficient matrices for a multi-group neutron diffusion solver.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Number of cells in x, y, and z directions
    dimensions : ndarray
        Array of cell dimensions, shape (nx, ny, nz, 3)
    zones : ndarray
        Array of material zone identifiers
        
    Returns:
    --------
    A_fast : sparse matrix
        Fast neutron diffusion operator
    A_thermal : sparse matrix
        Thermal neutron diffusion operator
    S_fast : sparse matrix
        Fast neutron source term
    S_thermal : sparse matrix
        Thermal neutron source term
    Sig12_mat : sparse matrix
        Scattering matrix from fast to thermal group
    """
    # Initialize matrices
    n_total = nx * ny * nz
    A_fast = np.zeros((n_total, n_total))
    A_thermal = np.zeros((n_total, n_total))
    S_fast = np.zeros((n_total, n_total))
    S_thermal = np.zeros((n_total, n_total))
    Sig12_mat = np.zeros((n_total, n_total))
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                # --- global index ---
                i = idx(ix, iy, iz, nx, ny)

                # --- local cell widths ---
                dx, dy, dz = dimensions[ix, iy, iz, 0:3]

                # --- local material properties ---
                D_fast, D_thermal, SigR_fast, SigR_thermal, NuSigF_fast, NuSigF_thermal, Sig12_fast, Sig21_thermal = get_props(zones[ix, iy, iz])

                # --- initialize central values ---
                A_fast_center, A_thermal_center = SigR_fast, SigR_thermal

                # --- x axis ---
                
                # left boundary
                if ix == 0:
                    A_fast_center += coe1(D_fast, dx)
                    A_thermal_center += coe1(D_thermal, dx)
                # left face
                else:
                    D_fast_x, D_thermal_x = get_props(zones[ix-1, iy, iz])[:2]               
                    dx_x = dimensions[ix-1, iy, iz, 0]
                    i_x = idx(ix-1, iy, iz, nx, ny)
                   
                    A_fast_center += coe1(D_fast, dx)
                    A_thermal_center += coe1(D_thermal, dx)
                    A_fast[i, i_x] = -coe1(D_fast, dx)
                    A_thermal[i, i_x] = -coe1(D_thermal, dx)

                # right boundary
                if ix == nx-1:
                    A_fast_center += coe1(D_fast, dx)
                    A_thermal_center += coe1(D_thermal, dx)
                # right face
                else:
                    D_fast_x, D_thermal_x = get_props(zones[ix+1, iy, iz])[:2]               
                    dx_x = dimensions[ix+1, iy, iz, 0]
                    i_x = idx(ix+1, iy, iz, nx, ny)

                    A_fast_center += coe1(D_fast, dx)
                    A_thermal_center += coe1(D_thermal, dx)
                    A_fast[i, i_x] = -coe1(D_fast, dx)
                    A_thermal[i, i_x] = -coe1(D_thermal, dx)

                # --- y‐direction ---
                # back boundary
                if iy == 0:
                    A_fast_center += coe1(D_fast, dy)
                    A_thermal_center += coe1(D_thermal, dy)
                # back face
                else:
                    D_fast_y, D_thermal_y = get_props(zones[ix, iy-1, iz])[:2]               
                    dy_y = dimensions[ix, iy-1, iz, 1]  
                    i_y = idx(ix, iy-1, iz, nx, ny)

                    A_fast_center += coe1(D_fast, dy)
                    A_thermal_center += coe1(D_thermal, dy)
                    A_fast[i, i_y] = -coe1(D_fast, dy)
                    A_thermal[i, i_y] = -coe1(D_thermal, dy)

                # front boundary
                if iy == ny-1:
                    A_fast_center += coe1(D_fast, dy)
                    A_thermal_center += coe1(D_thermal, dy)
                # front face
                else:
                    D_fast_y, D_thermal_y = get_props(zones[ix, iy+1, iz])[:2]               
                    dy_y = dimensions[ix, iy+1, iz, 1]  
                    i_y = idx(ix, iy+1, iz, nx, ny)

                    A_fast_center += coe1(D_fast, dy)
                    A_thermal_center += coe1(D_thermal, dy)
                    A_fast[i, i_y] = -coe1(D_fast, dy)
                    A_thermal[i, i_y] = -coe1(D_thermal, dy)

                # --- z‐direction ---
                # up boundary
                if iz == 0:
                    A_fast_center += coe1(D_fast, dz)
                    A_thermal_center += coe1(D_thermal, dz)
                # up face
                else:
                    D_fast_z, D_thermal_z = get_props(zones[ix, iy, iz-1])[:2]               
                    dz_z = dimensions[ix, iy, iz-1, 2]  
                    i_z = idx(ix, iy, iz-1, nx, ny)

                    A_fast_center += coe1(D_fast, dz)
                    A_thermal_center += coe1(D_thermal, dz)
                    A_fast[i, i_z] = -coe1(D_fast, dz)
                    A_thermal[i, i_z] = -coe1(D_thermal, dz)

                # down boundary
                if iz == nz-1:
                    A_fast_center += coe1(D_fast, dz)
                    A_thermal_center += coe1(D_thermal, dz)
                # down face
                else:
                    D_fast_z, D_thermal_z = get_props(zones[ix, iy, iz+1])[:2]               
                    dz_z = dimensions[ix, iy, iz+1, 2]  
                    i_z = idx(ix, iy, iz+1, nx, ny)

                    A_fast_center += coe1(D_fast, dz)
                    A_thermal_center += coe1(D_thermal, dz)
                    A_fast[i, i_z] = -coe1(D_fast, dz)
                    A_thermal[i, i_z] = -coe1(D_thermal, dz)

                # Add removal terms to diagonal
                A_fast[i, i] = A_fast_center
                A_thermal[i, i] = A_thermal_center
                
                # --- source terms ---
                S_fast[i, i] = NuSigF_fast
                S_thermal[i, i] = NuSigF_thermal
                Sig12_mat[i, i] = Sig12_fast

    # Convert all matrices to sparse format at once
    return map(csr_matrix, [A_fast, A_thermal, S_fast, S_thermal, Sig12_mat])


def power_iteration_k(zones):
    """ power iteration for two‐group k‐eigenvalue """
    phi1 = np.ones(nx*ny*nz)    # fast flux guess
    phi2 = np.copy(phi1)        # thermal flux guess
    k    = 1.0                  # initial k guess
    tol  = 1e-5                 # convergence tolerance 
    diff = 1e9                  # assigns diff variable
    max_iter = 1000             # maximum iterations 
    A_fast, A_thermal, S_fast, S_thermal, Sig12_mat = builder(nx, ny, nz, dimensions, zones) # build matrices
    
    start = time.time()    
    for i in range(max_iter):
        # source terms
        S_fast_phi1 = S_fast.dot(phi1)
        S_thermal_phi2 = S_thermal.dot(phi2)
        src1 = S_fast_phi1 + S_thermal_phi2
        src1_sum = src1.sum()  
        
        # solve flux values
        phi1_new = spsolve(A_fast, src1) / k
        phi2_new = spsolve(A_thermal, Sig12_mat.dot(phi1_new))
        
        # new source terms and k-eigenvalue
        S_fast_phi1_new = S_fast.dot(phi1_new)
        S_thermal_phi2_new = S_thermal.dot(phi2_new)
        src1_new = S_fast_phi1_new + S_thermal_phi2_new
        k_new = k * src1_new.sum() / src1_sum
        
        # Calculate convergence metrics
        k_diff = abs(k - k_new)
        
        # Update values for next iteration
        phi1, phi2, k = phi1_new, phi2_new, k_new
        
        # Check convergence
        if k_diff < tol:
            print(f"Elapsed time: {time.time() - start:.2f} seconds")
            print(f"Converged in {i+1} iterations → keff = {k:.6f}")
            break
        elif i == max_iter-1:
            print(f'Did not converge in {i+1} iterations')
        
    # Normalize power
    phi1, phi2 = power_norm(S_fast, S_thermal, phi1, phi2)
    
    return phi1, phi2, k


def visualize_k(phi1, phi2, k):
    """
    reshape to 3D and visualize central slice
    """
    phi1_3d = phi1.reshape((nx, ny, nz))
    phi2_3d = phi2.reshape((nx, ny, nz))
    mx = nx // 2  # Middle index for x
    my = ny // 2  # Middle index for y
    mz = nz // 2  # Middle index for z
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # X direction plot
    axs[0].plot(range(nx), phi1_3d[:, my, mz], 'b-', label='Fast Flux')
    axs[0].plot(range(nx), phi2_3d[:, my, mz], 'r-', label='Thermal Flux')
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Flux Magnitude')
    axs[0].set_title(f'Flux Along X at Y=Z={mz}')
    axs[0].legend()
    axs[0].grid(True)
    
    # Y direction plot
    axs[1].plot(range(ny), phi1_3d[mx, :, mz], 'b-', label='Fast Flux')
    axs[1].plot(range(ny), phi2_3d[mx, :, mz], 'r-', label='Thermal Flux')
    axs[1].set_xlabel('Y Position')
    axs[1].set_ylabel('Flux Magnitude')
    axs[1].set_title(f'Flux Along Y at X=Z={mz}')
    axs[1].legend()
    axs[1].grid(True)
    
    # Z direction plot
    axs[2].plot(range(nz), phi1_3d[mx, my, :], 'b-', label='Fast Flux')
    axs[2].plot(range(nz), phi2_3d[mx, my, :], 'r-', label='Thermal Flux')
    axs[2].set_xlabel('Z Position')
    axs[2].set_ylabel('Flux Magnitude')
    axs[2].set_title(f'Flux Along Z at X=Y={my}')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.savefig('plots/flux_profiles.png')
    
    # Create 2D contour plots of the middle slices
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fast flux - XY plane
    im1 = axs[0, 0].imshow(phi1_3d[:, :, mz].T, cmap='jet', origin='lower')
    axs[0, 0].set_title('Fast Flux - XY Plane (Z=middle)')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axs[0, 0])  

    # Fast flux - XZ plane
    im2 = axs[0, 1].imshow(phi1_3d[:, my, :].T, cmap='jet', origin='lower')
    axs[0, 1].set_title('Fast Flux - XZ Plane (Y=middle)')
    axs[0, 1].set_xlabel('X Position')
    axs[0, 1].set_ylabel('Z Position')
    plt.colorbar(im2, ax=axs[0, 1]) 

    # Fast flux - YZ plane
    im3 = axs[0, 2].imshow(phi1_3d[mx, :, :].T, cmap='jet', origin='lower')
    axs[0, 2].set_title('Fast Flux - YZ Plane (X=middle)')
    axs[0, 2].set_xlabel('Y Position')
    axs[0, 2].set_ylabel('Z Position')
    plt.colorbar(im3, ax=axs[0, 2]) 

    # Thermal flux - XY plane
    im4 = axs[1, 0].imshow(phi2_3d[:, :, mz].T, cmap='jet', origin='lower')
    axs[1, 0].set_title('Thermal Flux - XY Plane (Z=middle)')
    axs[1, 0].set_xlabel('X Position')
    axs[1, 0].set_ylabel('Y Position')
    plt.colorbar(im4, ax=axs[1, 0]) 

    # Thermal flux - XZ plane
    im5 = axs[1, 1].imshow(phi2_3d[:, my, :].T, cmap='jet', origin='lower')
    axs[1, 1].set_title('Thermal Flux - XZ Plane (Y=middle)')
    axs[1, 1].set_xlabel('X Position')
    axs[1, 1].set_ylabel('Z Position')
    plt.colorbar(im5, ax=axs[1, 1]) 

    # Thermal flux - YZ plane
    im6 = axs[1, 2].imshow(phi2_3d[mx, :, :].T, cmap='jet', origin='lower')
    axs[1, 2].set_title('Thermal Flux - YZ Plane (X=middle)')
    axs[1, 2].set_xlabel('Y Position')
    axs[1, 2].set_ylabel('Z Position')
    plt.colorbar(im6, ax=axs[1, 2])  
    
    plt.savefig('plots/flux_contours.png')
    
    # Display k-effectivez
    print(f"Final k-effective: {k:.6f}")
    
    return None


def visualize_materials(zones):
    """
    Visualizes the material distribution in a 3D array at its midplane using matplotlib.

    Parameters:
    - zones: A 3D numpy array of integers representing material types.
    """
    # Get the dimensions of the zones array
    nx, ny, nz = zones.shape

    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    
    x_midplane_slice = zones[mid_x, :, :]
    y_midplane_slice = zones[:, mid_y, :]
    z_midplane_slice = zones[:, :, mid_z]

    # Define a custom color map.  Matplotlib colormaps are defined
    # as a list of tuples, where each tuple is (r, g, b).
    #
    # 0: low enrichment, 2.25 w/o    --> Blue
    # 1: med enrichment, 3.35 w/o    --> Green
    # 2: upp enrichment, 4.45 w/o    --> Red
    # 3: reflector                  --> Gray
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0.5, 0.5, 0.5)] # RGB for Blue, Green, Red, Gray
    cmap = plt.cm.colors.ListedColormap(colors)

    # X MIDPLANE 
    plt.figure(figsize=(8, 8), layout='constrained') 
    plt.imshow(x_midplane_slice, cmap=cmap)  
    plt.colorbar(ticks=np.arange(4), label='Material ID')
    plt.xlabel('Z-axis'), plt.ylabel('Y-axis')
    plt.title('Material Distribution at X Midplane')
    
    # Y MIDPLANE
    plt.figure(figsize=(8, 8), layout='constrained')  
    plt.imshow(y_midplane_slice, cmap=cmap) 
    plt.colorbar(ticks=np.arange(4), label='Material ID')
    plt.xlabel('Z-axis'), plt.ylabel('X-axis')
    plt.title('Material Distribution at Y Midplane')
    
    # Z MIDPLANE
    plt.figure(figsize=(8, 8), layout='constrained')  
    plt.imshow(z_midplane_slice, cmap=cmap)  
    plt.colorbar(ticks=np.arange(4), label='Material ID')  
    plt.xlabel('Y-axis'), plt.ylabel('X-axis')
    plt.title('Material Distribution at Z Midplane')
    
    return None

if __name__ == "__main__":
    ### Define node counts ###
    """
    Node counts are integers where each represents the number of nodes in x, y, and z directions.
    nx/y/z: Number of nodes in the x/y/z direction
    """
    nx, ny, nz = 13,13,13
    nodes = [nx, ny, nz]
    
    # Define materials
    """
    Materials are defined as a 3D array of integers, where each integer represents a different material.
    The materials are defined as follows:
    0: low enrichment, 2.25 w/o 
    1: med enrichment, 3.35 w/o
    2: upp enrichment, 4.45 w/o
    3: reflector
    
    select fuel pattern though commenting
    """
    ## zones
    
    " bare 2.25 w/o enriched u235 unreflected "
    zones = solid_2p25(*nodes)
    
    " bare 3.35 w/o enriched u235 unreflected "
    #zones = solid_3p35(*nodes)
    
    " bare 4.45 w/o enriched u235 unreflected "
    #zones = solid_4p45(*nodes)
    
    " bare reflector "
    #zones = just_ref(*nodes)
    
    " reactor no coolant "
    #zones = add_reflector(checker(*nodes))
    
    ## dimensions
    
    " for tetragonal shape "
    #dimensions, material_properties = widths(*nodes) 
    
    " for cubic shape "
    dimensions, material_properties = widths(*nodes, shape="cubic") 
    
    ## run
    
    " visualize the midplane views of the core material model "
    visualize_materials(zones)
    
    " power iteration for k "
    phi1, phi2, k = power_iteration_k(zones)
    
    " visualize power iteration output"
    visualize_k(phi1, phi2, k)
    
    ## plot
    
    " show everything all at once @ end "
    plt.show()
       
    
    

    

##################################################################
### end ##########################################################
##################################################################
