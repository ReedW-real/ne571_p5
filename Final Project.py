# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:15:37 2025

@author: kateh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
import time

start_time = time.time()
print('Bare core critical dimensions = 83x83x83')
print()
print('With reflector critical at 64 cm core and 1 cm reflector')
print('With reflector critical at 59 cm core and 3 cm reflector')
print('With reflector critical at 58 cm core and 4 cm reflector')
print('With reflector critical at 57 cm core and 5 cm reflector')
print('With reflector critical at 56 cm core and 6 cm reflector')
print('With reflector critical at 56.5 cm core and 7 cm reflector')
print('With reflector critical at 58 cm core and 10 cm reflector')
print()
print('Script run time for 15 nodes is 3.462294816970825 s')
print('Script run time for 30 nodes is 213.86379265785217 s')
print('Script run time for 36 nodes is 771.1584618091583 s')

# Define region sizes (cm)
wx1 = 49.7 # Width of region 1 (core) x direction (cm)
wy1 = 49.7  # Width of region 1 y direction (cm)
wz1 = 49.7  # Width of region 1 z direction (cm)
wx2 = 2    # Width of region 2 (reflector) x direction (cm)
wy2 = 2    # Width of region 2 y direction (cm)
wz2 = 2    # Width of region 2 z direction (cm)

# Define number of nodes in each region
nx1 = 8  # Nodes in First Region x direction (core)
ny1 = 8  # Nodes in First Region y direction (core)
nz1 = 8  # Nodes in First Region z direction (core)
nx2 = 2  # Nodes in Second Region x direction (reflector)
ny2 = 2  # Nodes in Second Region y direction (reflector)
nz2 = 2  # Nodes in Second Region z direction (reflector)

# Calculate total width of all regions
print('Total width of all regions x direction:', wx1 + wx2, 'cm')
print()
print('Total width of all regions y direction:', wy1 + wy2, 'cm')
print()
print('Total width of all regions z direction:', wz1 + wz2, 'cm')
print()

# Define Interface nodes
print('Interface nodes in x: ')
print('Node', nx1, '@', wx1, 'cm')
print('Node', nx1 + nx2, '@', wx1 + wx2, 'cm')

print('Interface nodes in y: ')
print('Node', ny1, '@', wy1, 'cm')
print('Node', ny1 + ny2, '@', wy1 + wy2, 'cm')

print('Interface nodes in z: ')
print('Node', nz1, '@', wz1, 'cm')
print('Node', nz1 + nz2, '@', wz1 + wz2, 'cm')

#%% Defining Variables
# Core material properties
nusigmaf1 = 0.008476
nusigmaf2 = 0.0
nusigmafth1 = 0.18514
nusigmafth2 = 0.0  # nu*sigmaf for flux 2 thermal
Df1 = 1.2627  # Diffusion coefficient for fast flux
Df2 = 1.13
Dth1 = 0.3543
Dth2 = 0.16
SigmaRf1 = 0.02619
SigmaRf2 = 0.0494  # Removal cross section for fast flux
SigmaAf1 = 0.01207
SigmaAf2 = 0.0004  # Absorption Coefficient of fast flux
SigmaAth1 = 0.1210
SigmaAth2 = 0.0197  # Absorption coefficient of thermal flux
SigmaS12c = SigmaRf1 - SigmaAf1
SigmaS12r = SigmaRf2

Sigmaff=0.003320
Sigmafth=0.07537

# Core material grid
core_mat_file = np.genfromtxt('bwr.example.csv',delimiter=',',skip_header =13, skip_footer =24)
core_mat=core_mat_file[:,1:9]
nan_mask = np.isnan(core_mat)
core_mat[nan_mask] = core_mat.T[nan_mask]
core_3d = np.repeat(core_mat[:, :, np.newaxis], nz1, axis=2)

core_and_ref = np.pad(core_3d, pad_width=nz2, mode='constant', constant_values=8)

#0 =  
#1 = fuel
# 

# Reactor power and energy per fission
P = 3000  # Power in MWth
Er = 200 * (1.602e-13)  # Usable energy per fission (J)

# Initial guess for Flux, k-eff, and Source

distx1 = wx1 / (nx1-1) if nx1 > 0 else 0  # Distance between each node in region 1
disty1 = wy1 / (ny1-1) if ny1 > 0 else 0  # Distance between each node in region 1
distz1 = wz1 / (nz1-1) if nz1 > 0 else 0  # Distance between each node in region 1

# Only calculate distances for region 2 (reflector) if nx2, ny2, nz2 > 0
if nx2 > 0 and wx2 > 0:
    distx2 = wx2 / (nx2-1)  # Distance between each node in region 2
else:
    distx2 = 0  # Set to 0 or any other placeholder value

if ny2 > 0 and wy2 > 0:
    disty2 = wy2 / (ny2-1)  # Distance between each node in region 2
else:
    disty2 = 0  # Set to 0 or any other placeholder value

if nz2 > 0 and wz2 > 0:
    distz2 = wz2 / (nz2-1)  # Distance between each node in region 2
else:
    distz2 = 0  # Set to 0 or any other placeholder value
    
#%% Reflector Savings
# For maximum reflector savings, the reflector savings can be approximated
# with the D_core * L_reflector / D_reflector per Weston Stacey's
# Nuclear Reactor Physics 
# L^2 = D/sigma_abs per Weston Stacy
L_thr = np.sqrt(Dth2 / SigmaAth2);
L_fr    = np.sqrt(Df2 / SigmaAf2);

reflector_savings = (Dth1) * (L_thr) / (Dth2)

reflector_savings_both = (Dth1 + Df1) * (L_thr + L_fr) / (Dth2 + Df2)
print(reflector_savings_both)
print(reflector_savings)
    
#%% Fast Flux
#%% Fast Core Interior Nodes
#east=west, north=south, up=down
central_f=[]
east_f=[]
west_f=[]
north_f=[]
south_f=[]
up_f=[]
down_f=[]

#Fast Core

central_equ_fc=2*Df1/(distx1**2) + 2*Df1/(disty1**2) + 2*Df1/(distz1**2) + SigmaRf1
east_equ_fc=-Df1/(distx1**2)
west_equ_fc=-Df1/(distx1**2)
north_equ_fc=-Df1/(disty1**2)
south_equ_fc=-Df1/(disty1**2)
up_equ_fc=-Df1/(distz1**2)
down_equ_fc=-Df1/(distz1**2)

# Reflector Fast

if nx2 > 0 and ny2 > 0 and nz2 > 0:  # Check if the reflector region exists
    central_equ_fr=2*Df2/(distx2**2) + 2*Df2/(disty2**2) + 2*Df2/(distz2**2) + SigmaRf2
    east_equ_fr=-Df2/(distx2**2)
    west_equ_fr=-Df2/(distx2**2)
    north_equ_fr=-Df2/(disty2**2)
    south_equ_fr=-Df2/(disty2**2)
    up_equ_fr=-Df2/(distz2**2)
    down_equ_fr=-Df2/(distz2**2)
else:
    central_equ_fr= None   
    east_equ_fr= None   
    west_equ_fr= None   
    north_equ_fr= None   
    south_equ_fr= None   
    up_equ_fr= None   
    down_equ_fr= None    

print("Hello Dr. Maldonado")

#fill arrays based on 3d location
# Fill arrays based on 3D location for fast flux

for k in range(nz1 + 2 * nz2):
    for j in range(ny1 + 2 * ny2):
        for i in range(nx1 + 2 * nx2):
            if core_and_ref[i, j, k] == 1:
                central_f = np.append(central_f, central_equ_fc)
            elif core_and_ref[i, j, k] == 2:
                central_f = np.append(central_f, central_equ_fr)
            
            if i == nx1 + 2 * nx2 - 1:  # End node in the x-direction
                east_f = np.append(east_f, 0)
            else:
                if i < nx1 + 2 * nx2 - 1:  # End node in the x-direction
                    if core_and_ref[i+1, j, k] == 1:
                        east_f = np.append(east_f, east_equ_fc)
                    elif core_and_ref[i+1, j, k] == 2:
                        east_f = np.append(east_f, east_equ_fr)
            
            if i == 0:  # Start node in the x-direction
                west_f = np.append(west_f, 0)
            else:
                if core_and_ref[i-1, j, k] == 1:
                    west_f = np.append(west_f, west_equ_fc)
                elif core_and_ref[i-1, j, k] == 2:
                    west_f = np.append(west_f, west_equ_fr)
            
            if j == ny1 + 2 * ny2 - 1:  # End node in the y-direction
                north_f = np.append(north_f, 0)
            else:
                if j < nx1 + 2 * nx2 - 1:  # Ensure i+1 is within bounds
                    if core_and_ref[i,j+1,k] == 1:
                        north_f=np.append(north_f,north_equ_fc)
                    elif core_and_ref[i,j+1,k] == 2:
                        north_f=np.append(north_f,north_equ_fr)
            
            if j == 0:  # Start node in the y-direction
                south_f = np.append(south_f, 0)
            else:
                if core_and_ref[i, j-1, k] == 1:
                    south_f = np.append(south_f, south_equ_fc)
                elif core_and_ref[i, j-1, k] == 2:
                    south_f = np.append(south_f, south_equ_fr)
            
            if k == nz1 + 2 * nz2 - 1:  # End node in the z-direction
                up_f = np.append(up_f, 0)
            else:
                if k < nx1 + 2 * nx2 - 1:  # Ensure i+1 is within bounds
                    if core_and_ref[i,j,k+1] == 1:
                        up_f=np.append(up_f,up_equ_fc)
                    elif core_and_ref[i,j,k+1] == 2:
                        up_f=np.append(up_f,up_equ_fr)
            
            if k == 0:  # Start node in the z-direction
                down_f = np.append(down_f, 0)
            else:
                if core_and_ref[i, j, k-1] == 1:
                    down_f = np.append(down_f, down_equ_fc)
                elif core_and_ref[i, j, k-1] == 2:
                    down_f = np.append(down_f, down_equ_fr)
                

#%% Make Fast Flux Arrays               
print(len(central_f), len(east_f), len(west_f), len(north_f), len(south_f), len(up_f), len(down_f))  
    
#take arrays and input into a matrix to build A matrix
matrix_fast=np.diag(central_f)+ \
    np.diag(east_f[:-1],k=1)+ \
    np.diag(west_f[1:],k=-1)+ \
    np.diag(north_f[:-(nx1+2*nx2)],k=(nx1+2*nx2))+ \
    np.diag(south_f[(nx1+2*nx2):],k=-(nx1+2*nx2))+ \
    np.diag(up_f[:-(nx1+2*nx2)*(ny1+2*ny2)],k=(nx1+2*nx2)*(ny1+2*ny2))+ \
    np.diag(down_f[(nx1+2*nx2)*(ny1+2*ny2):],k=-(nx1+2*nx2)*(ny1+2*ny2))
   
plt.spy(matrix_fast)
plt.show()

#%% Thermal Flux
#%% Thermal Core Interior Nodes
#east=west, north=south, up=down
central_thc=[]
east_thc=[]
west_thc=[]
north_thc=[]
south_thc=[]
up_thc=[]
down_thc=[]

central_equ_thc=2*Dth1/(distx1**2) + 2*Dth1/(disty1**2) + 2*Dth1/(distz1**2) + SigmaAth1
east_equ_thc=-Dth1/(distx1**2)
west_equ_thc=-Dth1/(distx1**2)
north_equ_thc=-Dth1/(disty1**2)
south_equ_thc=-Dth1/(disty1**2)
up_equ_thc=-Dth1/(distz1**2)
down_equ_thc=-Dth1/(distz1**2)

if nx2 > 0 and ny2 > 0 and nz2 > 0:  # Check if the reflector region exists
    central_equ_thr=2*Dth2/(distx2**2) + 2*Dth2/(disty2**2) + 2*Dth2/(distz2**2) + SigmaAth2
    east_equ_thr=-Dth2/(distx2**2)
    west_equ_thr=-Dth2/(distx2**2)
    north_equ_thr=-Dth2/(disty2**2)
    south_equ_thr=-Dth2/(disty2**2)
    up_equ_thr=-Dth2/(distz2**2)
    down_equ_thr=-Dth2/(distz2**2)
else:
    central_equ_thr= None
    east_equ_thr= None
    west_equ_thr= None
    north_equ_thr= None
    south_equ_thr= None
    up_equ_thr= None
    down_equ_thr= None
    
#fill arrays based on 3d location
for k in range(nz1 + 2 * nz2):
    for j in range(ny1 + 2 * ny2):
        for i in range(nx1 + 2 * nx2):
            if core_and_ref[i, j, k] == 1:
                central_thc = np.append(central_thc, central_equ_thc)
            elif core_and_ref[i, j, k] == 2:
                central_thc = np.append(central_thc, central_equ_thr)
            
            if i == nx1 + 2 * nx2 - 1:  # End node in the x-direction
                east_thc = np.append(east_thc, 0)
            else:
                if core_and_ref[i+1, j, k] == 1:
                    east_thc = np.append(east_thc, east_equ_thc)
                elif core_and_ref[i+1, j, k] == 2:
                    east_thc = np.append(east_thc, east_equ_thr)
            
            if i == 0:  # Start node in the x-direction
                west_thc = np.append(west_thc, 0)
            else:
                if core_and_ref[i - 1, j, k] == 1:
                    west_thc = np.append(west_thc, west_equ_thc)
                elif core_and_ref[i - 1, j, k] == 2:
                    west_thc = np.append(west_thc, west_equ_thr)
            
            if j == ny1 + 2 * ny2 - 1:  # End node in the y-direction
                north_thc = np.append(north_thc, 0)
            else:
                if core_and_ref[i, j+1, k] == 1:
                    north_thc = np.append(north_thc, north_equ_thc)
                elif core_and_ref[i, j+1, k] == 2:
                    north_thc = np.append(north_thc, north_equ_thr)
            
            if j == 0:  # Start node in the y-direction
                south_thc = np.append(south_thc, 0)
            else:
                if core_and_ref[i, j - 1, k] == 1:
                    south_thc = np.append(south_thc, south_equ_thc)
                elif core_and_ref[i, j - 1, k] == 2:
                    south_thc = np.append(south_thc, south_equ_thr)
            
            if k == nz1 + 2 * nz2 - 1:  # End node in the z-direction
                up_thc = np.append(up_thc, 0)
            else:
                if core_and_ref[i, j, k+1] == 1:
                    up_thc = np.append(up_thc, up_equ_thc)
                elif core_and_ref[i, j, k+1] == 2:
                    up_thc = np.append(up_thc, up_equ_thr)
            
            if k == 0:  # Start node in the z-direction
                down_thc = np.append(down_thc, 0)
            else:
                if core_and_ref[i, j, k - 1] == 1:
                    down_thc = np.append(down_thc, down_equ_thc)
                elif core_and_ref[i, j, k - 1] == 2:
                    down_thc = np.append(down_thc, down_equ_thr)
                

#%% Make Thermal Flux Arrays   
#take arrays and input into a matrix to build A matrix
matrix_thermal=np.diag(central_thc)+ \
    np.diag(east_thc[:-1],k=1)+ \
    np.diag(west_thc[1:],k=-1)+ \
    np.diag(north_thc[:-(nx1+2*nx2)],k=(nx1+2*nx2))+ \
    np.diag(south_thc[(nx1+2*nx2):],k=-((nx1+2*nx2)))+ \
    np.diag(up_thc[:-((nx1+2*nx2)*(ny1+2*ny2))],k=(nx1+2*nx2)*(ny1+2*ny2))+ \
    np.diag(down_thc[(nx1+2*nx2)*(ny1+2*ny2):],k=-(nx1+2*nx2)*(ny1+2*ny2))
   
plt.spy(matrix_thermal)
plt.show()
          
#%% B matrix with diagonal of fission term

fast_1_flux=[]
for k in range(nz1+2*nz2):
    for j in range(ny1+2*ny2):
        for i in range(nx1+2*nx2):
            if core_and_ref[i,j,k] == 1:                
                fast_1_flux=np.append(fast_1_flux,nusigmaf1)
            elif core_and_ref[i,j,k] == 2:
                fast_1_flux=np.append(fast_1_flux,nusigmaf2)
fast_flux1=np.diag(fast_1_flux)

fast_2_flux=[]
for k in range(nz1+2*nz2):
    for j in range(ny1+2*ny2):
        for i in range(nx1+2*nx2):
            if core_and_ref[i,j,k] == 1:                
                fast_2_flux=np.append(fast_2_flux,nusigmafth1)
            elif core_and_ref[i,j,k] == 2:
                fast_2_flux=np.append(fast_2_flux,nusigmafth2)
fast_flux2=np.diag(fast_2_flux)

thermal_flux=[]
for k in range(nz1+2*nz2):
    for j in range(ny1+2*ny2):
        for i in range(nx1+2*nx2):
            if core_and_ref[i,j,k] == 1:                
                thermal_flux=np.append(thermal_flux,SigmaS12c)
            elif core_and_ref[i,j,k] == 2:
                thermal_flux=np.append(thermal_flux,SigmaS12r)
Thermal_flux=np.diag(thermal_flux)

plt.spy(fast_flux1)
plt.show()

flux1 = np.full(((nx1+2*nx2)*(ny1+2*ny2)*(nz1+2*nz2)),1)

flux2 = np.full(((nx1+2*nx2)*(ny1+2*ny2)*(nz1+2*nz2)),1)


k = 1
Sfast=fast_flux1.dot(flux1) +fast_flux2.dot(flux2)
Sthermal = Thermal_flux.dot(flux1)
flxdiff1=1
flxdiff2=1
flxdiff=1

iter=0

while (flxdiff > 0.0001):
    iter=iter+1

    oldflux1=flux1
    oldflux2=flux2
    oldk=k
    oldSfast=Sfast
    oldSthermal=Sthermal

    flux1 = np.dot(np.linalg.inv(matrix_fast), oldSfast) / oldk
    flux2=np.dot(np.linalg.inv(matrix_thermal),oldSthermal)    
    Sfast=fast_flux1.dot(flux1) +fast_flux2.dot(flux2)
    Sthermal = Thermal_flux.dot(flux1)   
    k=oldk*(sum(Sfast)/sum(oldSfast))

    # Calculate RMS difference of flux between successive iterations
    flxdiff1=0
    flxdiff2=0
    for i in range(nx1+2*nx2):
        flxdiff1 = flxdiff1 + (flux1[i]-oldflux1[i])**2
        flxdiff2 = flxdiff2 + (flux2[i]-oldflux2[i])**2
    flxdiff=np.sqrt(flxdiff1)+np.sqrt(flxdiff2)
   
print('numerical keff',k)
#plt.plot(flux1)
plt.plot(flux2)
plt.show()

#%% Normalize Flux
#Power Normalization
Power=1*2.247e+22 #MWh convert to MeV
wf=200 #MeV/fission

power_num_thermal=np.sum(flux2)*distx1*disty1*distz1
power_num_fast=np.sum(flux1)*distx1*disty1*distz1

normalization=Power/(wf*((Sigmaff*power_num_fast)+(Sigmafth*power_num_thermal)))
            
normalized_flux_fast=flux1*normalization
normalized_flux_thermal=flux2*normalization


#reshape flux array into 3D matrix
flux1=normalized_flux_fast.reshape((nx1+2*nx2),(ny1+2*ny2),(nz1+2*nz2))
pad_width = ((1, 1), (1, 1), (1, 1))
flux1_with_boundary = np.pad(flux1, pad_width, mode='constant', constant_values=0)  

flux2=normalized_flux_thermal.reshape((nx1+2*nx2),(ny1+2*ny2),(nz1+2*nz2))
pad_width = ((1, 1), (1, 1), (1, 1))
flux2_with_boundary = np.pad(flux2, pad_width, mode='constant', constant_values=0)   


#%% Plotting
from mpl_toolkits.mplot3d import Axes3D

slice_2d = flux2_with_boundary[nx1+2*nx2 ,: , :]  # This selects a 2D slice

# Plotting the 2D slice
plt.imshow(slice_2d, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Flux Value')  # Adds a color bar for reference
plt.title("Normalized Thermal Flux")
plt.show()

slice_2d = flux1_with_boundary[nx1+2*nx2 ,: , :]  # This selects a 2D slice
# Plotting the 2D slice
plt.imshow(slice_2d, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Flux Value')  # Adds a color bar for reference
plt.title("Normalized Fast Flux")
plt.show()

end_time = time.time()
print("Script run time for", nx1+2*nx2,"nodes is",end_time-start_time, "s")