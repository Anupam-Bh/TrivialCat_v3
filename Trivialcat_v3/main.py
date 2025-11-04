## Main
import matplotlib.cm as cm
import os
from matplotlib.collections import LineCollection
import pdb
import time
import multiprocessing
from multiprocessing import Process
import json
import sys
import pandas as pd
import ast
from src import *
# from pymatgen.io.vasp import outputs
# from pymatgen.core import Structure
# from pymatgen.analysis.graphs import StructureGraph
# from pymatgen.analysis.local_env import NearNeighbors
# import pymatgen.analysis.local_env
# import subprocess
# import tensorflow as tf
# import keras
# from tensorflow.keras.utils import plot_model
# import cv2
# import networkx as nx
# import numpy as np
# import re, scipy
# import matplotlib.pyplot as plt
# import itertools, tqdm




## Parameters ## Read data files
start = time.time()
exe_model_path = os.path.abspath('external/')
filepath = os.path.abspath(sys.argv[1])
mp_id = sys.argv[1].split('/')[-1]
print(mp_id)
#filepath = os.path.abspath('data/CoSn')
#exe_model_path = '/mnt/c/Users/a97824ab/OneDrive - The University of Manchester/Desktop/Research/Wavecar_parsing_new/updated_Wavecar_metric_codes/'
#model_path = r'/mnt/c/Users/a97824ab/OneDrive - The University of Manchester/Desktop/Research/Wavecar_parsing_new/model.keras'
#filepath = os.path.abspath('/mnt/c/Users/a97824ab/OneDrive - The University of Manchester/Desktop/Research/Wavecar_parsing_new/updated_Wavecar_metric_codes/CoSn')
os.chdir(filepath)
flatness_threshold = 0.5
prediction_bandwidth = 2
n_samp_pts =  21
hopping_cutoff = 4 ## Angstrom
core_rad = 0.25   ### Assume 0.2*atom rad is core radius where density calculation is incorrect
charge_density_retention_threshold = 0.10  ## 15% charge density retention can identify delocalized flat bands
destructive_interference_threshold = 0.2   ##  sum(wavefunction)/sum(abs(wavefunction)) <20% identifies as destructive interference


### Extract wavefunctions from Wavecars as plane wave coefficients
wavecar, Fouriergrid , fermi = get_wavecar_band_data(exe_model_path)    # [-1:1] eV for bandwidth=2

## Get Emax, Emin, sublattice from previous calculation results
flatness_data = pd.read_csv(exe_model_path+'/Compound_flatness_score_segments.csv')
sublattice_data = pd.read_csv(exe_model_path+'/Flat_band_sublattices.csv')
segments = ast.literal_eval(flatness_data.loc[flatness_data['MP-ID'] == mp_id]['Flat segments'].tolist()[0])
Emin = (2 - max(segments))*0.5
Emax = (3 - max(segments))*0.5
sublattice_elem = sublattice_data.loc[sublattice_data['MP-ID'] == mp_id]['sublattice_element'].tolist()[0]
print(f'Emin: {Emin}, Emax: {Emax}, Sublattice element: {sublattice_elem}')

# ## create bandstructure, predict flatband ## Get Emax, Emin, sublattice
# predicted, Emin, Emax = plot_bands_and_predict(exe_model_path, wavecar, fermi, prediction_bandwidth, flatness_threshold)
# sublattice_elem = get_sublattice(Emax+fermi,Emin+fermi,fermi,sublattice_selection = 'sumdensity')   ## Options 'maxdensity/sumdensity'
## Create graph for sublattice element or the full lattice (for full lattice, sublattice_elem = None )***** 
structure_graph,graph = Make_graph(n_samp_pts, hopping_cutoff = hopping_cutoff, sublattice_element = str(sublattice_elem))   ## get coordinates in graph.edges   
end = time.time()
print('time taken for reading wavefunction and predict flatness',(end-start))


### Pool for multiprocessing: real space wavefunction calculation
print("Number of cpu : ", multiprocessing.cpu_count())
num_process_use = multiprocessing.cpu_count()
# num_process_use = 24
print(f"We are using {num_process_use} cpus")
if __name__ == "__main__":     ## wrapping in main for multiprocessing
    start = time.time()
    argument_matrix_for_pool = []
    for _,_,data in graph.edges(data=True):
        fractional_path = data['path_1stcell']
        path = [wavecar.a1*point[0]+wavecar.a2*point[1]+wavecar.a3*point[2]        for point in fractional_path]        ### Path need to be in Cartesian coordinates
        argument_matrix_for_pool.append((path, Emin+fermi, Emax + fermi, len(path), wavecar, Fouriergrid))
    densities_and_abswvfn_along_path = []
    with multiprocessing.Pool(processes=num_process_use) as pool:
        # Distribute the work across processes; # each tuple in argument_triplets is passed to compute_array
        densities_and_abswvfn_along_path = pool.starmap(charge_density_along_a_path, argument_matrix_for_pool)
    ## calculate time
    end = time.time()
    print('time taken for calculating psi along graphs: ',(end-start))

### add psi, rho to data to graph
for n,data in enumerate(graph.edges(data=True)):
    data[2]['density_along_path'] = densities_and_abswvfn_along_path[n][0]
    data[2]['rho_snr_along_path'] = densities_and_abswvfn_along_path[n][1]
    data[2]['psi_snkr_along_path'] = densities_and_abswvfn_along_path[n][2]

### Now we have the graph with charge-density and added mod(wavefunction) along the edges: Lets first create an extended graph and plot it
extended_graph = create_extended_graph(graph,wavecar,n_samp_pts)
### plot options plot = 'rho_r'/ 'rho_snr'/ 'rho_snkr'
plot_extended_graph(extended_graph, n_samp_pts, wavecar, plot= 'rho_r' , frac_coords_bool =True, atom_label=False, nodenum_label=False)

### Plot charge densities along all edges  (truncated at both end by core_radius)
plot_chg_along_graph_edges(extended_graph, wavecar,core_rad,  exe_model_path)

### find charge density retention and drop along all paths in the structure between periodic copies of atoms
charge_drop,charge_retention = identify_charge_retention(extended_graph)

### Analyze wavefunction at k=Gamma for destructive interference
if charge_retention > charge_density_retention_threshold:
    analyse_wavefunctions_for_destructive_interference(extended_graph,destructive_interference_threshold, Emin+fermi, Emax + fermi, wavecar, n_samp_pts)
