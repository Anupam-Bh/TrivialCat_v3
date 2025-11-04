from pymatgen.core import Structure
import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import NearNeighbors
import pymatgen.analysis.local_env
import numpy as np

## Make the structure graph
def Make_graph(n_samp_pts, hopping_cutoff, sublattice_element = None): 
    structure = Structure.from_file('POSCAR')
    
    print(f'Structure: \n ========================\n {structure} \n')

    #### If sublattice is used
    if sublattice_element:
        unwanted_elements = [element.name for element in structure.elements]
        unwanted_elements.remove(sublattice_element)
        structure.remove_species(unwanted_elements)
    
    #### define graph connectivity
    #chgcar = chgcar_input
    # neighbour_strategy = pymatgen.analysis.local_env.MinimumDistanceNN(max(structure.lattice.abc)*1.01)
    # neighbour_strategy = pymatgen.analysis.local_env.IsayevNN(tol = 0.25, targets = None, 
    #                                                           cutoff= 13.0, allow_pathological= False, 
    #                                                           extra_nn_info= True, compute_adj_neighbors = True)  ## https://www.nature.com/articles/ncomms15679
    # neighbour_strategy = pymatgen.analysis.local_env.VoronoiNN(tol=0, targets=None, 
    #                                                            cutoff=13.0, allow_pathological=False, 
    #                                                            weight='solid_angle', extra_nn_info=True, compute_adj_neighbors=True)
    neighbour_strategy = pymatgen.analysis.local_env.MinimumDistanceNN(tol = 0.2, cutoff = hopping_cutoff, get_all_sites=True)
    # neighbour_strategy = pymatgen.analysis.local_env.CrystalNN(weighted_cn=False, cation_anion=False, 
    #                                                            distance_cutoffs=(0.5, 1), x_diff_weight=3.0, 
    #                                                            porous_adjustment=False, search_cutoff=7, fingerprint_length=None)
    # neighbour_strategy = pymatgen.analysis.local_env.Critic2NN
    # neighbour_strategy = MinimumDistanceNN(structure,NearNeighbors)
    #structure = chgcar.structure

    print(f'Sublattice Structure: \n ========================\n {structure} \n')
    
    structure_graph = StructureGraph.from_local_env_strategy(structure, neighbour_strategy, weights=True)
    # structure_graph = StructureGraph(structure)
    graph = nx.MultiGraph(structure_graph.graph) # from dimultigraph to multigraph!

    for node, data in graph.nodes(data=True):
        site = structure_graph.structure[node]
        data["element"] = site.specie.name
        data["frac_coords"] = site.frac_coords
        data["cart_coords"] = site.coords
    print('\n----------------------------\nStructure graph:')
    print('initial_coord    final_coord      image_in_periodic_cell\n -------------------------------------------------')
    for u, v, data in graph.edges(data=True):
        image = data["to_jimage"]
        i = graph.nodes[u]["frac_coords"] 
        f = (graph.nodes[v]["frac_coords"] + image)
        print (i,f, image)                                                           ### points along an edge
        ##### Create path from node0---node1  through 1st unit cell (for calculating wavefunction along this path, 
        ##### as we have wavefunction, charge density only in 1st unit cell.
        pathincell_frac_x = [vv-np.floor(vv) for vv in np.linspace(i[0],f[0],n_samp_pts)]  # brings coordinates back to [0,1)  
        pathincell_frac_y = [vv-np.floor(vv) for vv in np.linspace(i[1],f[1],n_samp_pts)]
        pathincell_frac_z = [vv-np.floor(vv) for vv in np.linspace(i[2],f[2],n_samp_pts)]
        pathincell = [que for que in zip(pathincell_frac_x, pathincell_frac_y,pathincell_frac_z)]
        data['path_1stcell'] = np.array(pathincell)
        #### Calculate actual path from node0----node1
        #### this will be used for distance calculation along graph 
        path_frac_x = np.linspace(i[0],f[0],n_samp_pts)  # brings coordinates back to [0,1)  
        path_frac_y = np.linspace(i[1],f[1],n_samp_pts)
        path_frac_z = np.linspace(i[2],f[2],n_samp_pts)
        path = [que for que in zip(path_frac_x, path_frac_y,path_frac_z)]
        data['actual_path'] = np.array(path)
        
    print('--------------------------------------------------------------------------------------')
    print(f'The graph is created. It has {len(graph.edges)} edges and {len(graph.nodes)} nodes')
    return structure_graph, graph