import json 
import numpy as np

## Read Json for atom radii to truncate path near atom centers along a path ### Returns a vector with near-core points as zero 

def truncate_near_atom(cartesian_path,exe_model_path, atom1,atom2,corerad=0.25):
    atomfile = read_json(exe_model_path)
    rad1 = find_atomic_radii(atomfile,atom1)
    rad2 = find_atomic_radii(atomfile,atom2)
    # print(rad1,type(rad2))
    startpoint = cartesian_path[0]
    endpoint = cartesian_path[-1]
    truncated_path_index = []
    for num, point in enumerate(cartesian_path):
        # print(np.linalg.norm(point-startpoint))
        if np.linalg.norm(point-startpoint) < rad1*corerad or np.linalg.norm(point-endpoint) < rad2*corerad:
            truncated_path_index.append(0)
        else:
            truncated_path_index.append(1)
    return truncated_path_index

def read_json(exe_model_path):
    with open(exe_model_path+'/Elements_info.json') as f:
        d = json.load(f)
    return d

def find_atomic_radii(d,elem):
    for i in d['Table']['Row']:
        if i['Cell'][1] == elem:
            #print(i)
            return float(i['Cell'][7])/100  ### to convert to Angstrom from pico
            

        