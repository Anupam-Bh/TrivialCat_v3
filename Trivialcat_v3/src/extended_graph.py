import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


## Extended Graph and Plotting  
def gradient_line_3D(ax, x1, y1, z1, x2, y2, z2, color_values):
    # Parameter t goes from 0 to 1
    t = np.linspace(0, 1, len(color_values))
    # Parametric equations for x(t), y(t), z(t)
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    cmap = plt.get_cmap('viridis')
    # We'll plot each small line segment with a single color
    for i in range(len(color_values) - 1):
        # Segment from i to i+1
        xs = [x[i], x[i+1]]
        ys = [y[i], y[i+1]]
        zs = [z[i], z[i+1]]
        c = cmap(color_values[i])
        ax.plot(xs, ys, zs, color=c, linewidth=3)

def create_extended_graph(graph,wavecar,n_samp_pts):
    ## create extended graph
    nodes = []
    frac_coords = []
    cart_coords = []
    elements = []
    edges = []
    nodecount = []
    extended_graph = nx.Graph()
    for i,j,data in graph.edges(data=True):
        # creating a vector for each node in an  edge [atom#, cellx, celly, cellz]
        a = [i,0,0,0]         ## in the original graph an edge always starts from [0,0,0] unit cell
        b = [j,data['to_jimage'][0],data['to_jimage'][1],data['to_jimage'][2]]
        for node in [a,b]:
            if node not in nodes:
                frac_coords.append(graph.nodes(data=True)[node[0]]['frac_coords'] + node[1:])
                cart_coords.append(graph.nodes(data=True)[node[0]]['cart_coords'] + (node[1]*wavecar.a1+node[2]*wavecar.a2+node[3]*wavecar.a3))
                elements.append(graph.nodes(data=True)[node[0]]['element'])
                nodes.append(node)
                extended_graph.add_node(len(nodes)-1 ,site = node[0], frac_coords = frac_coords[-1] , cart_coords = cart_coords[-1] ,
                                        element = elements[-1], incell = node[1:], 
                                        frac_coord_in_1stunitcell = graph.nodes(data=True)[node[0]]['frac_coords'] )               
        extended_graph.add_edge(nodes.index(a),nodes.index(b), path_1stcell = data['path_1stcell'], actual_path = data['actual_path'], rho_r_along_path = data['density_along_path'], 
                                rho_snr_along_path = data['rho_snr_along_path'], psi_snkr_along_path = data['psi_snkr_along_path'],
                                chg_drop = (max(data['density_along_path'])-min(data['density_along_path']))/max(data['density_along_path']))
    

    ##### modifying and truncating unoccupied bands
    for _,_,ext in extended_graph.edges(data=True):
        
        ## reorganize the dimensions
        numpts = n_samp_pts
        numspins = wavecar.nsdim
        numbands= wavecar.nbdim
        numks = wavecar.nwdim
        
        rho_r = np.empty((numpts),dtype=float)
        rho_n_s_r = np.empty((numbands,numspins,numpts),dtype=float)
        psi_n_s_k_r = np.empty((numbands,numspins,numks,numpts),dtype=complex)

        for nb in range(numbands):   ## bands
            for ns in range(numspins):   ## spins
                for p in range(numpts):      ### loop over points
                    rho_n_s_r[nb,ns,p] = ext['rho_snr_along_path'][p][ns][nb]                
                    for nk in range(numks):
                        psi_n_s_k_r[nb,ns,nk,p] = ext['psi_snkr_along_path'][p][ns][nb][nk]
                        
        ### truncating only occupied bands: remove band indices which have zero density for all spin and k
        truncated_rho = []
        truncated_psi = []
        for nb in range(numbands):
            if np.any(rho_n_s_r[nb,:,:]):
                truncated_rho.append(rho_n_s_r[nb,:,:])
                truncated_psi.append(psi_n_s_k_r[nb,:,:,:])            
        truncated_rho = np.array(truncated_rho)
        truncated_psi = np.array(truncated_psi)
        ## deleting the old edge attributes
        ext['rho_snr_along_path'] = []
        ext['psi_snkr_along_path'] = []
        ## adding new data
        ext['rho_snr_along_path'] = truncated_rho
        ext['psi_snkr_along_path'] = truncated_psi
    
    return extended_graph


def plot_extended_graph(extended_graph, n_samp_pts, wavecar, plot , frac_coords_bool =True, atom_label=False, nodenum_label=False):
    ##data
    numpts = n_samp_pts
    numspins = wavecar.nsdim
    numbands= wavecar.nbdim
    numks = wavecar.nwdim
    elements = []
    frac_coords = []
    cart_coords = []
    quantity_along_paths = []
    for _,i in extended_graph.nodes(data=True):
        elements.append(i['element'])
        frac_coords.append(i['frac_coords'])   
        cart_coords.append(i['cart_coords'])

    if plot == 'rho_r':   
        for _,_,i in extended_graph.edges(data=True):
            quantity_along_paths.append(i['rho_r_along_path'])
            ncol = 1
            nrow = 1
    elif plot == 'rho_snr':
        for _,_,i in extended_graph.edges(data=True):
            quantity_along_paths.append(i['rho_snr_along_path'])
            ncol = numspins
            nrow = i['rho_snr_along_path'].shape[0]
    elif plot == 'psi_snkr':
        for _,_,i in extended_graph.edges(data=True):
            quantity_along_paths.append(i['psi_snkr_along_path'])  
            ncol = numspins*numks
            nrow = i['rho_snr_along_path'].shape[0]
    else: 
        print('No valid plot options are entered')
    
    #%matplotlib widget
    fig = plt.figure(figsize=(8,8))
    for nr in range(nrow):
        for nc in range(ncol):
            ax = fig.add_subplot(nrow,ncol,nr*ncol+nc+1,projection='3d')   ##starts count with 1
            
            #Plot nodes    
            element_colors = np.unique(elements,return_inverse=True)[1]
            if frac_coords_bool == True:
                arr = np.array(frac_coords)
                for nodename,ix in extended_graph.nodes(data=True):                #### annotate nodes
                    ax.text(ix['frac_coords'][0],ix['frac_coords'][1],ix['frac_coords'][2], [str(nodename),ix['element']])
            elif frac_coords_bool == False:
                arr = np.array(cart_coords)
                for nodename,ix in extended_graph.nodes(data=True):                #### annotate nodes
                    ax.text(ix['cart_coords'][0],ix['cart_coords'][1],ix['cart_coords'][2], 
                            [x for n,x in enumerate([str(nodename),ix['element']]) if [nodenum_label,atom_label][n]==True])                
            ax.scatter(*np.transpose(arr), c= element_colors,  cmap = 'flag', depthshade=True) 

            #Plot edges
            for i , j , data in extended_graph.edges(data=True):
                if frac_coords_bool == True:
                    x1,y1,z1 = extended_graph.nodes(data=True)[i]['frac_coords']
                    x2,y2,z2 = extended_graph.nodes(data=True)[j]['frac_coords'] 
                else:
                    x1,y1,z1 = extended_graph.nodes(data=True)[i]['cart_coords']
                    x2,y2,z2 = extended_graph.nodes(data=True)[j]['cart_coords']                 
                gradient_line_3D(ax, x1, y1, z1, x2, y2, z2, data['rho_r_along_path']/np.mean(quantity_along_paths))  ##color normalized within all the paths
            ax.set_box_aspect([1,1,1])
        
            ###Plot and create unit cell
            vertices = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
            cubeedges = [(0, 1), (0, 2),(0, 4),  (1, 3),  (1, 5),  (2, 3),  (2, 6),  (3, 7),  (4, 5),  (4, 6),  (5, 7), (6, 7)]
            for edge in cubeedges:
                if frac_coords_bool==True:
                    # edge is a pair like (0, 1). Let's get the two endpoints:
                    start = vertices[edge[0]]
                    end   = vertices[edge[1]]
                else:
                    start = vertices[edge[0]][0]*wavecar.a1 + vertices[edge[0]][1]*wavecar.a2 + vertices[edge[0]][2]*wavecar.a3
                    end = vertices[edge[1]][0]*wavecar.a1 + vertices[edge[1]][1]*wavecar.a2 + vertices[edge[1]][2]*wavecar.a3
                # start = (x1, y1, z1), end = (x2, y2, z2)
                x_vals = [start[0], end[0]]
                y_vals = [start[1], end[1]]
                z_vals = [start[2], end[2]]                  
                # Plot a line between the two endpoints
                ax.plot(x_vals, y_vals, z_vals, color='b',linestyle=':')
    ax.set_zlim(0,20)
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=90, azim=0)
    plt.axis('off')
    plt.title('3D visualization of charge densities along edges')
    plt.savefig('extended_graph.png',dpi =500)
    
    #plt.show()
    
