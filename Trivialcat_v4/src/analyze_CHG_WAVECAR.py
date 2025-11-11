import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import itertools
from .atom_radius import truncate_near_atom
import scipy


## Analyze the charge density along the paths 
def plot_chg_along_graph_edges(extended_graph,wavecar,core_rad, exe_model_path):
    plt.close('all')
    fig,axx = plt.subplots(1,1, figsize=(8, 4))
    cmap = plt.get_cmap('plasma')
    for num, i in enumerate(extended_graph.edges(data=True)):
        density_along_path = i[2]['rho_r_along_path']
        truncated_density_along_path = np.empty((len(density_along_path)),dtype=float)
        atom1 = extended_graph.nodes(data=True)[i[0]]['element']
        atom2 = extended_graph.nodes(data=True)[i[1]]['element']
        actual_path = i[2]['actual_path']
        cartesian_path = np.array([x[0]*wavecar.a1+x[1]*wavecar.a2+x[2]*wavecar.a3  for x in actual_path])
        truncated_path_index = truncate_near_atom(cartesian_path,exe_model_path, atom1,atom2,corerad=core_rad)
        for n,_ in enumerate(density_along_path):
            if truncated_path_index[n] == 0:
                truncated_density_along_path[n] = None
            else:
                truncated_density_along_path[n] = density_along_path[n]
        i[2]['chg_drop'] = (np.nanmax(truncated_density_along_path)-np.nanmin(truncated_density_along_path))/np.nanmax(truncated_density_along_path)
        i[2]['chg_retention'] = (np.nanmin(truncated_density_along_path))/np.nanmax(truncated_density_along_path)
        i[2]['truncated_density_along_path'] = truncated_density_along_path
        axx.plot(truncated_density_along_path, color = cmap(num/ len(extended_graph.edges)))
        axx.text(0.05, 0.95 - (0.25*num /len(extended_graph.edges)) , str(i[0])+'--'+str(i[1]), 
                    transform = axx.transAxes, color = cmap(num/ len(extended_graph.edges)))        
        axx.set_title('charge densities along \n edges of the extended graph')
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('charge_density_drop_along_edges.png',dpi = 500)
    #plt.show()

def identify_charge_retention(extended_graph):    
    ### Now we have the extended graph with charge densitites and wavefunction along the edges
    ## Lets trying to find connections between the periodic copies of atoms (gives tuneeling pathways) and give them score based on chg_drop and wvfcn continuity
    ## path       -->  link between adjacent atoms in the graph
    ## connection -->  link between same atom in two periodic cell (there may be may connections joinning two copies, each consisting of several paths in series)
    overall_chg_drop_score=[]
    overall_chg_retention_score = []
    all_connections = []
    for u in extended_graph.nodes(data=True):
        # print(f"checking extended graph node {u[0]}, atom {u[1]['site']} from POSCAR" )
        for v in extended_graph.nodes(data=True):
            if u[1]['site'] == v[1]['site'] and   u[0] < v[0]: 
                ############### Truncating only connections which are at max 10 paths long
                connections = nx.all_simple_paths(extended_graph,u[0],v[0],cutoff = 10)   ## its a generator, a type of iterable
                chg_drop_score = []
                chg_retention_score = []
                #wv_connection_score = []
                connect = []
                for connection in connections:
                    pair_list1 = itertools.pairwise(connection)
                    ## in each connection, paths are in series. Hence the maximum drop in a path segment shows the drop along the connection
                    chg_drop_score.append(max([extended_graph.edges[pair]['chg_drop'] for pair in pair_list1]))    ## this score shows drop in chg_density
                    
                    pair_list2 = itertools.pairwise(connection)  ## iterables
                    chg_retention_score.append(min([extended_graph.edges[pair]['chg_retention'] for pair in pair_list2]))    ## this score shows drop in chg_density
    
                    # wv_connection_score.append(max([extended_graph.edges[pair]['wvfn_drop'] for pair in pair_list2]))    ## this score shows drop in chg_density
                    connect.append(connection)
                    overall_chg_drop_score.append(min(chg_drop_score))
                    overall_chg_retention_score.append(max(chg_retention_score))
                #### if high verbosity is required
                # print(f"  Between {u[0]} and {v[0]} connecions are {connect}") 
                # print(f"   correcponsing charge density drop = {np.array2string(np.array(chg_drop_score)*100, precision=2, floatmode='fixed')}% ")
                # print(f"   correcponsing charge density retention = {np.array2string(np.array(chg_retention_score), precision=2, floatmode='fixed')} ")
                all_connections.append(connect)
                #print('No connection could be found between 
    if not all_connections:
        print(f'No coonection could be found between periodic copies of atoms for given cutoff {hopping_cutoff} Angstrom')
        overall_chg_drop_score = [1.0]
        overall_chg_retention_score = [0.0]
    print(f'Material has chg drop: {min(overall_chg_drop_score)}; chg retention: {max(overall_chg_retention_score)}')
    return min(overall_chg_drop_score), max(overall_chg_retention_score)



## Analyze wavefunction(k=0,0,0) along graph only if charge density retention is > 15% (exterimental)
def analyse_wavefunctions_for_destructive_interference(extended_graph,destructive_interference_threshold, Emin, Emax, wavecar, n_samp_pts):
    cmap = plt.get_cmap('plasma')
    print(f'Analyze wavefunction(k=0,0,0) along graph only if charge density retention is > {destructive_interference_threshold*100}% (exterimental)\n')
    print('==================================================================================================\n')
    
    
    ## Extended graph stores only nonzero occupancy bands: hence to find truncated numbands, we take the shape of extended_graph
    for _,_,i in extended_graph.edges(data=True):
        trunc_numbands = i['rho_snr_along_path'].shape[0]
    print(f'Number of bands with nonzero density within {Emin} and {Emax}: {trunc_numbands}')
    
    ## figures will have 2columns (one up spin, one down spin) and nbands rows [Number of k-points is 1(gamma only)
    #fig0,ax0 = plt.subplots( wavecar.nsdim*1, trunc_numbands, figsize =(10,5))  ## for plotting psi
    fig0,ax0 = plt.subplots( 2, trunc_numbands, figsize =(10,5))  ## for plotting psi

    gamma_point_psi = np.empty((trunc_numbands,wavecar.nsdim,len(extended_graph.edges), n_samp_pts),dtype = complex)  ## For k=gamma 
    edge_destructive_interference=[]
    for nb in range(trunc_numbands):
        for ns in range(wavecar.nsdim):
            for nk in range(wavecar.nwdim):
                if abs(wavecar.wk[0][nk][0]) < 1e-6 and abs(wavecar.wk[0][nk][1]) < 1e-6 and abs(wavecar.wk[0][nk][2]) < 1e-6:  ### checks if a k point is Gamma
                    print(f'band:{nb} spin:{ns} k:{nk}')
                    #psi = []
                    ax00 = ax0[ns,nb]
                    ax00.set_title(f'band:{nb} spin:{ns} k:{nk}')    
                    
                    for num,[n0,n1,ext] in enumerate(extended_graph.edges(data=True)):
                        colormap = plt.get_cmap('viridis')
                        gamma_point_psi[nb,ns,num,:] = ext['psi_snkr_along_path'][nb][ns][nk]
                        atoms_path = [extended_graph.nodes(data= True)[n0]['element'], extended_graph.nodes(data= True)[n1]['element']]  ## save atom types
                        start_cart_coord = extended_graph.nodes(data=True)[n0]['cart_coords']
                        end_cart_coord = extended_graph.nodes(data=True)[n1]['cart_coords']
                        cart_path = list(zip(np.linspace(start_cart_coord[0],end_cart_coord[0],n_samp_pts), 
                                             np.linspace(start_cart_coord[1],end_cart_coord[1],n_samp_pts), 
                                             np.linspace(start_cart_coord[2],end_cart_coord[2],n_samp_pts)))
                        path_distance = [np.linalg.norm(np.array(cart_path[xx])-np.array(cart_path[0])) for xx in range(len(cart_path))]
                        path_integral = scipy.integrate.trapezoid(np.real(gamma_point_psi[nb,ns,num,:]),path_distance)
                        path_abs_integral = scipy.integrate.trapezoid(np.abs(np.real(gamma_point_psi[nb,ns,num,:])),path_distance)
                        #### checking if destructive interference is present
                        if path_abs_integral != 0:
                            dest_ratio = abs(path_integral) / path_abs_integral
                            if dest_ratio < destructive_interference_threshold:   ### less than 15 percent
                                print(f'{atoms_path[0]}--{atoms_path[1]} (node{n0}--node{n1}) on edge {num} has destructive interference for band:{nb}, spin{ns}')    
                                edge_destructive_interference.append([nb,ns,num,dest_ratio])
                        ax00.plot(np.real(gamma_point_psi[nb,ns,num,:]), color =  cmap(num/ len(gamma_point_psi)))
                        if ns ==0 and nk == 0:
                            ax00.text(0.05, 0.95 - (0.4*num /len(extended_graph.edges)) , str(n0)+'--'+str(n1), 
                            transform = ax0[ns,nb].transAxes, color = cmap(num/ len(extended_graph.edges)))
    fig0.tight_layout()
    fig0.suptitle('Amplitude of wavefunctions',fontsize =10, y=1.05)
    fig0.savefig('Amplitude of wavefunctions',dpi =500)

    if not edge_destructive_interference:
        print(f'We could not find significant destructive interference in the lattice/sublattice') 
    
    ####### Experimental: Summing all real(wavefunction) for the bands within the bandwidth
    #fig3,ax3 = plt.subplots(wavecar.nsdim*1, len(extended_graph.edges), figsize =(15,5))  ## for plotting psi
    fig3,ax3 = plt.subplots(2, len(extended_graph.edges), figsize =(15,5))  ## for plotting psi
    fig3.suptitle(f'Summed wavefunctions at k=Gamma over occupied bands in {(Emin)}eV to{(Emax)}eV over all the edges')
    summed_gamma_psi = np.real(np.sum(gamma_point_psi,axis = 0))
    summed_gamma_psi.shape
    for num,data in enumerate(extended_graph.edges(data=True)):
        for nns in range(wavecar.nsdim):
            ax3[nns][num].plot(summed_gamma_psi[0,num,:])
            ax3[nns][num].set_title(f'edge:{num} spin:{0}')
            # #a = np.trapz(extended_graph
            # ax3[1][num].plot(summed_gamma_psi[1,num,:])
            # ax3[1][num].set_title(f'edge:{num} spin:{1}')
    
    fig3.tight_layout()
    fig3.savefig('Summed wavefunctions.png',dpi=500)
    #plt.show()
    
    ### Destructive interference found in edges?
    Destructive_interf_edges = np.unique([x[2] for x in edge_destructive_interference])
