from pymatgen.io.vasp import outputs
import numpy as np
import scipy

## Get sublattice from DOS from vasprun.xml ### Find species which has largest DOS
def get_sublattice(Emax,Emin,fermi,sublattice_selection = 'maxdensity'):
    ldos_max={}
    ldos_sum={}
    dos = outputs.Vasprun('vasprun.xml').complete_dos
    elems = outputs.Vasprun('vasprun.xml').final_structure.elements
    elem_DOS = dos.get_element_dos()
    eng=dos.energies
    for i in elems:
        denst= elem_DOS[i].as_dict()['densities']
        for j in denst.keys():   #for spin polarized case
            if j=='1':
                sum_density=np.array(denst[j])
            else:
                sum_density=sum_density+np.array(denst[j])
        dos_full = np.stack((eng,sum_density),axis=1)
        range_dos=[Emax,Emin]
        b = dos_full[dos_full[:, 0] <= range_dos[0]]
        b = b[b[:, 0] >= range_dos[1]]
        #print(b)
        ldos_max[i]=max(b[:,1])
        ldos_sum[i]=scipy.integrate.trapezoid(b[:,1],b[:,0])
    if sublattice_selection == 'maxdensity':
        bb=max(ldos_max,key=ldos_max.get)    ### sublattice found from maximum peak height in bandwidth
    elif sublattice_selection == 'sumdensity':
        bb=max(ldos_sum,key=ldos_sum.get)    ### sublattice identified from heighest integrated PDOS in bandwidth
    print(f'Sublattice elemet: {bb}\nFlatness and Sublattice Prediction done \n============================================\n')
    return bb