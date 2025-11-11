import pandas as pd
#import jarvis
import numpy as np
import scipy

#from jarvis.io.vasp.outputs import Vasprun
import matplotlib.pyplot as plt
from pymatgen.io.vasp import outputs
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.plotter import BSDOSPlotter
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Chgcar
from pymatgen.io.lobster import Doscar

from mp_api.client import MPRester

MP_API_KEY="Your MP API KEY"

mp_id = '20536'
# with MPRester(MP_API_KEY) as mpr:
# #    chg = mpr.get_charge_density_from_material_id('mp-'+mp_id)
# #with mp_api.client.MPRester("Jq3p6rY2iay8B8QMDKOOopsEjsSmFVcZ") as mpr:
#     bs = mpr.get_bandstructure_by_material_id('mp-'+mp_id)
# Efermi = bs.efermi
# print(Efermi)

Path = '/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'+'mp-' + mp_id + '/'
# vrun_scf = outputs.Vasprun(Path+'scf/vasprun.xml',parse_dos = True)
# chgcar_data = Chgcar.from_file(Path+"bands/CHGCAR")
#locpot = outputs.Locpot.from_file(Path+'scf/LOCPOT')
#Evac = max(locpot.get_average_along_axis(3))
#Efermi = chgcar_data.efermi
#Efermi = vrun_scf.efermi
#print(Efermi)
vrun_bands = outputs.Vasprun(Path+'bands/vasprun.xml',\
                         parse_dos = False,parse_eigen = True, parse_projected_eigen = True, parse_potcar_file= False,\
                         occu_tol= 1e-08, separate_spins = True, exception_on_bad_xml= True)
## Reading DOSCAR
with open(Path+'bands/DOSCAR', mode="r", encoding="utf-8") as file:
    file.readline()  # Skip the first line
    Efermi = float([file.readline() for nn in range(5)][4].split()[3])
print(Efermi)
kpath_bands = Path+'bands/KPOINTS'
bands = vrun_bands.get_band_structure(kpoints_filename = kpath_bands, efermi=Efermi  , line_mode = True )
#dos = vrun_scf.complete_dos
bsp = BSDOSPlotter(bs_projection='elements', dos_projection=None,egrid_interval = 0.5, vb_energy_range=1.0, cb_energy_range=1.0,fixed_cb_energy=True, axis_fontsize=15, tick_fontsize=15, legend_fontsize=12, bs_legend='best', dos_legend=None, rgb_legend=True, fig_size=(8,6))
#print(bands,dos)
image=bsp.get_plot(bands,dos=None)
#print(image)
#bsp.save_plot('a.png')


#### Wanniertools output
#data = np.loadtxt(Path+"scf/bulkek.dat", comments="#")
#k = data[:, 0]
#E = data[:, 1]
## infer number of k-points per band by detecting when k resets
#resets = np.where(np.diff(k) < 0)[0] + 1
#n_k = resets[0] if resets.size else len(k)
#n_bands = len(k) // n_k
#
#K = k[:n_k]
#Ebands = E.reshape(n_bands, n_k)
#
#plt.figure(figsize=(6, 4))
#for band in Ebands:
#    plt.plot(K, band, lw=1)
#plt.ylim(-4,4)
#plt.xlabel("k-path length")
#plt.ylabel("Energy (eV)")
#plt.title("Band structure")
#plt.tight_layout()
#plt.show()

plt.savefig(Path+'bands/bands.png',dpi=400)
#plt.show()
