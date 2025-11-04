from mp_api.client import MPRester
import pymatgen
#from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.sets import MPScanStaticSet
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.electronic_structure.plotter import BSPlotter, BSDOSPlotter

Path = '/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'
material_id = "mp-20536"

with MPRester(api_key="ZvKikSZ2FD5MYpklS0TDlmCiHPhO7Qm6") as mpr:
    bs = mpr.get_bandstructure_by_material_id(material_id)
    print(bs.efermi)
    tasks = mpr.get_task_ids_associated_with_material_id(material_id)
    
    summary_doc = mpr.materials.summary.search(material_ids=[material_id])[0]
    task_id = summary_doc.bandstructure.setyawan_curtarolo.task_id
    doc = mpr.tasks.search(task_ids=[task_id], fields=["input", "calcs_reversed"])
    #print(summary_doc.bandstructure.setyawan_curtarolo)
### Print Band structure
import matplotlib.pyplot as plt
#plotter = BSPlotter(bs)
#plotter.get_plot(zero_to_efermi=True, ylim=[-1,1], smooth=False)
plotter = BSDOSPlotter(bs_projection='elements', dos_projection=None,egrid_interval = 0.5, vb_energy_range=1.0, cb_energy_range=1.0,fixed_cb_energy=True, axis_fontsize=15, tick_fontsize=15, legend_fontsize=12, bs_legend='best', dos_legend=None, rgb_legend=True, fig_size=(8,6))
plotter.get_plot(bs,dos=None)
plt.savefig(Path+'bs.png',dpi=300)

## Get input files
# struc = doc[0].structure
# incar_param = doc[0].input.incar
# write_path = r'/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'
# inputband= pymatgen.io.vasp.sets.MPNonSCFSet(struc,user_incar_settings = incar_param, mode='Line')
# inputband.write_input(output_dir=write_path+'/'+material_id+'/bands/')
# #print(incar_param)
# incar_param['LCHARG']=True
# incar_param['ICHARG']=2
# #print(incar_param)
# inputscf= pymatgen.io.vasp.sets.MPStaticSet(struc,user_incar_settings = incar_param)    ### Change ICHARG in INCAR 
# inputscf.write_input(output_dir=write_path+'/'+material_id+'/scf/')
