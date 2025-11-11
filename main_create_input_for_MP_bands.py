########
# Generate Input files to create identical band structures from Materials Project (only flat band materials given in 'Compound_flatness_score_segments.csv' doc)
# ####
# Usage:   
# python create_input_for_pymatgen_bands.py start_id  end_id  Wavefunction_save  bands_input  scf_input  scf_wannier_input  ref_BS_plot 
#######
# First and second arguments : int
# Rest arguments bool
###############################

MP_API_KEY="Your MP API KEY"


from mp_api.client import MPRester
import mp_api
# import pickle
# import json
# import gc
import ast
import os, shutil, re
import pandas as pd
import sys
import numpy as np
# import glob
import pymatgen
import math
import matplotlib.pyplot as plt
from emmet.core.summary import HasProps
from pymatgen.electronic_structure.plotter import BSPlotter, BSDOSPlotter
# from pymatgen.ext.matproj import MPRester
# from pymatgen.io.vasp.sets import MPScanStaticSet
# from pymatgen.core import Lattice, Structure, Molecule
# import subprocess

read_path = r'/home/anupam/TrivialCat_v4/external/'
write_path = r'/home/anupam/TrivialCat_v4/Calc_wavecar_new/'

## Read flatband list
df = pd.read_csv(read_path+'Compound_flatness_score_segments.csv')
# print(df)
flat_df = df[df['Flat/not-flat'] ==True]
flat_df=flat_df.reset_index(drop=True)
print(flat_df)

##Find all materials in mat Proj that has BS, and DOS
# print('Scanning Materials Project for materials with BS and DOS')
# with mp_api.client.MPRester(MP_API_KEY) as mpr:
#    docs = mpr.materials.summary.search(has_props=[HasProps.bandstructure,HasProps.dos], fields=["material_id"])
# mid=[]
# matid=[]
# for item in docs:
#    mid.append(int(item.material_id.split('-')[1]))
# mid.sort()
# for item in mid:
#    matid.append('mp-'+ str(item))
# # with open(write_path+'list_of_materials_with_BS_DOS_CHGCAR','wb') as ff:
# #     pickle.dump(matid,ff)
# with open(read_path+ r'list_of_materials_with_BS_DOS_readable','w') as fp:
#    for item in matid:
#        fp.write("%s\n" % item)
#    print('Done scanning\n ================================\n')

matid=[]
with open(read_path+'list_of_materials_with_BS_DOS_readable','r') as ff:
    for line in ff:
        matid.append(line.rstrip())
# print(pd.DataFrame(matid))

cwdir=os.getcwd()

##give the range of mp-id
## Getting arguments
start= int(sys.argv[1])
end = int(sys.argv[2])
wavefn_save = True if len(sys.argv)<4 else bool(sys.argv[3])   ## default true
bands_input = True if len(sys.argv)<5 else bool(sys.argv[4])
scf_input = True if len(sys.argv)<6  else bool(sys.argv[5])
scf_wannier_input = False if len(sys.argv)<7  else bool(sys.argv[6])
ref_BS_plot_from_MP = False if len(sys.argv)<8 else bool(sys.argv[7])
print(f'start_id= {start}, end_id={end}, wavefn_save = {wavefn_save}, bands_input = {bands_input}, scf_input = {scf_input}, scf_Wannier_input = {scf_wannier_input}, plot_ref_BS_from_MP={ref_BS_plot_from_MP}')
#mat_num = 1

for index,row in flat_df.iterrows():
    #print(index)
    if (row['MP-ID'] in matid):
        material_id = row['MP-ID']
        id = int(material_id.split('-')[1])
        if id <= end  and  id >= start :
            print(material_id)
            if os.path.isdir(write_path+material_id):
                shutil.rmtree(write_path+material_id)
            with mp_api.client.MPRester(MP_API_KEY) as mpr:
                bs = mpr.get_bandstructure_by_material_id(material_id)
                print(bs.efermi)
                tasks = mpr.get_task_ids_associated_with_material_id(material_id)
                
                summary_doc = mpr.materials.summary.search(material_ids=[material_id])[0]
                task_id = summary_doc.bandstructure.setyawan_curtarolo.task_id
                #doc = mpr.tasks.search(task_ids=[task_id], fields=["input", "calcs_reversed"])
                doc = mpr.tasks.search(task_ids=[task_id], fields=["input","structure"])
                #print(summary_doc.bandstructure.setyawan_curtarolo)

                ## Get input files
                struc = doc[0].structure
                incar_param = doc[0].input.incar   ## gets from Task ID
                #write_path = r'/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'
                inputband= pymatgen.io.vasp.sets.MPNonSCFSet(struc,user_incar_settings = incar_param, mode='Line')
                if bands_input:
                    inputband.write_input(output_dir=write_path+'/'+material_id+'/bands/')
                #print(incar_param)
                if scf_input:
                    scf_incar_param = incar_param
                    scf_incar_param['ALGO'] = 'Normal'
                    scf_incar_param['LWAVE']=True
                    scf_incar_param['LCHARG']=True
                    scf_incar_param['ICHARG'] = 2     ## change ICHARG from Bands input
                    inputscf= pymatgen.io.vasp.sets.MPStaticSet(struc,user_incar_settings = scf_incar_param)
                    inputscf.write_input(output_dir=write_path+'/'+material_id+'/scf/')
                if scf_wannier_input:
                    scf_wann_incar_param = scf_incar_param
                    scf_wann_incar_param['ALGO'] = 'Normal'
                    scf_wann_incar_param['LWAVE']=True
                    scf_wann_incar_param['LCHARG']=True
                    scf_wann_incar_param['ICHARG'] = 2
                    ## Extra inputs for wannier90
                    num_core=24
                    scf_wann_incar_param['LSCDM'] = True
                    scf_incar_param['NBANDS'] = math.ceil(scf_incar_param['NBANDS']/num_core)*num_core   ## This happens as parellel run changes NBANDS
                    scf_wann_incar_param['NUM_WANN'] = scf_incar_param['NBANDS']
                    scf_wann_incar_param['LWANNIER90']=True
                    #scf_wann_incar_param['KPAR'] = 1
                    #scf_wann_incar_param['NCORE'] = 1

                    inputwannscf = pymatgen.io.vasp.sets.MPStaticSet(struc,user_incar_settings = scf_wann_incar_param)    
                    inputwannscf.write_input(output_dir=write_path+'/'+material_id+'/scf/')
                    
                    ## plot band-structure with wannier90
                    ## Find k-path
                    kpoints= bs.kpoints
                    branches = bs.branches
                    high_symm_kpoints= []
                    labels = []
                    # print(branches)
                    for nn,branch in enumerate(branches):
                        high_symm_kpoints.append([kpoints[branch['start_index']].frac_coords, kpoints[branch['end_index']].frac_coords])
                        labels.append([branch['name'].split('-')[0], branch['name'].split('-')[1]])
                    # high_symm_kpoints.append(kpoints[branch['end_index']].frac_coords)
                    # labels.append(branch['name'].split('-')[1])
                    # print(high_symm_kpoints)
                    # print(labels)
                    with open(write_path+material_id+'/scf/INCAR','a') as f:
                        # f.write(f'NUM_WANN = {scf_incar_param['NBANDS']}\n')
                        f.write('# add this lines to wannier90.win \n')
                        f.write('WANNIER90_WIN = " \n')
                        f.write(f'WRITE_HR = TRUE \n')
                        f.write('BANDS_PLOT = TRUE \n')
                        f.write('begin kpoint_path \n')
                        for nn,symm_pt in enumerate(high_symm_kpoints):
                            # print(type(high_symm_kpoints[nn][0]))
                            f.write(f'{labels[nn][0]}  {str(high_symm_kpoints[nn][0])[1:-1]}    {labels[nn][1]}  {str(high_symm_kpoints[nn][1])[1:-1]}  \n')
                        f.write('end kpoint_path \n')
                        f.write('"\n')

                    # segments = ast.literal_eval(df.loc[df['MP-ID'] == material_id]['Flat segments'].tolist()[0])
                    # Emin = (2 - max(segments))*0.5
                    # Emax = (3 - max(segments))*0.5
                    # print(Emin,Emax)
                    
                    # print(bs.is_spin_polarized,'\n',list(bs.bands.values())[0])
                    # eigenval_mat_up = list(bs.bands.values())[0]
                    
                    # if bs.is_spin_polarized:
                    #     eigenval_mat_dn =  list(bs.bands.values())[1]
                    # print(eigenval_mat_dn)
                    
                    
                ### Print Band structure
                if ref_BS_plot_from_MP:
                    #import matplotlib.pyplot as plt
                    #plotter = BSPlotter(bs)
                    #plotter.get_plot(zero_to_efermi=True, ylim=[-1,1], smooth=False)
                    plotter = BSDOSPlotter(bs_projection='elements', dos_projection=None,egrid_interval = 0.5, 
                                           vb_energy_range=1.0, cb_energy_range=1.0,fixed_cb_energy=True, axis_fontsize=15, 
                                           tick_fontsize=15, legend_fontsize=12, bs_legend='best', 
                                           dos_legend=None, rgb_legend=True, fig_size=(8,6))
                    plotter.get_plot(bs,dos=None)
                    #plt.show()
                    plt.savefig(write_path + material_id +'/MP_ref_bs.png',dpi=300)

                ## get input
                #task_ids= m.get_entry_by_material_id(i)
                #TID = task_ids[-1].data['task_id']   ### last task id
                #docs = m.materials.tasks.search(task_ids=[TID], fields=['input'])
                #kspace = docs[0].input.incar['KSPACING']

                ## get chgcar and structure
                #chg = m.get_charge_density_from_material_id(i)
                #struc = m.get_structure_by_material_id(i)
                # struc = chg.structure
                # print(dir(chg))
                # incarsettings = docs[0].input.incar
                # kptsettings = docs[0].input.kpoints
                # potcarsettings = docs[0].input.potcar_spec
                # inputs = pymatgen.io.vasp.sets.MPStaticSet(struc, user_incar_settings = incarsettings)
                # # inputs = MPScanStaticSet( struc, user_incar_settings = incarsettings)
                # # inputs= MPScanStaticSet( struc, user_incar_settings = docs[0].input.parameters)
                # inputs= MPScanStaticSet(structure = struc)

                ## generating KPOINTS file and deleting other files
                #inputstatic = pymatgen.io.vasp.sets.MPStaticSet(struc)
                #inputstatic.write_input(output_dir=write_path+i)
                #[os.remove(write_path+i+'/'+inpfile) for inpfile in ['INCAR','POSCAR','POTCAR']]

                #inputscan = pymatgen.io.vasp.sets.MPStaticSet(struc, user_incar_settings = {"LWAVE": True, "ICHARG": 10,"ISMEAR":0}, user_kpoints_settings ={})
                #inputscan.write_input(output_dir=write_path+i)
                #chg.write_file('./mp-2317/CHGCAR')
                ## Write band files
                # inputband= pymatgen.io.vasp.sets.MPNonSCFSet(struc,mode='Line')
                # inputband.write_input(output_dir=write_path+i+'/bands/')
                # os.remove(write_path+i+'/bands/KPOINTS')
                # vaspkit_input = "3\n303\n"
                # os.chdir(write_path+i+'/bands/')
                # try:
                #     process = subprocess.run(
                #                 ['vaspkit'],
                #                 input=vaspkit_input,
                #                 capture_output=True,
                #                 text=True,
                #                 check=True
                #                 )
                #     # The output from the command-line tool is now captured.
                #     files_to_rem = ['HIGH_SYMMETRY_POINTS',  'PRIMCELL.vasp',  'SYMMETRY','Brillouin_Zone_3D.jpg','PLOT.in']
                #     [os.remove(write_path+i+'/bands/'+ name) for name in files_to_rem]
                #     os.rename('KPATH.in','KPOINTS')
                #     print("VASPkit Output:\n", process.stdout)
                #     print("KPOINTS file generated successfully.")
                # except subprocess.CalledProcessError as e:
                #     print(f"An error occurred while running VASPkit: {e}")
                #     print("STDERR:", e.stderr)
                # os.chdir(cwdir)                
                # chg.write_file(write_path+i+'/bands/CHGCAR')

                # ## Write scf files
                # #os.chdir(write_path+i+'/scf_wavecar_calc')
                # #chg = m.get_charge_density_from_material_id(i)
                # inputscf = pymatgen.io.vasp.sets.MPStaticSet(struc, user_incar_settings = {"LWAVE": True, "ICHARG": 1,"ISMEAR":0})
                # inputscf.write_input(output_dir=write_path+i+'/scf_wavecar_calc/')
                # chg.write_file(write_path+i+'/scf_wavecar_calc/CHGCAR')
                # [os.remove(write_path+i+'/scf_wavecar_calc/'+inpfile) for inpfile in ['INCAR','POSCAR','POTCAR']]
                # inputscanscf = pymatgen.io.vasp.sets.MPScanStaticSet(struc, user_incar_settings = {"LWAVE": True, "ICHARG": 2,"ISMEAR":0, 
                #                                                                            "NBANDS" :48 , "NUM_WANN": 48,"LSCDM": True, "LWANNIER90": True,
                #                                                                            "LWRITE_UNK":True, "KPAR": 1, "NCORE": 1, 
                #                                                                            "WANNIER90_WIN": "dis_num_iter = 0"})
                # inputscanscf.write_input(output_dir=write_path+i+'/scf_wavecar_calc/')
        elif index >= end:
            break
