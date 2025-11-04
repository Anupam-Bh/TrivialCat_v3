MP_API_KEY="Jq3p6rY2iay8B8QMDKOOopsEjsSmFVcZ"


from mp_api.client import MPRester
import mp_api
import pickle
import json
import gc
import os, shutil, re
import pandas as pd
import glob
import pymatgen
from emmet.core.summary import HasProps
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.sets import MPScanStaticSet
from pymatgen.core import Lattice, Structure, Molecule
import subprocess

read_path = r'/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'
write_path = r'/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/'

## Read flatband list
df = pd.read_table(read_path+'1.Compound_flatness_score.dat')
flat_df = df[df['Flat/not-flat'] ==True]
flat_df=flat_df.reset_index(drop=True)
print(flat_df)

##Find all materials in mat Proj that has BS, DOS and Charge density
#with mp_api.client.MPRester(MP_API_KEY) as mpr:
#    docs = mpr.materials.summary.search(has_props=[HasProps.bandstructure,HasProps.dos, HasProps.charge_density], fields=["material_id"])
#mid=[]
#matid=[]
#for item in docs:
#    mid.append(int(item.material_id.split('-')[1]))
#mid.sort()
#for item in mid:
#    matid.append('mp-'+ str(item))
## with open(write_path+'list_of_materials_with_BS_DOS_CHGCAR','wb') as ff:
##     pickle.dump(matid,ff)
#with open(write_path+ r'list_of_materials_with_BS_DOS_CHGCAR_readable','w') as fp:
#    for item in matid:
#        fp.write("%s\n" % item)
#    print('Done')

matid=[]
with open(read_path+'list_of_materials_with_BS_DOS_CHGCAR_readable','r') as ff:
    for line in ff:
        matid.append(line.rstrip())
print(pd.DataFrame(matid))

cwdir=os.getcwd()

##give the range of mp-id
start= 20536
end = 20537
#mat_num = 1

for index,row in flat_df.iterrows():
    #print(index)
    if (row['MP-ID'] in matid):
        i = row['MP-ID']
        id = int(i.split('-')[1])
        if id < end  and  id >= start :
            print(i)
            if os.path.isdir(write_path+i):
                shutil.rmtree(write_path+i)
            with mp_api.client.MPRester(MP_API_KEY) as m:
                ## get input
                #task_ids= m.get_entry_by_material_id(i)
                #TID = task_ids[-1].data['task_id']   ### last task id
                #docs = m.materials.tasks.search(task_ids=[TID], fields=['input'])
                #kspace = docs[0].input.incar['KSPACING']

                ## get chgcar and structure
                chg = m.get_charge_density_from_material_id(i)
                #struc = m.get_structure_by_material_id(i)
                struc = chg.structure
                print(dir(chg))
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
                inputband= pymatgen.io.vasp.sets.MPNonSCFSet(struc,mode='Line')
                inputband.write_input(output_dir=write_path+i+'/bands/')
                os.remove(write_path+i+'/bands/KPOINTS')
                vaspkit_input = "3\n303\n"
                os.chdir(write_path+i+'/bands/')
                try:
                    process = subprocess.run(
                                ['vaspkit'],
                                input=vaspkit_input,
                                capture_output=True,
                                text=True,
                                check=True
                                )
                    # The output from the command-line tool is now captured.
                    files_to_rem = ['HIGH_SYMMETRY_POINTS',  'PRIMCELL.vasp',  'SYMMETRY','Brillouin_Zone_3D.jpg','PLOT.in']
                    [os.remove(write_path+i+'/bands/'+ name) for name in files_to_rem]
                    os.rename('KPATH.in','KPOINTS')
                    print("VASPkit Output:\n", process.stdout)
                    print("KPOINTS file generated successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while running VASPkit: {e}")
                    print("STDERR:", e.stderr)
                os.chdir(cwdir)                
                chg.write_file(write_path+i+'/bands/CHGCAR')

                ## Write scf files
                #os.chdir(write_path+i+'/scf_wavecar_calc')
                #chg = m.get_charge_density_from_material_id(i)
                inputscf = pymatgen.io.vasp.sets.MPStaticSet(struc, user_incar_settings = {"LWAVE": True, "ICHARG": 1,"ISMEAR":0})
                inputscf.write_input(output_dir=write_path+i+'/scf_wavecar_calc/')
                chg.write_file(write_path+i+'/scf_wavecar_calc/CHGCAR')
                [os.remove(write_path+i+'/scf_wavecar_calc/'+inpfile) for inpfile in ['INCAR','POSCAR','POTCAR']]
                inputscanscf = pymatgen.io.vasp.sets.MPScanStaticSet(struc, user_incar_settings = {"LWAVE": True, "ICHARG": 1,"ISMEAR":0, 
                                                                                           "NBANDS" :48 , "NUM_WANN": 48,"LSCDM": True, "LWANNIER90": True,
                                                                                           "LWRITE_UNK":True, "KPAR": 1, "NCORE": 1, 
                                                                                           "WANNIER90_WIN": "dis_num_iter = 0"})
                inputscanscf.write_input(output_dir=write_path+i+'/scf_wavecar_calc/')
        elif index >= end:
            break
