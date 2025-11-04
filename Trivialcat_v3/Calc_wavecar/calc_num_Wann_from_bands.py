import numpy as np
import pandas as pd
import ast

df = pd.read_csv('/mnt/int_1tb/Anupam/TrivialCat_validation_codes/Calc_wavecar/1.Compound_flatness_score.dat', delimiter='\t')
print(df)
mat = df.loc[df['MP-ID']=='mp-1475733']
print(mat)
flatsegs = mat.iloc[:,4].values[0]
#flatsegs = flatsegs.strip('[').strip(']')

print(flatsegs)
one_seg= []
if '3' in flatsegs:
    one_seg=3
elif '2' in flatsegs:
    one_seg=2
elif '4' in flatsegs:
    one_seg=4
elif '1' in flatsegs:
    one_seg=[]

if one_seg:
    print(f'Target bandwidth:{one_seg}')
    
else:
    print('No flat segment found')
