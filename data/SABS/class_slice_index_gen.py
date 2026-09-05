import numpy as np
import glob
import json
import niftiio as nio

SEG_BNAME="./sabs_CT_normalized/label_*.nii.gz"

segs = glob.glob(SEG_BNAME)
segs = [ fid for fid in sorted(segs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]
if not segs:
    raise ValueError(f'No label volumes found for {SEG_BNAME}')

classmap = {}
LABEL_NAME = [
    "BG", "SPLEEN", "RK", "LK", "GALLBLADDER", "ESOPHAGUS", "LIVER",
    "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS", "AG_R", "AG_L",
]


MIN_TP = 1 # minimum number of positive label pixels to be recorded. Use >100 when training with manual annotations for more stable training

fid = f'./sabs_CT_normalized/classmap_{MIN_TP}.json' # name of the output file. 
for _lb in LABEL_NAME:
    classmap[_lb] = {}
    for _sid in segs:
        pid = _sid.split("_")[-1].split(".nii.gz")[0]
        classmap[_lb][pid] = []

for seg in segs:
    pid = seg.split("_")[-1].split(".nii.gz")[0]
    lb_vol = nio.read_nii_bysitk(seg)
    n_slice = lb_vol.shape[0]
    for slc in range(n_slice):
        for cls in range(len(LABEL_NAME)):
            if np.count_nonzero(lb_vol[slc, ...] == cls) >= MIN_TP:
                classmap[LABEL_NAME[cls]][str(pid)].append(slc)
    print(f'pid {str(pid)} finished!')
    
with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
