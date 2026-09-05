import os
import glob
import re
import SimpleITK as sitk


IMG_FOLDER="./data/SABS/img/"
SEG_FOLDER="./data/SABS/label/"
OUT_FOLDER="./tmp_normalized/"

def case_id(path):
    match = re.search(r'(\d+)(?=\.nii(?:\.gz)?$)', os.path.basename(path))
    if match is None:
        raise ValueError(f'Cannot extract case id from {path}')
    return int(match.group(1))


def index_cases(paths, kind):
    indexed = {}
    for path in paths:
        identifier = case_id(path)
        if identifier in indexed:
            raise ValueError(f'Duplicate {kind} case id {identifier}')
        indexed[identifier] = path
    return indexed


images = index_cases(glob.glob(IMG_FOLDER + "/*.nii.gz"), 'image')
labels = index_cases(glob.glob(SEG_FOLDER + "/*.nii.gz"), 'label')
if not images:
    raise ValueError(f'No input images found in {IMG_FOLDER}')
if set(images) != set(labels):
    raise ValueError(
        f'Image/label case ids differ: {sorted(set(images).symmetric_difference(labels))}'
    )
cases = [(images[identifier], labels[identifier]) for identifier in sorted(images)]


# helper function
def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

scan_dir = OUT_FOLDER
LIR = -125
HIR = 275
os.makedirs(scan_dir, exist_ok = True)

reindex = 0
for img_fid, seg_fid in cases:

    img_obj = sitk.ReadImage( img_fid )
    seg_obj = sitk.ReadImage( seg_fid )

    array = sitk.GetArrayFromImage(img_obj)

    array[array > HIR] = HIR
    array[array < LIR] = LIR
    
    intensity_range = array.max() - array.min()
    if intensity_range == 0:
        raise ValueError(f'Cannot normalize constant image {img_fid}')
    array = (array - array.min()) / intensity_range * 255.0
    
    # then normalize this
    
    wined_img = sitk.GetImageFromArray(array)
    wined_img = copy_spacing_ori(img_obj, wined_img)
    
    out_img_fid = os.path.join( scan_dir, f'image_{str(reindex)}.nii.gz' )
    out_lb_fid  = os.path.join( scan_dir, f'label_{str(reindex)}.nii.gz' ) 
    
    # then save
    sitk.WriteImage(wined_img, out_img_fid, True) 
    sitk.WriteImage(seg_obj, out_lb_fid, True) 
    print("{} has been save".format(out_img_fid))
    print("{} has been save".format(out_lb_fid))
    reindex += 1

