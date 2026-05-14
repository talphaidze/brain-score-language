# credit Dufour et al., 2013
# source: https://saxelab.mit.edu/use-our-theory-mind-group-maps/
import numpy as np                                                                                      
from nilearn import datasets, surface, image
                                                                                                        
out = '/mnt/alphaidz/brain-score-language/atlases'                                                    
dl = f'{out}/downloads'
fsavg5 = datasets.fetch_surf_fsaverage('fsaverage5')

tom_regions = ['DMPFC_binary.nii.gz', 'LTPJ_binary.nii.gz', 'MMPFC_binary.nii.gz',
                'PC_binary.nii.gz', 'RSTS_binary.nii.gz', 'RTPJ_binary.nii.gz', 'VMPFC_binary.nii.gz']

lh_mask = np.zeros(10242, dtype=bool)
rh_mask = np.zeros(10242, dtype=bool)

for region in tom_regions:
    img = image.load_img(f'{dl}/{region}')
    lh = surface.vol_to_surf(img, fsavg5['pial_left'], interpolation='nearest_most_frequent')
    rh = surface.vol_to_surf(img, fsavg5['pial_right'], interpolation='nearest_most_frequent')
    lh_mask |= (lh > 0)
    rh_mask |= (rh > 0)
    print(f"{region:<25} LH: {(lh>0).sum():>4}  RH: {(rh>0).sum():>4}")

tom_mask = np.concatenate([lh_mask, rh_mask])
np.save(f'{out}/saxe_tom_fsavg5_mask.npy', tom_mask)
print(f"\nTotal ToM: {tom_mask.sum()} voxels")