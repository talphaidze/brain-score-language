import numpy as np                                                                                      
from nilearn import datasets, surface
                                                                                                        
out = '/mnt/alphaidz/brain-score-language/atlases'                                                    
fsavg5 = datasets.fetch_surf_fsaverage('fsaverage5')
yeo = datasets.fetch_atlas_yeo_2011()

print(type(yeo['maps']), yeo['maps'] if isinstance(yeo['maps'], str) else '')

lh_yeo = surface.vol_to_surf(yeo['maps'], fsavg5['pial_left'], interpolation='nearest_most_frequent')
rh_yeo = surface.vol_to_surf(yeo['maps'], fsavg5['pial_right'], interpolation='nearest_most_frequent')

print(f"Unique labels LH: {np.unique(lh_yeo)}")
print(f"Unique labels RH: {np.unique(rh_yeo)}")

md_mask = np.concatenate([(lh_yeo == 6), (rh_yeo == 6)])
dmn_mask = np.concatenate([(lh_yeo == 7), (rh_yeo == 7)])

np.save(f'{out}/yeo_md_fsavg5_mask.npy', md_mask)
np.save(f'{out}/yeo_dmn_fsavg5_mask.npy', dmn_mask)

print(f"MD:  {md_mask.sum()} voxels")
print(f"DMN: {dmn_mask.sum()} voxels")