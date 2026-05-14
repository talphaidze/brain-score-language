import nibabel as nib
import numpy as np
from nilearn import surface, datasets

fsavg = datasets.fetch_surf_fsaverage('fsaverage')
fsavg5 = datasets.fetch_surf_fsaverage('fsaverage5')

# Load full-res annot
lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot('/mnt/alphaidz/brain-score-language/atlases/downloads/lh.HCP-MMP1.annot')
rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot('/mnt/alphaidz/brain-score-language/atlases/downloads/rh.HCP-MMP1.annot')
lh_names = [n.decode() if isinstance(n, bytes) else n for n in lh_names]
rh_names = [n.decode() if isinstance(n, bytes) else n for n in rh_names]

from scipy.spatial import cKDTree

def resample_surface_nn(labels_full, mesh_full_path, mesh_5_path):
    """Nearest-neighbor resample from fsaverage to fsaverage5."""
    coords_full = surface.load_surf_mesh(mesh_full_path).coordinates
    coords_5 = surface.load_surf_mesh(mesh_5_path).coordinates
    tree = cKDTree(coords_full)
    _, idx = tree.query(coords_5)
    return labels_full[idx]

lh_labels_5 = resample_surface_nn(lh_labels, fsavg['pial_left'], fsavg5['pial_left'])
rh_labels_5 = resample_surface_nn(rh_labels, fsavg['pial_right'], fsavg5['pial_right'])

# Now extract regions
def make_mask(lh, rh, lh_names, rh_names, regions):
    lh_mask = np.isin(lh, [lh_names.index(f'L_{r}_ROI') for r in regions if f'L_{r}_ROI' in lh_names])
    rh_mask = np.isin(rh, [rh_names.index(f'R_{r}_ROI') for r in regions if f'R_{r}_ROI' in rh_names])
    return np.concatenate([lh_mask, rh_mask])

out = '/mnt/alphaidz/brain-score-language/atlases'

# Visual
np.save(f'{out}/glasser_v2_fsavg5_mask.npy', make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names,
['V2']))
np.save(f'{out}/glasser_v4_fsavg5_mask.npy', make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names,
['V4']))
np.save(f'{out}/glasser_it_fsavg5_mask.npy', make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names,
['TE2a', 'TE2p'])) # https://pmc.ncbi.nlm.nih.gov/articles/PMC6887748/

# Auditory
np.save(f'{out}/glasser_a2_fsavg5_mask.npy', make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names,
['LBelt', 'MBelt'])) # https://en.wikipedia.org/wiki/Auditory_cortex

# Prefrontal
np.save(f'{out}/glasser_dlpfc_fsavg5_mask.npy', make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names,
['46', '9-46d', 'p9-46v', 'a9-46v', '9a', '9p'])) # https://pmc.ncbi.nlm.nih.gov/articles/PMC10917992/pdf/fnimg-03-1339244.pdf

print("Done! Verify against existing:")
a1_new = make_mask(lh_labels_5, rh_labels_5, lh_names, rh_names, ['A1'])
a1_old = np.load(f'{out}/glasser_a1_fsavg5_mask_bool_fixed.npy')
print(f"A1 match: {(a1_new == a1_old).mean()*100:.1f}%")

