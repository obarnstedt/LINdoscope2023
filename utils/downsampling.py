import caiman as cm
import glob
import os
from tqdm import tqdm

def ds_folder(dir, filename='', filetype='.tif', ext='_ds', temp_ds=1/3, spatial_ds=1/3, out_folder=None, overwrite=False):
    files = [f for f in glob.glob(os.path.join(dir, '*'+filetype)) \
             if not ext in os.path.basename(f) and filename in f]  # collecting all raw files
    print(f'Processing files {files}...')

    for file in tqdm(files):
        if os.path.exists(file[:-4]+ext+filetype) and overwrite==False:
            continue
        else:
            movie_full = cm.load(file)
            try:
                movie_ds = movie_full.resize(fx=spatial_ds, fy=spatial_ds, fz=temp_ds)
                if out_folder:
                    movie_ds.save(os.path.join(out_folder, os.path.basename(file)[:-4]+ext+filetype))
                else:
                    movie_ds.save(file[:-4]+ext+filetype)
            except:
                pass