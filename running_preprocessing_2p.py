
import caiman as cm
import numpy as np
import os
import logging
import data_base as db
import matplotlib.pylab as plt

mouse = 1
year = 2022
month = 1
date = 17
example = 2

data_base = db.open_data_base()
selected_entry = db.select(data_base, mouse, year,month,date,example)

if not os.path.isfile(eval(selected_entry.iloc[0]['raw_output'])['main']):
    print('File does not exist')
#load tif file
movie = cm.load(eval(selected_entry.iloc[0]['raw_output'])['main'])
figure, axes = plt.subplots(1)
axes.imshow(np.mean(movie,axis=0), cmap = 'gray')
axes.set_title('Mean Image')

### general dictionary for motion correction
parameters_motion_correction = {'pw_rigid': False, 'save_movie_rig': False,
                                'gSig_filt': (9, 9), 'max_shifts': (25, 25), 'niter_rig': 1,
                                'strides': (48, 48),
                                'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                                'max_deviation_rigid': 15,
                                'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}


