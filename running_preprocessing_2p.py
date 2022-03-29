
import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_cropper, run_motion_correction, run_alignment, run_source_extraction, cropping_interval
from caiman.source_extraction.cnmf.cnmf import load_CNMF

figure_path = '/scratch/melisa/photon2_test/figures/'

mouse = 3
year = 2022
month = 3
date = 23
example = 0


states = db.open_data_base()
selected_row = db.select(states, mouse, year,month,date)

n_processes = psutil.cpu_count()
#cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)

if not os.path.isfile(eval(selected_row.iloc[0]['raw_output'])['main']):
    print('File does not exist')
#load tif file
# movie = cm.load(eval(selected_row.iloc[0]['raw_output'])['main'])
# figure, axes = plt.subplots(1)
# axes.imshow(np.mean(movie,axis=0), cmap = 'gray')
# axes.set_title('Mean Image')

parameters_cropping = cropping_interval()  # check whether it is better to do it like this or to use the functions get
### run cropper
for i in range(2):
    row = selected_row.iloc[i]
    updated_row = run_cropper(row, parameters_cropping)
    states = db.update_data_base(states, updated_row)
    db.save_analysis_states_database(states)

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,0,0,0])

### general dictionary for motion correction
parameters_motion_correction = {'pw_rigid': False, 'save_movie_rig': False,
                                'gSig_filt': (5, 5), 'max_shifts': (25, 25), 'niter_rig': 1,
                                'strides': (48, 48),
                                'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                                'max_deviation_rigid': 15,
                                'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

### run motion correction
for i in range(2):
    row = selected_rows.iloc[i]
    updated_row = run_motion_correction(row, parameters_motion_correction, dview)
    states = db.update_data_base(states, updated_row)
    db.save_analysis_states_database(states)

parameters_alignment = {'make_template_from_trial': '1', 'gSig_filt': (5, 5), 'max_shifts': (50, 50), 'niter_rig': 1,
                        'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                        'max_deviation_rigid': 15, 'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True,
                        'border_nan': 'copy'}

### run alignment
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0])
new_selected_rows = run_alignment(selected_rows, parameters_alignment, dview)

for i in range(2):
    new_name = db.replace_at_index1(new_selected_rows.iloc[i].name, 7, new_selected_rows.iloc[i].name[5])
    row_new = new_selected_rows.iloc[i].copy()
    row_new.name = new_name
    states = db.update_data_base(states, row_new)
    db.save_analysis_states_database(states)


gSig = 7
gSiz = 4 * gSig + 1
min_corr = 0.6
min_pnr = 6
##trial_wise_parameters
parameters_source_extraction = {'fr': 34, 'decay_time': 0.1,
                                'min_corr': min_corr,
                                'min_pnr': min_pnr, 'p': 1, 'K': None, 'gSig': (gSig, gSig),
                                'gSiz': (gSiz, gSiz),
                                'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
                                'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0,
                                'ssub_B': 2,
                                'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
                                'method_deconvolution': 'oasis', 'update_background_components': True,
                                'center_psf': True, 'border_pix': 0, 'normalize_init': False,
                                'del_duplicates': True, 'only_init': True}


### run source extraction
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,0])

mouse_row_new =run_source_extraction(selected_rows.iloc[0], parameters_source_extraction, states, dview, multiple_files= True)
states = db.update_data_base(states, mouse_row_new)
db.save_analysis_states_database(states)


cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

figure, axes = plt.subplots()
axes.imshow(cn_filter,cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes.plot(*v.T, c='r')
    counter = counter + 1
figure.savefig(figure_path + 'source_extraction_ma_2022_3_23_example.png')

figure, axes = plt.subplots()
C_0 = cnm.estimates.C.copy()
C_0[0] += C_0[0].min()
for i in range(45, len(C_0)):
    C_0[i] += C_0[i].min() + C_0[:i].max()
    axes.plot(C_0[i], c = 'k')
figure.set_size_inches([15., 25])
figure.savefig(figure_path + 'source_extraction_ma_2022_3_14_traces_example.png')
