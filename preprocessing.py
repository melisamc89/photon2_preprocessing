'''
Created on Wed Jan 19 2022
@Melisa

Create a set of function to save the processed data using Caiman
'''

import os
import logging
import data_base as db
import numpy as np
import pickle
import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params


def run_motion_correction(entry,parameters,dview):

    '''
    Runs motion correction and returns the modified entry for the data base
    :param entry:
    :param parameters:
    :param dview:
    :return:
    '''

    entry_local = entry.copy()
    name = entry_local.name
    data_base = db.open_data_base()

    if not parameters['pw_rigid']:
        parameters['save_movie_rig'] = True

    input_tif_file_path = eval(entry_local.loc['raw_output'])['main']

    if not os.path.isfile(input_tif_file_path):
        logging.error('File not found. Cancelling motion correction.')
        return entry_local

    ### write used parameters in the data base
    entry_local.loc['motion_correction_parameters'] = str(parameters)
    ### update motion correction version in the data base
    entry_local = db.modify_data_base_entry(entry_local, data_base, 1)
    name = entry_local.name

    ### create output data for data base
    data_dir = '/ceph/imaging1/melisa/photon2_test/data_processing/motion_correction/'  ### replace this in OS
    file_name = db.create_file_name(1, name)
    output_meta_pkl_file_path = data_dir + db.create_file_path('meta', name, 'motion_correction', '.pkl')

    # Create a dictionary with the output
    output = { 'meta':  output_meta_pkl_file_path}

    # Calculate movie minimum to subtract from movie
    min_mov = np.min(cm.load(input_tif_file_path))
    # Apply the parameters to the CaImAn algorithm
    caiman_parameters = parameters.copy()
    caiman_parameters['min_mov'] = min_mov
    opts = params.CNMFParams(params_dict = caiman_parameters)

    # Rigid motion correction (in both cases)
    logging.info(f'{name} Performing rigid motion correction')
    # Create a MotionCorrect object
    mc = MotionCorrect([input_tif_file_path], dview=dview, **opts.get_group('motion'))
    # Perform rigid motion correction
    mc.motion_correct_rigid(save_movie=parameters['save_movie_rig'], template=None)
    # Obtain template, rigid shifts and border pixels
    total_template_rig = mc.total_template_rig
    shifts_rig = mc.shifts_rig

    if parameters['save_movie_rig']:
        # Load the movie saved by CaImAn, which is in the wrong
        # directory and is not yet cropped
        logging.info(f'{name} Loading rigid movie for cropping')
        m_rig = cm.load(mc.fname_tot_rig[0])
        logging.info(f'{name} Loaded rigid movie for cropping')
        # Get the cropping points determined by the maximal rigid shifts
        x_, _x, y_, _y = get_crop_from_rigid_shifts(shifts_rig)
        # Crop the movie
        logging.info(f'{name} Cropping and saving rigid movie with cropping points: [x_, _x, y_, _y] = {[x_, _x, y_, _y]}')
        m_rig = m_rig.crop(x_, _x, y_, _y, 0, 0)
        # Save the movie
        fname_tot_rig = m_rig.save(data_dir + 'main/' + file_name + '_rig' + '.mmap', order='C')
        logging.info(f'{name} Cropped and saved rigid movie as {fname_tot_rig}')
        # Remove the remaining non-cropped movie
        os.remove(mc.fname_tot_rig[0])

    if parameters['pw_rigid']:
        logging.info(f'{name} Performing piecewise-rigid motion correction')
        # Perform non-rigid (piecewise rigid) motion correction. Use the rigid result as a template.
        mc.motion_correct_pwrigid(save_movie=True, template=total_template_rig)
        # Obtain template and filename
        total_template_els = mc.total_template_els
        fname_tot_els = mc.fname_tot_els[0]

        # Load the movie saved by CaImAn, which is in the wrong
        # directory and is not yet cropped
        logging.info(f'{name} Loading pw-rigid movie for cropping')
        m_els = cm.load(fname_tot_els)
        logging.info(f'{name} Loaded pw-rigid movie for cropping')
        # Get the cropping points determined by the maximal rigid shifts
        x_, _x, y_, _y = get_crop_from_pw_rigid_shifts(np.array(mc.x_shifts_els),
                                                       np.array(mc.y_shifts_els))
        # Crop the movie
        logging.info(f'{name} Cropping and saving pw-rigid movie with cropping points: [x_, _x, y_, _y] = {[x_, _x, y_, _y]}')
        m_els = m_els.crop(x_, _x, y_, _y, 0, 0)
        # Save the movie
        fname_tot_els = m_els.save(data_dir + 'main/' + file_name + '_els' + '.mmap', order='C')
        logging.info(f'{name} Cropped and saved rigid movie as {fname_tot_els}')
        # Remove the remaining non-cropped movie
        os.remove(mc.fname_tot_els[0])

        # Store the total path in output
        output['main'] = fname_tot_els


    # Write necessary variables to the trial index and row
    entry_local.loc['motion_correction_output'] = str(output)
    entry_local.loc['motion_correction_parameters'] = str(parameters)

    return entry_local

def get_crop_from_rigid_shifts(shifts_rig):
    x_ = int(round(abs(np.array(shifts_rig)[:, 1].max()) if np.array(shifts_rig)[:, 1].max() > 0 else 0))
    _x = int(round(abs(np.array(shifts_rig)[:, 1].min()) if np.array(shifts_rig)[:, 1].min() < 0 else 0))
    y_ = int(round(abs(np.array(shifts_rig)[:, 0].max()) if np.array(shifts_rig)[:, 0].max() > 0 else 0))
    _y = int(round(abs(np.array(shifts_rig)[:, 0].min()) if np.array(shifts_rig)[:, 0].min() < 0 else 0))
    return x_, _x, y_, _y


def get_crop_from_pw_rigid_shifts(x_shifts_els, y_shifts_els):
    x_ = int(round(abs(x_shifts_els.max()) if x_shifts_els.max() > 0 else 0))
    _x = int(round(abs(x_shifts_els.min()) if x_shifts_els.min() < 0 else 0))
    y_ = int(round(abs(y_shifts_els.max()) if y_shifts_els.max() > 0 else 0))
    _y = int(round(abs(x_shifts_els.min()) if x_shifts_els.min() < 0 else 0))
    return x_, _x, y_, _y