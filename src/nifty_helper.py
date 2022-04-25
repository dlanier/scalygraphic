import warnings
warnings.filterwarnings('ignore')

import time, os, sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from dipy.viz import regtools
from dipy.align.imaffine import AffineMap, MutualInformationMetric
from dipy.align.imaffine import AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D
from dipy.align.transforms import AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric

def reg_a2b(nifty_name_a, nifty_name_b):
    out_data_dict = {}
    
    moving_img = nib.load(nifty_name_a)
    template_img = nib.load(nifty_name_b)
    print(moving_img.shape, template_img.shape)

    moving_data = moving_img.get_data()
    moving_data = moving_data.reshape(moving_data.shape[0:3]) # for hal-data compare
    moving_affine = moving_img.affine
    print('\nOpened:\n%s with shape:'%(nifty_name_a), 
          moving_data.shape, '\nmoving_affine')
    print(moving_affine)
    out_data_dict['moving_data'] = moving_data
    
    template_data = template_img.get_data()
    template_data = template_data.reshape(template_data.shape[0:3]) # hal-data compare
    template_affine = template_img.affine
    print('\nOpened:\n%s with shape:'%(nifty_name_b), 
          template_data.shape, '\ntemplate_affine')
    print(template_affine, '\n')
    out_data_dict['template_data'] = template_data

    identity = np.eye(4)
    affine_map = AffineMap(identity, template_data.shape, template_affine, moving_data.shape, moving_affine)
    resampled = affine_map.transform(moving_data)
    
    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    
    # The optimization strategy
    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, moving_data, 
                                  transform, params0, 
                                  template_affine, moving_affine)
    # translation.affine
    transformed = translation.transform(moving_data)
    transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, 
                            transform, params0, 
                            template_affine, moving_affine, 
                            starting_affine=translation.affine)
    # rigid.affine    
    transformed = rigid.transform(moving_data)    
    transform = AffineTransform3D()
    # Bump up the iterations to get an more exact fit
    affreg.level_iters = [1000, 1000, 100]
    
    moving_affine[:,3]= template_affine[:,3] # ! really !
    
    affine = affreg.optimize(template_data, moving_data, 
                             transform, params0, template_affine, 
                             moving_affine, starting_affine=rigid.affine)
    # affine.affine
    transformed = affine.transform(moving_data)
    
    out_data_dict['affine'] = affine.affine
        
    print('\n\n\t (finally) Transform moving_data with:')
    print('\t\t - template_affine\n', template_affine)
    print('\t\t - moving_affine\n', moving_affine)
    print('\t\t - affine.affine\n', affine.affine)

    out_data_dict['warped_moving'] = transformed
    
    return out_data_dict
    
def reg_a2b_nl(nifty_name_a, nifty_name_b):
    """
                    Reload data:
    """
    out_data_dict = {}
    
    moving_img = nib.load(nifty_name_a)
    template_img = nib.load(nifty_name_b)
    print(moving_img.shape, template_img.shape)

    moving_data = moving_img.get_data()
    moving_data = moving_data.reshape(moving_data.shape[0:3]) # for hal-data compare
    moving_affine = moving_img.affine
    print('\nOpened:\n%s with shape:'%(nifty_name_a), 
          moving_data.shape, '\nmoving_affine')
    print(moving_affine)
    out_data_dict['moving_data'] = moving_data
    
    template_data = template_img.get_data()
    template_data = template_data.reshape(template_data.shape[0:3]) # hal-data compare
    template_affine = template_img.affine
    print('\nOpened:\n%s with shape:'%(nifty_name_b), 
          template_data.shape, '\ntemplate_affine')
    print(template_affine, '\n')
    out_data_dict['template_data'] = template_data

    # The mismatch metric
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # The optimization strategy
    level_iters = [10, 10, 5]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric, 
                                level_iters=level_iters, 
                                sigmas=sigmas, 
                                factors=factors)


    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(template_data, 
                                  moving_data, 
                                  transform, 
                                  params0, 
                                  template_affine, 
                                  moving_affine)

    transform = AffineTransform3D()

    # transform = RigidTransform3D()
    rigid = affreg.optimize(template_data, moving_data, 
                            transform, params0, template_affine, 
                            moving_affine, starting_affine=translation.affine)

    # Bump up the iterations to get an more exact fit
    affreg.level_iters = [1000, 1000, 100]
    affine = affreg.optimize(template_data, 
                             moving_data, 
                             transform, 
                             params0, 
                             template_affine, 
                             moving_affine, 
                             starting_affine=rigid.affine)
    
    print('\nOptimized: affine.affine\n', affine.affine)

    metric = CCMetric(3)

    # The optimization strategy:
    level_iters = [10, 10, 5]
    # Registration object
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    moving_affine[:,3]= template_affine[:,3] # ! really !
    
    mapping = sdr.optimize(template_data, 
                           moving_data, 
                           template_affine, 
                           moving_affine, 
                           affine.affine)
    out_data_dict['affine'] = affine.affine
        
    print('\n\n\t (finally) Transform moving_data with:')
    print('\t\t - template_affine\n', template_affine)
    print('\t\t - moving_affine\n', moving_affine)
    print('\t\t - affine.affine\n', affine.affine)
    
    warped_moving = mapping.transform(moving_data)
    out_data_dict['warped_moving'] = warped_moving
    
    return out_data_dict


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, this_slice in enumerate(slices):
        axes[i].imshow(this_slice.T, cmap="gray", origin="lower")

        
def show_voxel_slices(im_arr, n_frames=4):
    x_max, y_max, z_max = im_arr.shape
    start_frame = 10
    
    step_frame = x_max // n_frames
    slice_list = []
    for i in range(start_frame, x_max, step_frame):
        slice_list.append(im_arr[i, :, :])
        
    show_slices(slice_list)
    
    step_frame = y_max // n_frames
    slice_list = []
    for i in range(start_frame, y_max, step_frame):
        slice_list.append(im_arr[:, i, :])
        
    show_slices(slice_list)
    
    step_frame = z_max // n_frames
    slice_list = []
    for i in range(start_frame, z_max, step_frame):
        slice_list.append(im_arr[:, :, i])
        
    show_slices(slice_list)
        
        
def sho_mi_slices(f_name):
    one_img = nib.load(f_name)
    one_data = one_img.get_data()
    if len(one_data.shape) != 3:
        one_data = one_data.reshape(one_data.shape[0:3])
    
    show_voxel_slices(one_data)
    
def sho_mi_summary(f_name):
    one_img = nib.load(f_name)
    one_data = one_img.get_data()
    print(f_name)
    print(one_data.shape, 'min: %0.3f max: %0.3f'%(one_data.min(), one_data.max()))
    print(one_img.affine)
