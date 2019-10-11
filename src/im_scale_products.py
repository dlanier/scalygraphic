"""
Collection of functions to run image production for machine learning applications

See the Makefile and ../data/run_files/ for usage examples

"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import time
import hashlib
import inspect
import traceback

import numpy as np
import yaml

import skimage.io as im_io

# development running from clone-mount directory or (this) src dir
sys.path.insert(0, '../src/')
sys.path.insert(0, 'scalygraphic/src/')

import zplain as zp
import eq_iter
import deg_0_ddeq
import impute_color as ncp

"""     Constants (lookups):

        name_functionhandle_dict is an enumerated dictionary of the function_name: function_handle
        number_function_name_dict is a dictionary index {name: enumeration_number} of name_functionhandle_dict
"""
name_functionhandle_dict = {k: v for k, v in enumerate(inspect.getmembers(deg_0_ddeq, inspect.isfunction))}
number_function_name_dict = {v[0]: k for k, v in name_functionhandle_dict.items()}

def get_traceback_bottom_line(S):
    """ S = traceback.extract_stack()
    Args:       S
    Returns:
        file_name:
        function_name:
        line_num:
    """
    s_list = str(S[-1]).strip().split('.py')
    file_name = s_list[0].split(os.sep)[-1] + '.py'
    f_name = s_list[-1].strip().split(' in ')[-1].strip().strip('>') + '()'
    line_num = s_list[1].strip().split(' ')[2]

    return file_name, f_name, line_num

def show_equations():
    """
    display the numbered dict of imported equations
    """
    file_name, f_name, line_num = get_traceback_bottom_line(traceback.extract_stack())
    print('\n\t%s with function %s:\n\n\tname_functionhandle_dict.items()'%(file_name, f_name))
    for n, t in name_functionhandle_dict.items():
        print('%03i' % (n), t)

def get_run_directory_and_run_file(args):
    """ Parse the input arguments to get the run_directory and run_file
    Args:
        system args:     -run_directory, -run_file (as below)

    Returns:
        run_directory:      where run_file is expected
        run_file:           yaml file with run parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_directory', type=str)
    parser.add_argument('-run_file', type=str)
    args = parser.parse_args()

    run_directory = args.run_directory
    run_file = args.run_file

    return run_directory, run_file


def get_default_run_parameters(results_dir=None):
    """ default set of run parameters = center area of complex plain

    Args:
         results_dir:   file-write destination

    Returns:
        run_parameters: default set of run parameters for calling iteration function

    """
    run_parameters = {}
    if results_dir is None:
        run_parameters['dir_path'] = os.getcwd()
    else:
        run_parameters['dir_path'] = results_dir
    run_parameters = get_default_domain_dict(run_parameters)
    run_parameters = get_default_iteration_dict(run_parameters)
    run_parameters['max_d'] = run_parameters['scale_dist'] / run_parameters['zoom']
    run_parameters['n_rows'] = 256
    run_parameters['n_cols'] = 256

    return run_parameters


def scaled_images_dataset(run_parameters):
    """ assemble input arguments and call scaled_images_dataset() function
    Args:
        run_parameters:   input arguments for write_n_image_sets

    Returns:
        nothing:
    """
    number_of_image_sets = run_parameters['number_of_image_sets']
    it_max = run_parameters['it_max']
    scale_dist = run_parameters['scale_dist']
    small_scale = [run_parameters['small_scale_rows'], run_parameters['small_scale_cols']]
    large_scale = [run_parameters['large_scale_rows'], run_parameters['large_scale_cols']]
    results_directory = run_parameters['results_directory']

    if 'greyscale' in run_parameters:
        greyscale = run_parameters['greyscale']
    else:
        greyscale = False
        run_parameters['greyscale'] = greyscale

    if 'use_one_eq' in run_parameters:
        use_one_eq = run_parameters['use_one_eq']
    else:
        use_one_eq = False
        run_parameters['use_one_eq'] = use_one_eq

    if 'hash_list' in run_parameters:
        hash_list = run_parameters['hash_list']
        # read & saving hash list function pending
    else:
        hash_list = []

    hash_list = write_n_image_sets(number_of_image_sets,
                                   it_max,
                                   scale_dist,
                                   small_scale,
                                   large_scale,
                                   results_directory,
                                   hash_list,
                                   greyscale,
                                   use_one_eq)

    print('\n%i pairs written \n'%(len(hash_list)))


def get_default_iteration_dict(iteration_dict=None):
    if iteration_dict is None:
        iteration_dict = {}
    iteration_dict['it_max'] = 64
    iteration_dict['scale_dist'] = 12
    return iteration_dict


def get_default_domain_dict(domain_dict=None):
    if domain_dict is None:
        domain_dict = {}
    domain_dict['center_point'] = 0.0 + 0.0j
    domain_dict['zoom'] = 0.5
    domain_dict['theta'] = 0.0
    return domain_dict


def get_run_parameters(run_directory, run_file):
    """ Read the input arguments into a dictionary
    Args:
        run_directory:      where run_file is expected
        run_file:           yaml file with run parameters

    Returns:
        run_parameters:     python dictionary of run parameters
    """
    run_file_name = os.path.join(run_directory, run_file)
    with open(run_file_name, 'r') as fh:
        run_parameters = yaml.load(fh)
    run_parameters['run_directory'] = run_directory
    run_parameters['run_file'] = run_file

    return run_parameters


def get_rand_eq_p_set():
    """ get a random equation and parameter set from the deg_0_ddeq module
    (No Args:)
    
    Returns:
        tuple:      (function_name, function_handle, parameter_set)
    """
    n = np.random.randint(0,len(name_functionhandle_dict),1)
    fcn_name, fcn = name_functionhandle_dict[n[0]]
    p = fcn(0.0, None)

    return (fcn_name, fcn, p)


def get_eq_by_name(fcn_name):
    """ get the function handle from the function name
    
    Args:
        fcn_name:   name of a function in deg_0_ddeq
        
    Returns:
        fcn_handle: callable function Z = fcn_name(Z, p, (Z0), (ET))
        
    """
    if fcn_name in number_function_name_dict:
        return name_functionhandle_dict[number_function_name_dict[fcn_name]][1]
    else:
        return None


def get_random_domain(bounds_dict=None):
    """ Usage: 
    domain_dict = get_random_domain(h, w, bounds_dict)
    
    Args:
        bounds_dict:    min - max limits for keys eg.
                            CP_magnitude_limits = {'min': 0, 'max': 7}
                            ZM_limits           = {'min': np.finfo(float).eps, 'max': 2}
                            theta_limits        = {'min': 0, 'max':2 * np.pi}
                        
    Returns:
        domain_dict:    with keys:
                            center_point
                            zoom
                            theta
    """
    domain_dict = {}
    if bounds_dict is None:
        CP_magnitude_limits =   {'min': 0, 'max': 2}
        ZM_limits =             {'min': np.finfo(float).eps, 'max': 1}
        theta_limits =          {'min': 0, 'max':2 * np.pi}
    else:
        CP_magnitude_limits =   bounds_dict['CP_magnitude_limits']
        ZM_limits =             bounds_dict['ZM_limits']
        theta_limits =          bounds_dict['theta_limits']

    r = np.random.uniform(low=0.0, high=2*np.pi) * 0.0+1.0j
    m = np.random.uniform(low=CP_magnitude_limits['min'], high=CP_magnitude_limits['max'])
    domain_dict['center_point'] = m*np.exp(r)
    domain_dict['zoom'] = np.random.uniform(low=ZM_limits['min'], high=ZM_limits['max'])
    domain_dict['theta'] = np.random.uniform(low=theta_limits['min'], high=theta_limits['max'])
    domain_dict['n_rows'] = 255
    domain_dict['n_cols'] = 255
    
    return domain_dict


def sha256sum(s):
    """ convert a string to a 256 bit hash key as a string
    
    Args:
        s:          string
        
    Returns:
        hash_key:   256 bit hex as string

    """
    h  = hashlib.sha256()
    h.update(bytes(s, 'ascii'))
    
    return h.hexdigest()


def hash_parameters(domain_dict, fcn_name, p):
    """ get a hash value of equation production parameters to compare uniqueness
    
    Args:
        domain_dict:    parameters defining numerical domain on the complex plain
        fcn_name:       equation function name
        p:              equation parameter inputs
        
    Returns:
        hash_key:   256 bit hex as string
    """
    N_DEC = 15
    f = zp.get_frame_from_dict(domain_dict)
    s = zp.complex_frame_dict_to_string(f, N_DEC) + '\n' + fcn_name
    if isinstance(p, list):
        p_str = ''
        for p_n in p:
            p_str += zp.complex_to_string(p_n, N_DEC)
    else:
        p_str = zp.complex_to_string(p, N_DEC)
    
    s += p_str

    if 'it_max' in domain_dict:
        s += '\n%s: %i\n'%('it_max',domain_dict['it_max'])

    if 'max_d' in domain_dict:
        s += '\n%s: %0.6f\n'%('max_d',domain_dict['max_d'])

    return sha256sum(s)


def now_name(prefi_str=None, suffi_str=None):
    """ get a human readable time stamp name 
    
    Args:
        prefi_str:
        suffi_str:
        
    Returns:
        now_string: prefi_str + '_' + formatted_time_string + suffi_str
                    eg.  myfilebasename_Mon_23_Sep_2019_06_06_54.tiff
    """
    t0 = time.time()
    t_dec = t0 - np.floor(t0)
    ahora_nombre = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()) 
    if prefi_str is None: prefi_str = ''
    if suffi_str is None: suffi_str = ''
        
    return prefi_str + '_' + ahora_nombre + suffi_str


def write_n_image_sets(number_of_image_sets, it_max, scale_dist,
                       small_scale, large_scale,
                       results_directory, hash_list, greyscale=False, use_one_eq=False):
    """ calculate and write a set of unique images at two scales
    Args:
        number_of_image_sets:   number of pairs of images
        it_max:                 maximum number of iterations
        scale_dist:             escape boundry in units of domain diagonal distance
        small_scale:            [h, w] for smaller copy of image
        large_scale:            [h, w] for larger copy of image
        results_directory:      directory will be written if DNE
        hash_list               empty list []  OR  list returned by this function - to avoid duplicates
        
    Returns:
        hash_list:          list of parameter encodings to insure uniqueness
                            (if producing large dataset over multiple sessions or locations)
                        
    Writes number_of_image_sets image pairs like:
                        asdregshaaldfkaproiapw_small.jpg
                        asdregshaaldfkaproiapw_large.jpg
    """
    if hash_list is None:
        hash_list = []
        
    if os.path.isdir(results_directory) == False:
        os.makedirs(results_directory)
                
    print(now_name('Write %i image pairs to \n%s\nStart '%(number_of_image_sets, results_directory)))
    
    if len(hash_list) > 0:
        print('checking duplicates using input hash_list size = %i'%(len(hash_list)))
    else:
        print('new hash list started')

    k_do = 0
    fcn_name, eq, p = get_rand_eq_p_set()
    for k_do in range(number_of_image_sets):
        if use_one_eq == False:
            fcn_name, eq, p = get_rand_eq_p_set()

        domain_dict = get_random_domain()
        domain_dict['it_max'] = it_max
        domain_dict['max_d'] = scale_dist / domain_dict['zoom']

        hash_idx = hash_parameters(domain_dict, fcn_name, p)
        if hash_idx in hash_list:
            pass
        else:
            hash_list.append(hash_idx)
            domain_dict['n_rows'] = small_scale[0]
            domain_dict['n_cols'] = small_scale[1]

            list_tuple = [(eq, (p))]

            t0 = time.time()
            ET, Z, Z0 = eq_iter.get_primitives(list_tuple, domain_dict)

            file_name = os.path.join(results_directory, hash_idx + '_' + 'small.tif')
            if greyscale == True:
                I = ncp.get_uint16_gray(ET, Z, Z0)
                #                   requires tiff (else cast to 8 bit with warn you)
                im_io.imsave(file_name, I)
            else:
                I = ncp.get_im(ET, Z, Z0)
                I.save(file_name)

            domain_dict['n_rows'] = large_scale[0]
            domain_dict['n_cols'] = large_scale[1]

            ET, Z, Z0 = eq_iter.get_primitives(list_tuple, domain_dict)

            file_name = os.path.join(results_directory, hash_idx + '_' + 'large.tif')
            if greyscale == True:
                I = ncp.get_uint16_gray(ET, Z, Z0)
                #                   requires tiff (else cast to 8 bit with warn you)
                im_io.imsave(file_name, I)
            else:
                I = ncp.get_im(ET, Z, Z0)
                I.save(file_name)

            print('\n%3i of %3i) %s\t\t'%(k_do+1, number_of_image_sets, fcn_name),
                  '%0.3f seconds (large & small image written)\n'%(time.time() - t0),
                  hash_idx)
                
    print('\n', now_name('%i pairs written,\nFinished '%(k_do + 1)))
    
    return hash_list
