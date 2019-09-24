"""
Collection of functions to run image production for machine learning applications

See the Makefile and ../data/run_files/ for usage examples
"""
import os
import sys
import argparse
import time
import hashlib
import inspect
from tempfile import TemporaryDirectory

import numpy as np
import PIL

import yaml

# development running from clone-mount directory or (this) src dir
sys.path.insert(0, '../src/')
sys.path.insert(0, 'scalygraphic/src/')

import zplain as zp
import eq_iter
import deg_0_ddeq
import numcolorpy as ncp

"""     Constants (lookups):

        EQUS_DICT is an enumerated dictionary of the functions in module deg_0_ddeq
        EQUS_DICT_NAMED_IDX is a dictionary index {name: enumeration_number} of EQUS_DICT
"""
EQUS_DICT = {k: v for k, v in enumerate(inspect.getmembers(deg_0_ddeq, inspect.isfunction))}
EQUS_DICT_NAMED_IDX = {v[0]: k for k, v in EQUS_DICT.items()}

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
    n = np.random.randint(0,len(EQUS_DICT),1)
    fcn_name, fcn = EQUS_DICT[n[0]]
    p = fcn(0.0, None)

    return (fcn_name, fcn, p)

def get_eq_by_name(fcn_name):
    """ get the function handle from the function name
    
    Args:
        fcn_name:   name of a function in deg_0_ddeq
        
    Returns:
        fcn_handle: callable function Z = fcn_name(Z, p, (Z0), (ET))
        
    """
    if fcn_name in EQUS_DICT_NAMED_IDX:
        return EQUS_DICT[EQUS_DICT_NAMED_IDX[fcn_name]][1]
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
    
    return sha256sum(s)

def get_im(ET, Z, Z0, domain_dict):
    """ get a color image from  the products of the escape-time-algorithm Using HSV - RGB model:
    ETn         normalized escape time matrix           Hue
    Zr          normalized rotation of |Z - Z0|         Saturation
    Zd          normalized magnitude of |Z - Z0|        Value
    
    Args:
        ET:     (Integer) matrix of the Escape Times    
        Z:      (complex) matrix of the final vectors   
        Z0:     (complex) matrix of the starting plane
        
    Returns:
        I:      RGB PIL image

    """
    
    Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)

    A = np.zeros((domain_dict['n_rows'], domain_dict['n_cols'],3))
    A[:,:,0] += ETn     # Hue
    A[:,:,1] += Zr      # Saturation
    A[:,:,2] += Zd      # Value
    I = PIL.Image.fromarray(np.uint8(A * 255), 'HSV').convert('RGB')
    
    return I

def get_gray_im(ET, Z, Z0, et_gray=False):
    """ get a gray-scale image from the products of the escape-time-algorithm
    
    Args:
        ET:     (Integer) matrix of the Escape Times
        Z:      (complex) matrix of the final vectors
        Z0:     (complex) matrix of the starting plane
        
    Returns:
        I:      grayscale PIL image
    """
    Zd, Zr, ETn = ncp.etg_norm(Z0, Z, ET)
    
    if et_gray:
        I = PIL.Image.fromarray(np.uint8(ETn * 255), 'L')
    else:
        I = PIL.Image.fromarray(np.uint8(Zd * 255), 'L')
    
    return I

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

def scaled_images_dataset(run_parameters):
    n_2_do = run_parameters['n_2_do']
    iteration_dict = run_parameters['iteration_dict']
    small_scale = run_parameters['run_parameters']
    large_scale = run_parameters['large_scale']
    output_dir = run_parameters['output_dir']

    if 'hash_list' in run_parameters:
        hash_list = run_parameters['hash_list']
        # read & saving hash list function pending
    else:
        hash_list = []

    hash_list = write_n_image_sets(n_2_do, iteration_dict, small_scale, large_scale, output_dir, hash_list)

    print('\n%i pairs written \n'%(len(hash_list)))

def write_n_image_sets(n_2_do, iteration_dict, small_scale, large_scale, output_dir, hash_list=None):
    """
    Args:
        n_2_do:         number of pairs of images
        iteration_dict: {'it_max': 64, 'scale_dist': 10} escape-time-algorithm iteration limits
        small_scale:    [h, w] for smaller copy of image
        large_scale:    [h, w] for larger copy of image
        output_dir:     directory will be written if DNE
        hash_list       empty list []  OR  list returned by this function - to avoid duplicates
        
    Returns:
        hash_list:      list of parameter encodings to insure uniqueness 
                        if producing large dataset over multiple sessions or locations
                        
    Writes n_2_do image pairs like:
                        asdregshaaldfkaproiapw_small.jpg
                        asdregshaaldfkaproiapw_large.jpg
    """
    if hash_list is None:
        hash_list = []
        
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
                
    print(now_name('Write %i image pairs to \n%s\nStart '%(n_2_do, output_dir)))
    
    if len(hash_list) > 0:
        print('checking duplicates using input hash_list size = %i'%(len(hash_list)))
    else:
        print('new hash list started')

    with TemporaryDirectory() as test_temporary_dir:

        for k_do in range(n_2_do):
            fcn_name, eq, p = get_rand_eq_p_set()
            domain_dict = get_random_domain()

            domain_dict['it_max'] = iteration_dict['it_max']
            domain_dict['max_d'] = iteration_dict['scale_dist'] / domain_dict['zoom']

            hash_idx = hash_parameters(domain_dict, fcn_name, p)
            if hash_idx in hash_list:
                pass
            else:
                hash_list.append(hash_idx)
                domain_dict['n_rows'] = small_scale[0]
                domain_dict['n_cols'] = small_scale[1]
                domain_dict['dir_path'] = test_temporary_dir

                list_tuple = [(eq, (p))]

                t0 = time.time()
                ET, Z, Z0 = eq_iter.get_primitives(list_tuple, domain_dict)
                I = get_im(ET, Z, Z0, domain_dict)
                file_name = os.path.join(output_dir, hash_idx + '_' + 'small.jpg')
                I.save(file_name)

                domain_dict['n_rows'] = large_scale[0]
                domain_dict['n_cols'] = large_scale[1]

                ET, Z, Z0 = eq_iter.get_primitives(list_tuple, domain_dict)
                I = get_im(ET, Z, Z0, domain_dict)
                file_name = os.path.join(output_dir, hash_idx + '_' + 'large.jpg')
                I.save(file_name)
                print('\n%3i of %3i) %s\t\t'%(k_do+1, n_2_do, fcn_name), 
                      '%0.3f seconds (large & small image written)\n'%(time.time() - t0), 
                      hash_idx)
                
    print('\n', now_name('%i pairs written,\nFinished '%(k_do + 1)))
    
    return hash_list
