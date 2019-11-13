
import sys
import inspect
import time

import numpy as np
# import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from PIL import TiffImagePlugin as tip
from PIL import Image

from IPython.display import display

sys.path.insert(0, '.')
from eq_iter import get_primitives
from impute_color import primitive_2_gray, etg_norm, get_gray_im
from im_scale_products import name_functionhandle_dict, named_function_handles_dict

def nb_imshow(im_arr):
    """ Usage:  nb_imshow_gray(im_arr)
                waffle between PIL and scikit-image image types
    Args:
        im_arr:     np.ndarray h, w, (d = None or 3)
    """
    if isinstance(im_arr, Image.Image):
        # display a Pillow (PIL fork :-) image
        display(im_arr)

    else:
        # display a numpy matrix
        dpi_here = mpl.rcParams['figure.dpi']
        h, w = im_arr.shape[0], im_arr.shape[1]
        fig_size = w / dpi_here, h / dpi_here
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        if len(im_arr.shape) == 2:
            ax.imshow(im_arr, cmap='gray')
        elif len(im_arr.shape) == 3:
            ax.imshow(im_arr)
        plt.show()


def cat_im_list_hori(im_list):
    """ combine a list of PIL images horizontaly
    """
    h = 0
    w = 0
    for im in im_list:
        w += im.size[0]
        h = max(h, im.size[1])

    new_im = tip.Image.new('L', (w, h), color=0)
    start_col = 0
    for im in im_list:
        end_col = start_col + im.size[0]
        box = (start_col, 0, end_col, h)

        new_im.paste(im, box)
        start_col = end_col + 1

    return new_im


def cat_im_list_verti(im_list):
    """ combine a list of PIL images vertically
    """
    h = 0
    w = 0
    for im in im_list:
        h += im.size[1]
        w = max(w, im.size[0])

    new_im = tip.Image.new('L', (w, h), color=0)
    start_row = 0
    for im in im_list:
        end_row = start_row + im.size[0]
        box = (0, start_row, w, end_row)
        new_im.paste(im, box)
        start_row = end_row + 1

    return new_im


def get_results_gray_set(list_tuple, domain_dict):
    """ Usage: new_im = get_results_gray_set(list_tuple, domain_dict)

    """
    ET, Z, Z0 = get_primitives(list_tuple, domain_dict)
    Zd_n2, Zr_n2, ETn_n2 = etg_norm(Z0, Z, ET)

    g_im_et = primitive_2_gray(ETn_n2)
    # complex result vectors: distance component
    g_im_Zd = primitive_2_gray(Zd_n2)

    # complex result vectors: rotation component
    g_im_Zr = primitive_2_gray(Zr_n2)

    im_gray = get_gray_im(ET, Z, Z0)

    im_list = [g_im_et, g_im_Zd, g_im_Zr, im_gray]
    new_im = cat_im_list_hori(im_list)

    return new_im


def display_calc_result(list_tuple, domain_dict):
    """ Usage: (None) display_calc_result(list_tuple, domain_dict)

    """
    fcn_name = domain_dict['function_name']
    p = list_tuple[0][1]

    t0 = time.time()
    new_im = get_results_gray_set(list_tuple, domain_dict)
    tt = time.time() - t0

    print('domain & run constraints:')
    for k, v in domain_dict.items():
        print('%30s: %s' % (k, v))

    print('\n')
    print('%20s: %0.6f seconds\n%20s:' % (fcn_name, tt, 'parameters'), p)
    print('\n%15s%30s%30s%30s' % ('ET', 'Zd', 'Zr', 'All'))
    display(new_im)


def calculate_and_display(domain_dict):
    fcn = named_function_handles_dict[domain_dict['function_name']]
    if 'p' in domain_dict:
        p = domain_dict['p']
    else:
        p = fcn(0.0, None)

    list_tuple = [(fcn, p)]

    display_calc_result(list_tuple, domain_dict)

def display_source(fctn_hndl):
    """ Usage: (None) display_source(fctn_hndl)
                print the source code for a givin function
    """
    S = inspect.getsource(fctn_hndl)
    print(S)


def get_default_iteration_dict(iteration_dict=None):
    """ Usage: iteration_dict = get_default_iteration_dict(iteration_dict=None)

    """
    if iteration_dict is None:
        iteration_dict = {}
    iteration_dict['it_max'] = 64
    iteration_dict['scale_dist'] = 12
    return iteration_dict


def get_default_domain_dict(domain_dict=None):
    """ Usage: domain_dict = get_default_domain_dict((optional:domain_dict))

     Args:
        domain_dict:        (optional) python dict

    Returns:
        domain_dict:        python dict with these keys set or reset
                                domain_dict['center_point'] = 0.0 + 0.0j
                                domain_dict['zoom'] = 0.5
                                domain_dict['theta'] = 0.0
    """
    if domain_dict is None:
        domain_dict = {}
    domain_dict['center_point'] = 0.0 + 0.0j
    domain_dict['zoom'] = 0.5
    domain_dict['theta'] = 0.0

    return domain_dict


def get_test_domain_dict():
    """ Usage: domain_dict = get_center_domain_dict(it_max, scale_dist)

     Args:
        No input arguments

    Returns:
        domain_dict:        python dict with these keys set or reset
                                domain_dict['center_point'] = 0.0 + 0.0j
                                domain_dict['zoom'] = 0.5
                                domain_dict['theta'] = 0.0
                                domain_dict['it_max'] = it_max
                                domain_dict['max_d'] = scale_dist / domain_dict['zoom']
                                domain_dict['n_rows'] = 255
                                domain_dict['n_cols'] = 255
    """
    domain_dict = get_default_iteration_dict()
    domain_dict = get_default_domain_dict(domain_dict)
    domain_dict['n_rows'] = 255
    domain_dict['n_cols'] = 255

    return domain_dict


def domain_update_zoom(domain_dict, zoom_value):
    """ Usage:  domain_dict = domain_update_zoom(domain_dict, new_value)
                check zoom value and update the max distance parameter
    Args:
        domain_dict:    with keys: zoom, max_d and (optional) scale_dist
        zoom_value:     new zoom value > epsilon

    Returns:
        domain_dict:    with zoom and max_d reset
    """
    epsln = np.finfo(np.float).eps
    zoom_value = max(epsln, zoom_value)
    if 'scale_dist' in domain_dict:
        scale_dist = domain_dict['scale_dist']
    else:
        scale_dist = domain_dict['zoom'] * domain_dict['max_d']

    domain_dict['zoom'] = zoom_value
    domain_dict['max_d'] = scale_dist / domain_dict['zoom']

    return domain_dict
