# %load ../scalygraphic/zplain.py
"""
complex plane under a grid of pixels
with printing interface
"""
import numpy as np

DEFAULT_PLAIN = {'center_point': 0 + 0j,
                 'zoom': 1/2,
                 'theta': 0,
                 'n_rows': 32,
                 'n_cols': 32}

def get_default_frame():
    """  """
    def_dict = DEFAULT_PLAIN
    complex_frame = get_frame_from_dict(DEFAULT_PLAIN)
    default_frame = {**complex_frame, **def_dict}
    
    return default_frame


def get_frame_from_dict(def_dict=None):
    """ complex_frame, def_dict = get_frame_from_dict(def_dict)
        legacy wrapper function.
    Args:
        def_dict: definition dictionary with keys:
                    'center_point', 'zoom', 'theta', 'n_rows', 'n_cols'
    Returns:
        complex_frame:
        def_dict:
    """
    if def_dict is None or not isinstance(def_dict, dict):
        def_dict = DEFAULT_PLAIN
        
    complex_frame = get_complex_frame(
        def_dict['center_point'],
        def_dict['zoom'],
        def_dict['theta'],
        def_dict['n_rows'],
        def_dict['n_cols'])

    return complex_frame


def get_complex_frame(CP, ZM, theta, h=1, w=1):
    """ get the complex numbers at ends and centers of a frame  """
    frame_dict = {'center_point':CP}
    if w >= h:
        frame_dict['top_center'] = np.exp(1j*(np.pi/2 + theta))/ZM
        frame_dict['right_center'] = (w/h) * np.exp(1j * theta) / ZM
    else:
        frame_dict['top_center'] = (h/w) * np.exp(1j*(np.pi/2 + theta)) / ZM
        frame_dict['right_center'] = np.exp(1j * theta) / ZM

    frame_dict['bottom_center'] = frame_dict['top_center'] * -1
    frame_dict['left_center'] = frame_dict['right_center'] * -1
    frame_dict['upper_right'] = frame_dict['right_center'] + frame_dict['top_center']
    frame_dict['bottom_right'] = frame_dict['right_center'] + frame_dict['bottom_center']

    frame_dict['upper_left'] = frame_dict['left_center'] + frame_dict['top_center']
    frame_dict['bottom_left'] = frame_dict['left_center'] + frame_dict['bottom_center']

    for k in frame_dict.keys():
        frame_dict[k] = frame_dict[k] + CP

    return frame_dict


def get_complex_pixel_matrix(frame_dict):
    """ complex_pixel_matrix = get_complex_pixel_matrix(frame_dict) 
    Args:
        frame_dict:             as defined herein
        
    Returns:
        complex_pixel_matrix:   matrix of pixels over points on the complex plain defined by frame_dict
    """
    n_rows = frame_dict['n_rows']
    n_cols = frame_dict['n_cols']
    
    complex_pixel_matrix = np.zeros((n_rows, n_cols), dtype=complex)
    left_style = np.linspace(frame_dict['upper_left'], frame_dict['bottom_left'], frame_dict['n_rows'])
    right_style = np.linspace(frame_dict['upper_right'], frame_dict['bottom_right'], frame_dict['n_rows'])
    for row_number in range(n_rows):
        complex_pixel_matrix[row_number,:] = np.linspace(left_style[row_number], 
                                                         right_style[row_number], 
                                                         n_cols)
    
    return complex_pixel_matrix


def complex_to_string(Z, n_dec=3):
    """ Z_string = complex_to_string(Z, N_DEC)
    complex number to string with decimal places N_DEC 
    Args:
        Z:          a complex number
        n_dec:      number of decimal places for output
    Returns:
        Z_str:      the complex number as a string
    """
    fmt = '.{}f'.format(n_dec)
    cfmt = "{z.real:" + fmt + "}{z.imag:+" + fmt + "}j"

    return cfmt.format(z=Z)


def show_complex_matrix(Z0,N_DEC=3):
    """ display a complex matrix or array """
    SPC = ' ' * 2
    if Z0.shape[0] == Z0.size:
        row_str = ''
        for col in range(0, Z0.shape[0]):
            row_str += complex_to_string(Z0[col], N_DEC) + SPC + '\n'
        print(row_str)
   
    else:
        for row in range(0,Z0.shape[0]):
            row_str = ''
            for col in range(0, Z0.shape[1]):
                row_str += complex_to_string(Z0[row, col], N_DEC) + SPC

            print(row_str)

            
def rnd_lambda(s=1):
    """ rnd_lambda, lambda_dict = rnd_lambda(s=1)
    random parameters s.t. a*d - b*c = s """
    b = np.random.random()
    c = np.random.random()
    ad = b*c + 1
    a = np.random.random()
    d = ad / a
    [a, b, c, d] = [a, b, c, d] * s
    abcd_dict = {'a': a, 'b': b, 'c': c, 'd': d}
    abcd_set = np.array([a, b, c, d])
    
    return abcd_set, abcd_dict


def complex_frame_dict_to_string(frame_dict, N_DEC=4):
    """ get a formatted list of strings """
    
    print_order = ['upper_left', 'top_center', 'upper_right',
                   'left_center', 'center_point', 'right_center',
                  'bottom_left', 'bottom_center', 'bottom_right']

    STR_L = 14
    frame_string = ''
    row = 0
    for k in print_order:
        z_str = complex_to_string(frame_dict[k], N_DEC)
        PAD = ' ' * (STR_L - len(z_str))
        frame_string += k + ':' + PAD + z_str
        row += 1
        if np.mod(row,3) == 0:
            frame_string += '\n'
        else:
            frame_string += '\t'
            
    return frame_string

