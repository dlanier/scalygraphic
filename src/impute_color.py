"""
impute_color.py
initial commit late September 2019


Extending old code from FlyingMachineFractal/src/numcolor.py
- keep normalization routines,
- extend color mapping
- improve integer, complex pairs to HSV to RGB (to grey)

Assign color to Normalized algebraic vectors (Z0, Z, ET)
Z0  = start vector matrix (complex)
Z   = final vector matrix (complex)
ET  = number of iterations (positive integer)

        WhyFor the code:
NormalizationUnderstandingNotebook.ipynb
ColorImputationUnderstandingNotebook.ipynb


#                       Create new color map from dict:
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

testCmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

#                       get matplotlib Named colormaps:
c_map = matplotlib.cm.get_cmap(c_map_name)
"""
import numpy as np

import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.pyplot as pyplot

from PIL import TiffImagePlugin as tip
from PIL import Image

#                       Define lists of matplotlib named color maps by type
cmaps = {}
cmaps['Perceptually_Uniform_Sequential'] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                       'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                       'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmaps['Sequential (2)'] = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                           'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                           'hot', 'afmhot', 'gist_heat', 'copper']

cmaps['Diverging'] = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                      'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']

cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']

cmaps['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                          'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                          'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

def show_color_maps(n_cols=5):
    """ print the matplotlib color map names available in this module
    
    Args:   
        n_cols: number of columns to display in each row of the list
        
    """
    #               cmap_list: sorted unique list of all available matplotlib named color maps
    cmap_list = []
    for k, v in cmaps.items():
        for vv in v:
            cmap_list.append(vv)
    cmap_list = sorted(list(set(cmap_list)))

    #               display the list of available maps
    acum = []
    for m in cmap_list:
        if len(acum) < n_cols:
            acum.append(m)
        else:
            s = ''
            for a in acum:
                s += '%18s'%(a)
            print(s)
            acum = []

    if len(acum) > 0:
        s = ''
        for a in acum:
            s += '%18s'%(a)
        print(s)


"""                                                                    Begin-Normalize                              """
def raw_graphic_norm(Z0, Z, ET):
    """ Zd, Zr, ETn = raw_graphic_norm(Z0, Z, ET)
    noramize escape time algorithm output for color mapping input
    
    Args:
        Z0:     matrix of complex starting points
        Z:      matrix of final points (complex)
        ET:     Escape-Time -- number of iterations
        
    Returns:
        Zd:     normalized rotation distance (Z0, Z) matrix
        Zr:     normalized rotation difference (Z0, Z) matrix
        ETn:    normalized Escape Time matrix
        
    """
    ETn = graphic_norm(ET)
    Zv = Z - Z0
    Zd = graphic_norm(Zv)
    Zr = graphic_norm(np.arctan2(np.imag(Zv), np.real(Zv)))
    
    return Zd, Zr, ETn


def etg_norm(Z0, Z, ET):
    """ Zd, Zr, ETn = etg_norm(Z0, Z, ET); Graphically usable matrices from escape time algorithm result 
    Args:
        Z0:     matrix of complex starting points
        Z:      matrix of final points (complex)
        ET:     Escape-Time -- number of iterations
        
    Returns:
        Zd:     distance -- flattend norm (0.0, 1.0) absolute value of difference |Z - Z0|
        Zr:     flattend norm of rotation difference
        ETn:    flattend norm Escape Time matrix
    """
    ETn = mat2graphic(ET)
    Zv = Z - Z0
    Zd = mat2graphic(Zv)
    Zr = mat2graphic(np.arctan2(np.imag(Zv), np.real(Zv)))
    
    return Zd, Zr, ETn


def mat2graphic(Z):
    """ M, nClrs = mat2graphic(Z)
        Use all the transformation tricks to prepare input matrix Z
        for conversion to a viewable image.
        
    Args:
        Z:      real or complex (rows x xcols x 1) matrix
        
    Returns:
        M:      real (rows x xcols x 1) matrix (0 <= M <= 1)
    """
    M, nClrs = flat_index(np.abs(Z))
    
    return graphic_norm(M)


def graphic_norm(Z):
    """ rescale matrix z to distance (float) s.t.   
        0 <= z <= 1  (will include 0,1 if it has more than 1 value)
  
    Args:
        Z: is a real or complex two dimensional matrix
    
    Returns:
        Z: same size, real valued matrix with smallest member = 0, largest = 1
    """
    EPSILON = 1e-15
    I = np.abs(Z)
    I = I - I.min()
    
    return I / max(EPSILON, I.max())


def flat_index(float_mat):
    """ convert the input matrix to integers from 0 to number of unique values.
    
    Args:
        float_mat: two dimensional matrix.
        
    Return:
        float_mat: re-enumerated so that the matrix values are all sequential ints.
        n_colors:  number of unique values in the input / output matrix
    """
    rows = float_mat.shape[0]
    cols = float_mat.shape[1]

    float_mat = np.reshape(float_mat, (1, float_mat.size))
    ixA = np.argsort(float_mat)[0]
    
    current_value = float_mat[0, ixA[0]]
    enumeration_value = 0
    for ix in ixA:
        if float_mat[0,ix] != current_value:
            current_value = float_mat[0,ix]
            enumeration_value += 1
        float_mat[0,ix] = enumeration_value
    
    float_mat = np.array(np.reshape(float_mat, (rows, cols)))
    float_mat = np.int_(float_mat)
    n_colors = enumeration_value + 1
    
    return float_mat, n_colors


def range_norm(Z, lo=0.0, hi=1.0):
    """ normaize input matrix Z within a lo - hi range

    Args:
        Z:        real or complex matrix
        lo:       smallest value in output range
        hi:       largest value in output range

    Returns:
        I:        matrix of size Z, normalized to the range of (lo, hi)

    """
    I = graphic_norm(Z)
    hi = max(min(hi, 1.0), 0.0)
    lo = min(max(lo, 0.0), 1.0)
    low_fence = min(hi, lo)
    hi_fence = max(hi, lo)

    if low_fence == hi_fence:
        return I

    v_span = hi_fence - low_fence
    I = I * v_span + low_fence

    return I

"""
        End Normalize
                                                        Begin Image Color mapping wrapping
"""

def get_grey_thumb(imfile_name, thumb_size=(128, 128)):
    """ im = get_grey_thumb(imfile_name):
    """
    c_map = mpl.cm.get_cmap('Greys')
    img_src = tip.Image.open(imfile_name).convert('L')
    img_src.thumbnail(thumb_size)
    im = np.array(img_src)
    im = c_map(im)
    im = np.uint8(im * 255)
    im = tip.Image.fromarray(im)
    
    return im


def primitive_2_gray(P):
    """
    Args:
         P:     Single layer matrix ET or abs(Z - Z0) etc.

    Returns:
        I:      grayscale image

    """
    n_rows = P.shape[0]
    n_cols = P.shape[1]
    
    A = np.zeros((n_rows, n_cols,3))
    A[:,:,0] += P     # Hue
    A[:,:,1] += P      # Saturation
    A[:,:,2] += P      # Value
    
    I = tip.Image.fromarray(np.uint8(A * 255), 'RGB').convert('L')
    
    return I


def map_raw_etg(Z0, Z, ET, c_map_name='afmhot'):
    """ get a color-mapped image of normalized distance

    Args:
        ET:     (Integer) matrix of the Escape Times
        Z:      (complex) matrix of the final vectors
        Z0:     (complex) matrix of the starting plane

    Returns:
        I:      RGB PIL image

    """
    Zd, Zr, ETn = etg_norm(Z0, Z, ET)
    
    c_map = mpl.cm.get_cmap(c_map_name)
    im = c_map(Zd)
    im = np.uint8(im * 255)
    im = tip.Image.fromarray(im)
    
    return im


def map_etg_composite(Z0, Z, ET, c_map_name='afmhot'):
    """ get an RGB image of HSV composite index to color map

    Args:
        ET:     (Integer) matrix of the Escape Times
        Z:      (complex) matrix of the final vectors
        Z0:     (complex) matrix of the starting plane

    Returns:
        I:      RGB PIL image
    """
    im = np.array(get_im(ET, Z, Z0).convert('L'))

    c_map = mpl.cm.get_cmap(c_map_name)
    im = c_map(im)
    
    im = np.uint8(im * 255)
    im = tip.Image.fromarray(im)
    
    return im


def im_file_map(imfile_name, cmap_name='hot', thumb_size=None):
    """ open an image file and color map it

    Args:
        imfile_name:    RGB or Greyscale image
        cmap_name:      name of a matplotlib color map
        (thumb_size):   thumnail image size eg (128, 128)

    Returns:
        I:              tif image
    """
    if isinstance(cmap_name, LinearSegmentedColormap):
        cm_hot = cmap_name
    else:
        cm_hot = mpl.cm.get_cmap(cmap_name)

    img_src = tip.Image.open(imfile_name).convert('L')

    if not thumb_size is None:
        img_src.thumbnail((thumb_size[0], thumb_size[1]))

    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = tip.Image.fromarray(im)

    return im


def get_im(ET, Z, Z0):
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
    n_rows = np.shape(ET)[0]
    n_cols = np.shape(ET)[1]
    Zd, Zr, ETn = etg_norm(Z0, Z, ET)

    A = np.zeros((n_rows, n_cols, 3))
    A[:,:,0] += ETn     # Hue
    A[:,:,1] += Zr      # Saturation
    A[:,:,2] += Zd      # Value
    I = tip.Image.fromarray(np.uint8(A * 255), 'HSV').convert('RGB')
    
    return I


def get_gray_im(ET, Z, Z0):
    """ get a gray-scale image from the products of the escape-time-algorithm
    
    Args:
        ET:     (Integer) matrix of the Escape Times
        Z:      (complex) matrix of the final vectors
        Z0:     (complex) matrix of the starting plane
        
    Returns:
        I:      grayscale PIL image
    """
    return get_im(Z0, Z, ET).convert('L')


def pil_rgb_2_uint16_gray(rgb_im):
    """ convert a pillow rgb image to an unsigned 16 bit numpy array
            - suitable for skimage.io.imsave
            - suitable for matplotlib color mapping

    Args:
        rgb_im:     PIL rgb image

    Returns:
        gray_int:   numpy.ndarray of uint16 (0, 65535)
    """
    rgb_im.getdata()
    r, g, b = rgb_im.split()

    ra = np.array(r)
    ga = np.array(g)
    ba = np.array(b)

    bit_depth = 2 ** 16 - 1

    R2g = 0.299
    G2g = 0.587
    B2g = 0.114

    gray_int = np.uint16((R2g * ra + G2g * ga + B2g * ba) * bit_depth)

    return gray_int


def get_uint16_gray(ET, Z, Z0):
    """  convert results data (ET, Z, Z0) to an unsigned 16 bit numpy array
            - Write with: skimage.io.imsave(f_name, gray_int)
            - colo map with matplotlib.cm
    Args:
        ET:     (Integer) matrix of the Escape Times
        Z:      (complex) matrix of the final vectors
        Z0:     (complex) matrix of the starting plane

    Returns:
        gray_16bit:     numpy.ndarray (16 bit integer)
    """
    img = get_im(ET, Z, Z0)
    gray_int = pil_rgb_2_uint16_gray(img)

    return gray_int


def cat_im_list_hori(im_list):
    """ combine a list of images horizontaly
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
    """ combine a list of images vertically
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

"""
#            to do:     Sort out the ugly details s.t. a natural color specification will produce
#                       images with photo-like histograms
                        
nat_spec_struct = { 'hsv_assingment': { 'hue': ['ET'], 
                                        'saturation': ['ET', 'Zd', 'Zr'], 
                                        'value': ['Zd']},
                    'hue': [v_low, v_high],
                    'saturation': [v_low, v_high], 
                    'value': [v_low, v_high], 
                    'c_map': [nx3 array] }
                    
def imp_natcho_color(ET, Z, Z0, nat_spec_struct):
    
    #                   Get the Hue channel by mapping the hue choice 
    #                   im = c_map(ET_normalized)                     ?? call mpl.cm.get_cmap direct ??
    G = np.array( primitive_2_gray(ET) )
    N = np.maximum(ET) + 1
    c_dict = nat_spec_struct['c_map']
    c_map_lcl = LinearSegmentedColormap('c_map_lcl', segmentdata=c_dict, N)
    # c_map = mpl.cm.get_cmap(c_map_name)
    im = c_map(G)
    im = np.uint8(im * 255)
    im = tip.Image.fromarray(im).convert('HSV')
    #        , ...
    
"""

"""
            End Image Color mapping wrapping
                                                                Begin Complex Matrix <<-->> Image
"""

def im_diff(im_gray_array, n=1):
    """ get the pixels difference as complex vectors
    Usage: Z = im_diff(im_gray_array, n=1)
    Args:
        im_gray_array:  gray scale image as a numpy array
        n_diff:         level of differences small integer 1 or 2 probably

    Returns:
        Z:              matrix of complex vectors == dir & magnitude of differences

    """
    D_l_r = (im_gray_array[:, n:] - im_gray_array[:, :-n])[n:, :]
    D_u_d = (im_gray_array[n:, :] - im_gray_array[:-n, :])[:, n:]

    Z = D_l_r.astype(np.float) + D_u_d.astype(np.float) * 1j

    return Z


def im_diag_diff(im_gray_array, n=1):
    """ get the pixel diagonal difference as complex vectors
    Usage: Z = im_diag_diff(im_gray_array, n)
    Args:
        im_gray_array:  gray scale image as a numpy array (uint8)
        n_diff:         level of differences small integer 1 or 2 probably

    Returns:
        Z:              matrix of complex vectors == dir & magnitude of differences

    """
    D_ul_lr = (im_gray_array[:-n, :-n] - im_gray_array[n:, n:])
    D_ur_ll = (im_gray_array[n:, :-n] - im_gray_array[:-n, n:])

    Z = D_ul_lr.astype(np.float) * (1 - 1j) + D_ur_ll.astype(np.float) * (-1 - 1j)

    return Z


def im_to_Z(im_gray_array, n=1):
    """ convert grayscale image into complex vectors of edges using both im_diff and im_diag_diff
    Usage: Z = im_to_Z(im_gray_array, n)

    Args:
        im_gray_array:  gray scale image as a numpy array
        n_diff:         level of differences small integer 1 or 2 probably

    Returns:
        Z:              matrix of complex vectors == max of the magnitudes

    """
    Z_ax = im_diff(im_gray_array, n)
    Z_dax = im_diag_diff(im_gray_array, n)

    Z = Z_ax + Z_dax

    return Z


def complex_mat_to_im(Z):
    """ image from complex matrix: normalized enumeration of the magnitude of Z as an image
        (it works - try not to think about it)

    Args:
        Z:      matrix (numpy array) of complex numbers

    Returns:
        im:     PIL grayscale image

    """
    Zd = np.abs(Z)
    Zr = np.arctan2(np.real(Z), np.imag(Z))

    im = Image.fromarray((mat2graphic(np.maximum(graphic_norm(Zr), graphic_norm(Zd))) * 255).astype(np.uint8))
    return im


def complex_magnitude_image(Z):
    """ normalized enumeration of the magnitude of Z as an image

    Args:
        Z:      matrix (numpy array) of complex numbers

    Returns:
        im:     PIL grayscale image

    """
    Zd = np.abs(Z)
    return Image.fromarray((mat2graphic(Zd) * 255).astype(np.uint8))


def complex_rotation_image(Z):
    """ normalized enumeration of the rotation of Z as an image

    Args:
        Z:      matrix (numpy array) of complex numbers

    Returns:
        im:     PIL grayscale image

    """
    Zr = np.arctan2(np.real(Z), np.imag(Z))
    return Image.fromarray((mat2graphic(Zr) * 255).astype(np.uint8))

"""
            End Complex Matrix <<-->> Image
            
                                                    Begin Image Statistics
"""
