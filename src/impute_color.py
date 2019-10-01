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
from matplotlib.colors import LinearSegmentedColormap

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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PIL import TiffImagePlugin as tip

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
