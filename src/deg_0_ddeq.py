# %load ../../FlyingMachineFractal/pyreimpic/functions_demo_01.py
"""
collection of functions
Note: third parameter (ET) is ghosted (future - past compatibility)
"""
import numpy as np

def bugga_bear(Z, p, Z0=None, ET=None):
    """ Z = bugga_bear(Z, p)
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    EPSILON = 1e-15

    if p is None:
        return [-0.851612290000, -1.794358160000, -4.450415130000]
    if Z == 0.0+0.0j:
        return np.Inf

    try:
        for n in range(1, len(p)+1):
            Zok = np.isfinite(Z)
            if Zok and np.abs(Z) != 0:
                Z = 1 / Z - Z**( n * (Z**( p[n-1]**n )) / np.sqrt(np.pi))
            elif Zok:
                Z = 1 / EPSILON - Z**( n * (Z**( p[n-1]**n )) / np.sqrt(np.pi))
    except:
        pass
    return Z


def starfish_ish(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 5/8

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = -0.040431211565 + 0.388620268274j
        return p
    elif Z == 0.0+0.0j:
        return np.Inf
    else:
        return Z**(-np.exp(Z**p)**(np.exp(Z**p)**(-np.exp(Z**p)**(np.exp(Z**p)**(-np.exp(Z**p))))))


def starfish_ish_II(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 5/8

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = -0.051448293230 + 0.304348945637j
        return p
    elif Z == 0.0+0.0j:
        return np.Inf

    Z = Z**(-np.exp(Z**p)**(np.exp(Z**p)**(-np.exp(Z**p)**(np.exp(Z**p)**(-np.exp(Z**p)**(np.exp(Z**p)**(-np.exp(Z**p))))))))
    return Z

def Nautuliz(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 1/3
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = [2.792422080721, 1.227827869496+0.063564967216j]
        return p
    elif Z == 0.0+0.0j:
        return np.Inf

    Z = Z**(-p[0]**(-Z**(-p[1]))) - p[0]**(-Z**(p[1]))
    return Z

def nlC3(Z, p, Z0=None, ET=None):
    """
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        P_mat = [[0.161145800333, 0.370248409961, 0.119609846764],
                 [-0.112413777603, -0.350438576801, 0.377731470314],
                 [-0.107692385723, 1.539081783286, 0.584733068445],
                 [-1.077419657727, 1.079694186592, 0.567453371505],
                 [0.106699281931,-1.446300888486, 0.763588120232],
                 [0.046632473878, 0.223406467719, 1.386264134664],
                 [0.830928213916, 2.171631574827, -0.651878138495]]
        n = np.random.randint(0, len(P_mat), 1)[0]
        return P_mat[n]
    elif Z == 0.0+0.0j:
        return np.Inf

    Z = Z**(2*Z**(-p[0]**( Z**-p[1] )**( Z**-p[2] )  ))
    return Z

def decPwrAFx(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 1/8

    Args:
        Z:    a real or complex number
        p:    array real of complex number
    Returns:
        Z:    the result (complex)
    Z = 1/Z - Z^(n*Z^(P(n)^n) / sqrt(pi));
    """
    if p is None:
        p = [np.sqrt(np.pi), 1.13761386, -0.11556857]
        return p
    elif Z == 0.0+0.0j:
        return np.Inf

    for n in range(1,len(p)):
        Z = 1/Z - Z**(n * Z**(p[n]**n) / p[0])
    return Z


def dreadSkull(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 0.4

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
        p[0]

    MATLAB:
    Z = (-Z)^(-exp(Z^x)^(exp(Z^x)^(-exp(Z^x)^(exp(Z^x)^(-exp(Z^x)^(exp(Z^x)^(-exp(Z^x))))))))
    """
    if p is None:
        p = -0.295887110004
        return p
    elif Z == 0.0+0.0j:
        return np.Inf

    ZEP = np.exp(Z ** p)
    Zout = (-Z) ** (-ZEP ** (ZEP ** (-ZEP ** (ZEP ** (-ZEP ** (ZEP ** (-ZEP)))))))
    return Zout

def de_Okeeffe(Z, p, Z0=None, ET=None):
    """ Z = de_Okeeffe(Z, p)
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = [1.2759, 0 + 0.074647j, -0.77504 + 0.007449j, 1.2902 - 2.238e-18j, 0.12875]
        return p

    Z = (p[2] - p[1]*Z) / (p[4] + p[3]*Z**p[0])
    return Z

def IslaLace(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 1/4

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        M = [[0.444476893762, 0.508164683992 + 0.420921535772j],
             [1.032258783648, 0.229133372072 + 0.959497026392j],
             [1.047484735483, -0.314137674746 - 0.226707640874j]]
        n = np.random.randint(0, len(M), 1)[0]
        return M[n]
    elif Z == 0.0+0.0j:
        return np.Inf

    x = p[0]
    c = p[1]
    Z = ( Z**(-x**(Z**(-c))) + x**(-Z**(-c**Z))) * (c**(-Z**(-x**Z)) - Z**(-x**(-Z**(-c))) ) + \
        ( Z**(-x**(Z**(-c))) - x**(-Z**(-c**Z))) * (c**(-Z**(-x**Z)) + Z**(-x**(-Z**(-c))) )
    return Z

def RoyalZ(Z, p=None, Z0=None, ET=None):
    """
    par_set['zoom'] = 1/3

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = [0.340859990388, 0.269282250320, -0.255017720861]
        return p

    nc = len(p)
    for n in range(0, nc):
        if Z == 0.0 + 0.0j:
            return np.Inf
        try:
            Zn = Z**(-1*np.exp(Z*p[n]))
        except:
            return Z
            pass
        if np.isfinite(Zn):
            return Zn
        else:
            return Z


def ItchicuPpwrF(Z, p=None, Z0=None, ET=None, Zm1=0, Zm2=0):
    """
    par_set['zoom'] = 0.16

    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = [0.56890021, -0.25564542, -0.37746896, -0.29588711, -1.47513451, -0.23400405, 0.11844484]
        return p
    for n in range(0, len(p) - 1):
        try:
            Zn = Z ** (2 * Z ** (-(p[n]) ** (Z ** (-p[n + 1]))))
        except:
            return Z
            pass

        if np.isfinite(Zn):
            Z = Zn
        else:
            return Z

    return Z


def ElGato(Z, p=None, Z0=None, ET=None):
    """ Z = bugga_bear(Z, p)
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
        p[0]

    MATLAB:
    Z^(2 * Z^( -c(1)^( Z^-c(2) )^( Z^-c(3) )^( Z^-c(4) )^( Z^-c(5) )^( Z^-c(6) )^( Z^-c(7))))
    """
    if p is None:
        p = [0.083821, -0.2362, 0.46518, -0.91572, 1.6049, -2.3531, 3.2664]
        return p
    elif Z == 0.0+0.0j:
        return np.Inf

    Zout = Z ** (2 * Z ** (-(
                (((((p[0] ** Z ** -p[1]) ** (Z ** -p[2])) ** (Z ** -p[3])) ** (Z ** -p[4])) ** (Z ** -p[5])) ** (
                    Z ** -p[6]))))
    return Zout

def thunderHead(Z, p, Z0=None, ET=None):
    """ Z = thunderHead(Z, p)
    Args:
        Z:    a real or complex number
        p:    a real of complex number
    Returns:
        Z:    the result (complex)
    """
    if p is None:
        p = [3.83796971, -0.09564841+0.83234946j]
        return p

    Z = Z**(2*Z**(-2*p[0]**(-2*Z**(-2*p[1]))));
    return Z