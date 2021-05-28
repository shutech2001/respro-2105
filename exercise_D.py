import numpy as np

def calc_lnka(fb):
    """
    calculate pseudo-equilibrium constant parameters
    --------------------------------------------------
    Parameters
        f_b : fraction bound to plasma protein at wquilibrium (rate) 

    Returns
        ln(K_a) : psuedo-equilibrium constant parameters (calculate from f_b)
    """
    if type(fb) != float and len(fb.split('-')) > 1:
    	fb_min = float(fb.split('-')[0])
    	fb_max = float(fb.split('-')[1])
    	fb = (fb_min + fb_max)/2
    else:
    	fb = float(fb)
    # C is a constant set to 0.3
    C = 0.3
    # prevent divergence of the lnK_a
    scaled_fb = fb*0.99 + 0.005
    
    return C*np.log(scaled_fb/(1-scaled_fb))