import numpy as np

from calibration import compute_kx_ky, estimate_f_b
from extract_patches import extract_patches


def triangulate(u_left, u_right, v, calib_dict):
    """
    Triangulate (determine 3D world coordinates of) a set of points given their projected coordinates in two images.
    These equations are according to the simple setup, where C' = (b, 0, 0)

    Parameters
    ----------
        u_left: NumPy array of shape (num_points,)
            Projected u-coordinates of the 3D-points in the left image
        u_right: NumPy array of shape (num_points,)
            Projected u-coordinates of the 3D-points in the right image
        v: NumPy array of shape (num_points,)
            Projected v-coordinates of the 3D-points (same for both images)
        calib_dict: dict
            Dict containing complete set of camera parameters.
            (Expected to contain kx, ky, f, b)
    
    Returns
    -------
        NumPy array of shape (num_points, 3):
            Triangulated 3D coordinates of the input - in units of [mm]
    """
    #
    # TO IMPLEMENT
    #
    return -1


def compute_ncc(img_l, img_r, p):
    """
    Calculate normalized cross-correlation (NCC) between patches at the same row in two images.
    
    The regions near the boundary of the image, where the patches go out of image, are ignored.
    That is, for an input image, "p" number of rows and columns will be ignored on each side.

    For input images of size (H, W, C), the output will be an array of size (H - 2*p, W - 2*p, W - 2*p)

    Parameters
    ----------
        img_l: NumPy array of shape (H, W, C)
            Left image
        img_r: NumPy array of shape (H, W, C)
            Right image
        p: int
            Defines square neighborhood. Patch-size is (2*p+1, 2*p+1)
                              
    Returns
    -------
        NumPy array of shape (H - 2*p, W - 2*p, W - 2*p):
        The value output[r, c_l, c_r] denotes the NCC between the patch centered at (r + p, c_l + p) 
        in the left image and the patch centered at  (r + p, c_r + p) at the right image.
    """

    assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
    assert img_l.shape == img_r.shape, "Shape mismatch."
    
    H, W, C = img_l.shape

    # Extract patches - patches_l/r are NumPy arrays of shape H, W, C * (2*p+1)**2
    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)
    
    # Standardize each patch
    #
    # TO IMPLEMENT
    #
    
    # Compute correlation (using matrix multiplication) - corr will be of shape H, W, W
    #
    # TO IMPLEMENT
    #
    corr = np.zeros(H, W, W)
    
    # Ignore boundaries
    return corr[p:H-p, p:W-p, p:W-p]


class Stereo3dReconstructor:
    def __init__(self, p=5, w_mode='none'):
        """
        Feel free to add hyper parameters here, but be sure to set defaults
        
        Args:
            p       ... Patch size for NCC computation
            w_mode  ... Weighting mode. I.e. method to compute certainty scores
        """
        self.p = p
        self.w_mode = w_mode

    def fill_calib_dict(self, calib_dict, calib_points):
        """ Fill missing entries in calib dict - nothing to do here """
        calib_dict['kx'], calib_dict['ky'] = compute_kx_ky(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        
        return calib_dict

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        """
        Compute point correspondences for two images and perform 3D reconstruction.

        Parameters
        ----------
            img_l: NumPy array of shape (H, W, C)
                Left image
            img_r: NumPy array of shape (H, W, C)
                Right image
            calib_dict: dict
                Dict containing complete set of camera parameters.
                (Expected to contain kx, ky, f, b)
        
        Returns
        -------
            NumPy array of shape (H, W, 4):
                Array containing the re-constructed 3D world coordinates for each pixel in the left image.
                Boundary points - which are not well defined for NCC - may be padded with 0s.
                4th dimension holds the certainties.
        """

        assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."
        
        H, W, C = img_l.shape
        
        #
        # TO IMPLEMENT
        #
        # Use the functions compute_ncc() and triangulate() here
        return np.zeros(H, W, 4)