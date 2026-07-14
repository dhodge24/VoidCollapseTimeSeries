"""

Code was converted and modified from MATLAB to python from Reference 1) below

References:
    1) "An Augmented Lagrangian Method for Total Variation Video Restoration" by S. Chan et al.

The purpose of the code is to deconvolve our experimental images given an estimated point spread function (PSF).
The PSF can be estimated by least squares error between a simulated and experimental image. Applying the deconvolution
helps us achieve what our true object image is and gives us more accurate phase reconstructions.

"""

import numpy as np
from tifffile import imread, imwrite

from SSPR.units import *
from SSPR.utilities import cropToCenter, showImg, FFT, IFFT, compare_plots_lineout, voigt_2d, fadeoutImage, removeOutliers
import matplotlib.pyplot as plt


class DeconvTV:
    """

    References:
        (1): "An Augmented Lagrangian Method for Total Variation Video Restoration" by S. Chan et al.

    Code was converted from MATLAB to python from here:
    https://www.mathworks.com/matlabcentral/fileexchange/43600-deconvtv-fast-algorithm-for-total-variation-deconvolution

    Deconvolves image g by solving a TV minimization problem.
    min  mu  || Hf - g ||_1 + ||f||_TV
    min mu/2 || Hf - g ||^2 + ||f||_TV

    where ||f||_TV = sum_{x,y,t} sqrt(a||Dxf||^2 + b||Dyf||^2 + c||Dtf||^2),
    Dxf = f(x+1, y, t) - f(x,y,t)
    Dyf = f(x, y+1, t) - f(x,y,t)
    Dtf = f(x, y, t+1) - f(x,y,t)
    """

    def __init__(self, g, H, mu, method='L2', rho_r=2, rho_o=50, beta=None, gamma=2, max_iter=20, alpha=0.7, tol=1e-3,
                 verbose=True):
        """
        Initializes the DeconvTV class with the given parameters.

        Parameters:
        - g: Observed image (grayscale).
        - H: Point spread function.
        - mu: Regularization parameter.
        - method: Type of deconvolution algorithm (L1 or L2) -- default is L2
        - rho_r: Initial penalty parameter for ||u-Df|| -- default is 2
        - rho_o: Initial penalty parameter for ||Hf-g-r|| -- default is 50
        - beta: Regularization parameter [a b c] for weighted TV norm
        - gamma: Update constant for rho_r -- default is 2
        - max_iter: Maximum number of iterations
        - alpha: Constant that determines constraint violation -- default 0.7
        - tol: Tolerance level on relative change -- default is 1e-3
        - verbose: Explicit verbosity flag -- default is True

        :returns: Deconvolved 2D grayscale image
        """

        self.g = g
        self.H = H
        self.rows = self.g.shape[0]
        self.cols = self.g.shape[1]

        self.mu = mu
        if not isinstance(self.mu, (int, float)):
            raise ValueError('mu must be a numeric value!')

        self.method = method
        self.rho_r = rho_r
        self.rho_o = rho_o
        if beta is None:
            self.beta = [1, 1]  # For single 2D image
        else:
            self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.f = self.g
        self.yx = np.zeros_like(self.g)
        self.yy = np.zeros_like(self.g)
        self.ux = np.zeros_like(self.g)
        self.uy = np.zeros_like(self.g)
        self.z = np.zeros_like(self.g)  # For L1 regularization
        self.verbose = verbose

    @staticmethod
    def D(x, beta):
        """Computes the first-order forward finite-difference derivative of a 2D image array along both the
        horizontal (x) and vertical (y) directions, as well as across sequential frames (z-direction), if any.
        It employs periodic boundary conditions to handle the edges of the image, ensuring smooth transitions
        across boundaries.

        :param: x: 2D array [N, M]
        :param: beta: List of regularization values for x, y, and z

        :returns: vector
        """
        Dx = beta[0] * np.concatenate((np.diff(x, axis=1), x[:, :1] - x[:, -1:]), axis=1)
        Dy = beta[1] * np.concatenate((np.diff(x, axis=0), x[:1, :] - x[-1:, :]), axis=0)
        return Dx, Dy

    @staticmethod
    def DT(X, Y, beta):
        """The transpose of the gradient is also known as the divergence:
        https://math.stackexchange.com/questions/44945/divergence-as-transpose-of-gradient

        The transpose of the gradient operates as the backward finite difference derivative or, in other words, the
        forward difference transposes to backward difference multiplied by -1:
        https://math.mit.edu/~gs/linearalgebra/ila5/TransposeDerivative01.pdf

        Also helpful -- the transposed derivatives are equal to the derivative multiplied by -1. For example, if D uses
        forward differences, then the transposed derivative uses backward differences:
        https://math.stackexchange.com/questions/1518887/gradient-transpose-along-x-or-y-direction-in-images

        :param: X: Vector component in x
        :param: Y: Vector component in y
        :param: Z: Vector component in z
        :param: beta: List of regularization values for x, y, and z

        :returns: scalar
        """
        DT_XY = np.concatenate((X[:, -1:] - X[:, :1], -np.diff(X, axis=1)), axis=1)
        DT_XY = beta[0] * DT_XY + beta[1] * np.concatenate((Y[-1:, :] - Y[:1, :], -np.diff(Y, axis=0)), axis=0)
        return DT_XY

    def deconvolve(self):
        """
        Performs the deconvolution based on the initialized parameters.

        Returns:
        A dictionary with the deconvolution results.
        """
        rows, cols = self.g.shape
        memory_condition = np.finfo(np.float32).max  # Memory condition check
        max_array_memory = memory_condition / 16
        if rows * cols > 0.1 * max_array_memory:
            print('Warning: possible memory issue')
            reply = input('Do you want to continue? [y/n]: ')
            if reply.lower() == 'n':
                return {'f': 0}

        if self.method == 'L2':
            return self.deconv_TV_L2()
        elif self.method == 'L1':
            return self.deconv_TV_L1()
        else:
            raise ValueError('Unknown method. Please choose "L1" or "L2" as the method.')

    def deconv_TV_L2(self):
        """
        Deconvolution using the L2 method.
        """

        # H conjugate transpose times H -- See Eq. 14/15 in Reference (1)
        HT_H = np.abs(np.fft.fftn(self.H, s=(self.rows, self.cols))) ** 2

        # Prepare the discrete derivative operator for x and y directions, ensuring correct 2D shape
        pad_vector_x = np.array([1, -1]).reshape((1, -1))  # Discrete derivative in x
        pad_vector_y = np.array([1, -1]).reshape((-1, 1))  # Discrete derivative in y

        DT_Dx = np.abs(self.beta[0] * np.fft.fftn(pad_vector_x, s=(self.rows, self.cols))) ** 2
        DT_Dy = np.abs(self.beta[1] * np.fft.fftn(pad_vector_y, s=(self.rows, self.cols))) ** 2
        DT_D = DT_Dx + DT_Dy

        # Circular convolution to calculate Htg
        # HT_g = convolve2d(self.g, self.H, mode='same', boundary='wrap')  # Super slow with large arrays, use FFT
        HT_g = np.real(IFFT(FFT(self.g) * FFT(self.H)))  # Faster convolution method, convolution theorem

        Dfx, Dfy = DeconvTV.D(x=self.f, beta=self.beta)
        rel_change_vals = []
        obj_vals = []

        if self.verbose:
            print('Running deconvTV (L2 version)')
            print(f'mu = {self.mu:10.2f}\n')
            print('itr      rel_change      ||Hf-g||^2       ||f||_TV        Obj Val           rho')

        rnorm = np.sqrt(np.linalg.norm(Dfx.flatten()) ** 2 + np.linalg.norm(Dfy.flatten()) ** 2)

        for i in range(self.max_iter):
            # Save the current estimate of f to compare later for relative change
            f_old = self.f.copy()

            # Solve the f-subproblem
            numerator = np.fft.fftn((self.mu / self.rho_r) * HT_g +
                                    DeconvTV.DT(self.ux - (1 / self.rho_r) * self.yx, self.uy -
                                                (1 / self.rho_r) * self.yy, beta=self.beta))
            denominator = (self.mu / self.rho_r) * HT_H + DT_D
            self.f = np.real(np.fft.ifftn(numerator / denominator))
            self.f[self.f < 0] = 0

            # Solve the u-subproblem
            Dfx, Dfy = DeconvTV.D(self.f, beta=self.beta)
            vx = Dfx + (1 / self.rho_r) * self.yx
            vy = Dfy + (1 / self.rho_r) * self.yy
            v = np.sqrt(vx ** 2 + vy ** 2)
            v[v == 0] = 1  # Prevent division by zero
            v = np.maximum(v - 1 / self.rho_r, 0) / v
            self.ux = vx * v
            self.uy = vy * v

            # Update y
            self.yx = self.yx - self.rho_r * (self.ux - Dfx)
            self.yy = self.yy - self.rho_r * (self.uy - Dfy)

            # Optionally update rho and compute objective value for printing
            if self.verbose:
                # r1 = convolve2d(self.f, self.H, mode='same', boundary='wrap') - self.g  # Super slow with large arrays
                r1 = np.real(IFFT(FFT(self.f) * FFT(self.H))) - self.g  # Faster convolution method, convolution theorem
                r1_norm = np.mean(r1.flatten() ** 2)
                r2_norm = np.mean(np.sqrt(Dfx.flatten() ** 2 + Dfy.flatten() ** 2))
                obj_val = (self.mu / 2) * r1_norm + r2_norm

            # Compute the Frobenius norm for the residual of Df and u
            rnorm_old = rnorm
            rnorm = np.sqrt(np.linalg.norm(Dfx - self.ux, 'fro') ** 2 + np.linalg.norm(Dfy - self.uy, 'fro') ** 2)

            # Adjust rho if necessary
            if rnorm > self.alpha * rnorm_old:
                self.rho_r *= self.gamma

            # Calculate and store the relative change of f
            rel_change = np.linalg.norm(self.f.flatten() - f_old.flatten()) / np.linalg.norm(f_old.flatten())
            rel_change_vals.append(rel_change)

            # Optionally print iteration details
            if self.verbose:
                obj_vals.append(obj_val)
                print(f'{i + 1:3d} \t {rel_change:6.4e} \t {r1_norm:6.4e} \t {r2_norm:6.4e} '
                      f'\t {obj_val:6.4e} \t {self.rho_r:6.4e}')

            # Check for convergence
            if rel_change < self.tol:
                break

        return self.f

    def deconv_TV_L1(self):
        """
        Deconvolution using the L1 method. Still need to convert this to python at a later time.
        """
        pass


save = True
plot_psf = False
extend_image = False
plot_comparison = False

run_wfs = "561"
run_holo = "590"

# Main directories
dir_main = "/Users/danielhodge/Desktop/"
dir_holo_preprocessed = "run" + run_holo + "_exp_preprocessed/"

# Files to import
tiff_holo_with_speckle_ffc_extended = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended.tiff"

# Files to save
tiff_holo_with_speckle_ffc_extended_decon = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended_decon.tiff"


Nx = 2500  # Pixels in x
Ny = 2500  # Pixels in y
N_pad = 2500
z01 = 120.41 * mm  # Distance from source to sample
z12 = 4.668995 * m  # Distance from sample to detector
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01
scale_fac = 4
detPixSize = 6.5 * um  # Detector pixel size
dx_eff = detPixSize / M / scale_fac  # Object pixel size in x, equals detector pixel size if no mag
dy_eff = detPixSize / M / scale_fac  # Object pixel size in y, equals detector pixel size if no mag
X = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=2 * np.pi / (Nx * dx_eff)))  # Spacing in real space
Y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=2 * np.pi / (Ny * dy_eff)))  # Spacing in real space
x, y = np.meshgrid(X, Y)
# Generate the PSF given the estimated parameters for the 2D Voigt profile -- Blur to incorporate effects of
# scintillator, finite source size, and partial degree of transverse coherence of the x-ray beam
sigma_g = 415.444e-9  # Gaussian sigma
gamma_l = 0e-9  # Lorentzian gamma
H = voigt_2d(x, y, sigma_g, gamma_l)

# Import image to be de-convolved (deblurred)
g = np.array(imread(dir_main + dir_holo_preprocessed + tiff_holo_with_speckle_ffc_extended))
g = cropToCenter(img=g, newSize=[2500, 2500])
g[g < 0] = 0
g = removeOutliers(originalImage=g)

H = cropToCenter(img=H, newSize=[2500, 2500])
H /= np.sum(np.sum(H))  # Normalize after crop operation

if plot_psf:
    showImg(H)

g[g < 0] = 0
# Higher mu = more grainy, higher beta = less grainy
# deconv = DeconvTV(g, H, mu=500, beta=[0.8, 0.8], rho_r=2000, gamma=2, max_iter=100, tol=1e-2, verbose=True)
deconv = DeconvTV(g, H, mu=100, beta=[1.0, 1.0], rho_r=2000, gamma=2, max_iter=100, tol=1e-2, verbose=True)
result = deconv.deconvolve()
result[result < 0] = 0

if extend_image:
    ellipse_size_y = 0.8
    ellipse_size_x = 0.8
    transition_length_y = 20
    transition_length_x = 20
    fade_to_val = None
    num_segments = 250
    result, _ = fadeoutImage(img=result,
                             fadeMethod='ellipse',
                             ellipseSize=[ellipse_size_y, ellipse_size_x],
                             transitionLength=[transition_length_y, transition_length_x],
                             fadeToVal=fade_to_val,
                             numSegments=num_segments,
                             bottomApply=False)
showImg(result)

# Before and after
if plot_comparison:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
    axes[0].imshow(g, cmap='Greys_r')
    axes[0].set_title('Blurred Intensity Image')
    axes[0].axis('off')  # Hide axes ticks
    axes[1].imshow(result, cmap='Greys_r')
    axes[1].set_title('De-convolved Intensity Image')
    axes[1].axis('off')  # Hide axes ticks
    plt.tight_layout()
    plt.show()

imwrite(dir_main + dir_holo_preprocessed + tiff_holo_with_speckle_ffc_extended_decon, result)
