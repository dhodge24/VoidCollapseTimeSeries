"""

The purpose of this code is to estimate the point spread function (PSF) by performing a minimization procedure between
an experimentally flat-field corrected (FFC) image and a synthetic (noiseless and without blur and speckle) version.
We can use this PSF value to deconvolve the experimental FFC image and then perform phase retrieval. The PSF value
determined here will also be used for blurring the simulated white field and static and dynamic void images.

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from mechanize import Browser
from scipy.interpolate import interp1d
import joblib
from tqdm import tqdm
from tifffile import imwrite

# PyTorch libraries
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# Import custom modules
from SSPR.Propagation.SimulateHologram_PyTorch import simulate_hologram
from SSPR.units import *
from SSPR.utilities_PyTorch import FFT, IFFT, create_circular_mask, compare_plots_lineout, fadeoutImage


def load_pca_model(filename):
    return joblib.load(filename)


def resize_with_anti_aliasing(phase, output_shape, sigma=(5, 5)):
    # Assuming phase is a 2D tensor of shape (H, W)
    # Add batch and channel dimensions
    phase = phase.unsqueeze(0).unsqueeze(0)

    # Resize using bilinear interpolation (order=3 is cubic, so bilinear is a close approximation)
    phase_resized = F.interpolate(phase, size=output_shape, mode='bilinear', align_corners=False)

    # Apply Gaussian blur for anti-aliasing
    blurred_phase = gaussian_blur(phase_resized, kernel_size=[2 * int(s) + 1 for s in sigma], sigma=sigma)

    # Remove batch and channel dimensions
    blurred_phase = blurred_phase.squeeze(0).squeeze(0)

    return blurred_phase


def gaussian_profile(x, y, sigma):
    """2D Gaussian function"""
    return torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (sigma * torch.sqrt(torch.tensor(2) * torch.pi))


def lorentzian_profile(x, y, gamma):
    """2D Lorentzian function"""
    return gamma / (torch.pi * (x ** 2 + y ** 2 + gamma ** 2))


def voigt_2d(x, y, sigma_g, gamma_l, device):
    """Generate a 2D Voigt profile by numerically convolving a Gaussian and Lorentzian."""
    gaussian_grid = gaussian_profile(x, y, sigma_g).to(device)
    lorentzian_grid = lorentzian_profile(x, y, gamma_l).to(device)

    # Convolve Gaussian and Lorentzian profiles
    voigt_grid = torch.real(IFFT(FFT(gaussian_grid) * FFT(lorentzian_grid)))
    voigt_grid /= torch.sum(torch.sum(voigt_grid))  # Normalize the output

    return voigt_grid


def func(sig, gam, simulated_hologram, experimental_hologram, device='cpu'):
    psf = voigt_2d(x, y, sig, gam, device=device)
    #psf = gaussian_profile(x, y, sigma=sig)
    psf /= torch.sum(torch.sum(psf))
    convolved_hologram = torch.real(IFFT(FFT(simulated_hologram) * FFT(psf)))
    convolved_hologram = convolved_hologram * mask
    experimental_hologram = experimental_hologram * mask
    loss = torch.sum(abs(convolved_hologram - experimental_hologram) ** 2)  # MSE loss
    print(loss)
    return loss, convolved_hologram


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_img = False
plot_index_of_refrac = False
plot = False  # To check how it is updating. Need to update this to have low frequency of plots

run_wfs = "562"
run_holo = "577"

# Directories
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_sim = "run" + run_holo + "_sim/"
dir_exp = "run" + run_holo + "_exp_preprocessed/"
dir_Gaussian_beams = "run" + run_wfs + "_Gaussian_beams/"
dir_wfs_no_speckle = "run" + run_wfs + "_holos_no_speckle/"

# Experimental files to import
tiff_ffc_exp = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended.tiff"

# Import experimental hologram
holo_exp = torch.tensor(imread(dir_main + dir_exp + tiff_ffc_exp), dtype=torch.float32)

holo_exp = holo_exp.unsqueeze(0).unsqueeze(0)
holo_exp = gaussian_blur(holo_exp, kernel_size=[2 * int(s) + 1 for s in (3, 3)], sigma=3)
holo_exp = holo_exp.squeeze(0).squeeze(0)

holo_exp, _ = fadeoutImage(img=holo_exp,
                           fadeMethod='ellipse',
                           fadeToVal=1.0,
                           transitionLength=[20, 20],
                           ellipseSize=[0.85, 0.85])

# Define experimental parameters
Ny, Nx = holo_exp.shape
N_pad = 6000
E = torch.tensor(18000.0, dtype=torch.float64)  # Energy of the beam
z01 = torch.tensor(120.41 * mm, dtype=torch.float64)  # Distance from source to sample
z12 = torch.tensor(4.668995 * m, dtype=torch.float64)  # Distance from sample to detector
scale_fac = torch.tensor(4.0, dtype=torch.float64)  # Scale factor we include to compensate for lens mag
det_pix_size = torch.tensor(6.5 * um, dtype=torch.float64)  # Detector pixel size
lam = (1240.0 / E) * nm  # Wavelength of the beam
z02 = z01 + z12  # Distance from source to detector
M = z02 / z01  # Magnification
dx_eff = det_pix_size / M / scale_fac  # Object effective pixel size in x, equals detector pixel size if no mag
dy_eff = det_pix_size / M / scale_fac  # Object effective pixel size in y, equals detector pixel size if no mag
X = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(Nx, d=2 * np.pi / (Nx * dx_eff), dtype=torch.float32))
Y = 2 * torch.pi * torch.fft.fftshift(torch.fft.fftfreq(Ny, d=2 * np.pi / (Ny * dy_eff), dtype=torch.float32))
y, x = torch.meshgrid(X, Y, indexing='ij')
z_eff = z12 / M  # Propagation distance
extent_x = Nx * dx_eff
extent_y = Ny * dy_eff
k0 = 2 * torch.pi / lam

# Define SU8 and SiO2 parameters (guesses)
d_void_out = torch.tensor(42 * um, dtype=torch.float32)
t_shell = torch.tensor(3.0 * um, dtype=torch.float32)
SU8_thickness = torch.tensor(450 * um, dtype=torch.float32)

# Create mask
mask = create_circular_mask(size=holo_exp.shape[0], percentage=0.85, smooth_pixels=20)

# Find delta and beta parameters for SiO2 -- Form here: https://henke.lbl.gov/optical_constants/getdb2.html
# SiO2 has density 2.2 g/cc
br1 = Browser()  # Open browser
br1.open("https://henke.lbl.gov/optical_constants/getdb2.html")  # Open to specified URL
br1.form = list(br1.forms())[0]  # If the form has no name
br1.form['Formula'] = 'SiO2'  # Choose element or combination of
br1.form['Density'] = str(2.2)  # SiO2 density is 2.2 g/cc
br1.form['Scan'] = ["Energy"]  # Specify energy
br1.form['Min'] = str(30)  # Minimum x-ray energy to consider -- 30 eV is the min energy allowed
br1.form['Max'] = str(30000)  # Max x-ray energy to consider -- 30000 eV is the max energy allowed
br1.form['Npts'] = str(500)  # More points for more accuracy in interpolation -- 500 is maximum
br1.form['Output'] = ["Text File"]  # Chooses text file so I can extract data
req1 = br1.submit()  # Submit the form
data_SiO2 = req1.get_data()  # Obtain the data
data_SiO2 = data_SiO2.splitlines(True)  # Splits string into a list
data_SiO2 = data_SiO2[2:]  # Remove title headers and extract only numerical values
E_vals_SiO2, delta_vals_SiO2, beta_vals_SiO2 = np.loadtxt(data_SiO2, unpack=True)  # Obtain the delta and beta values

# Find delta and beta parameters for SU8 -- Form here: https://henke.lbl.gov/optical_constants/getdb2.html
# SU8 is C87 H118 O16 with density 1.2 g/cc
br2 = Browser()  # Open browser
br2.open("https://henke.lbl.gov/optical_constants/getdb2.html")  # Open to specified URL
br2.form = list(br2.forms())[0]  # If the form has no name
br2.form['Formula'] = 'C87H118O16'  # Choose element or combination of
br2.form['Density'] = str(1.2)  # SU8 density is 1.2 g/cc
br2.form['Scan'] = ["Energy"]  # Specify energy
br2.form['Min'] = str(30)  # Minimum x-ray energy to consider -- 30 eV is the min energy allowed
br2.form['Max'] = str(30000)  # Max x-ray energy to consider -- 30000 eV is the max energy allowed
br2.form['Npts'] = str(500)  # More points for more accuracy in interpolation -- 500 is maximum
br2.form['Output'] = ["Text File"]  # Chooses text file so I can extract data
req2 = br2.submit()  # Submit the form
data_SU8 = req2.get_data()  # Obtain the data
data_SU8 = data_SU8.splitlines(True)  # Splits string into a list
data_SU8 = data_SU8[2:]  # Remove title headers and extract only numerical values
E_vals_SU8, delta_vals_SU8, beta_vals_SU8 = np.loadtxt(data_SU8, unpack=True)  # Obtain the delta and beta values

if plot_index_of_refrac:
    # Plot the refractive index for SiO2
    plt.figure()
    plt.loglog(E_vals_SiO2, delta_vals_SiO2, color='blue', label='Real part (delta)')
    plt.loglog(E_vals_SiO2, beta_vals_SiO2, color='red', label='Imaginary part (beta)')
    plt.title('Energies vs delta/beta Values for SiO2', fontsize=18)
    plt.xlabel('Photon Energy [eV]', fontsize=14)
    plt.ylabel('beta/delta Values', fontsize=14)
    plt.legend()
    plt.show()

    # Plot the refractive index for SU8
    plt.figure()
    plt.loglog(E_vals_SU8, delta_vals_SU8, color='blue', label='Real part (delta)')
    plt.loglog(E_vals_SU8, beta_vals_SU8, color='red', label='Imaginary part (beta)')
    plt.title('Energies vs delta/beta Values for SU8', fontsize=18)
    plt.xlabel('Photon Energy [eV]', fontsize=14)
    plt.ylabel('beta/delta Values', fontsize=14)
    plt.legend()
    plt.show()

# Interpolation to find the most accurate delta value for SiO2 corresponding to a specific beam energy
interp_delta_SiO2 = interp1d(E_vals_SiO2, delta_vals_SiO2, kind='cubic')
delta_SiO2 = interp_delta_SiO2(E)
interp_beta_SiO2 = interp1d(E_vals_SiO2, beta_vals_SiO2, kind='cubic')
beta_SiO2 = interp_beta_SiO2(E)

# Interpolation to find the most accurate delta value for SU8 corresponding to a specific beam energy
interp_delta_SU8 = interp1d(E_vals_SU8, delta_vals_SU8, kind='cubic')
delta_SU8 = interp_delta_SU8(E)
interp_beta_SU8 = interp1d(E_vals_SU8, beta_vals_SU8, kind='cubic')
beta_SU8 = interp_beta_SU8(E)

delta_SiO2 = torch.tensor(delta_SiO2, dtype=torch.float32)
beta_SiO2 = torch.tensor(beta_SiO2, dtype=torch.float32)
delta_SU8 = torch.tensor(delta_SU8, dtype=torch.float32)
beta_SU8 = torch.tensor(beta_SU8, dtype=torch.float32)

# I chose 400nm for the Gaussian and 10nm for the Lorentzian as it qualitatively matched the experimental data
sigma = torch.tensor(500e-9, dtype=torch.float32, requires_grad=True)  # Gaussian sigma guess, 400
gamma = torch.tensor(0e-9, dtype=torch.float32, requires_grad=True)  # Lorentzian gamma guess
x_c = torch.tensor(0e-6, dtype=torch.float32, requires_grad=True)
y_c = torch.tensor(0e-6, dtype=torch.float32, requires_grad=True)

# Create the void and embed it in the SU8 material
r_void_out = d_void_out / 2   # Radius of SiO2 void
SU8 = torch.zeros((Ny, Nx), dtype=torch.float32)  # Create 2D grid for SU8 material
SU8[...] = SU8_thickness  # Thickness of the SU8 along the x-ray propagation direction
arg_out = (r_void_out ** 2 - (x - x_c) ** 2 - (y - y_c) ** 2)  # Argument for a spherical bead
arg_out[arg_out < 0] = 0
t_void_out = 2.0 * torch.real(torch.sqrt(arg_out + 1e-15))  # Projection of spherical bead thickness to 2D
assert t_shell >= 0
r_void_in = (d_void_out - t_shell) / 2  # Inner radius of SiO2 void
arg_in = (r_void_in ** 2 - (x - x_c) ** 2 - (y - y_c) ** 2)  # Argument for spherical bead (smaller than tbead)
arg_in[arg_in < 0] = 0
t_in = 2.0 * torch.real(torch.sqrt(arg_in + 1e-15))  # Projection of spherical bead thickness to 2D
t_void = t_void_out - t_in
SU8 = SU8 - t_void_out  # Subtract the entire bead out from the SU8 material to get just SU8 thickness only

# Define separate learning rates for each parameter
learning_rates = {'sigma': 50e-9,  # was 50e-9
                  'gamma': 0e-9,  # was 5e-9
                  'x_c': 1e-6,  # 0.01 before
                  'y_c': 1e-6,  # 0.01 before
                  'd_void_out': 0.5e-6,
                  't_shell': 0.5e-6}  # 0.5

# Create a list of dictionaries specifying the parameters and learning rates
param_list = [{'params': [sigma], 'lr': learning_rates['sigma']},
              {'params': [gamma], 'lr': learning_rates['gamma']},
              {'params': [x_c], 'lr': learning_rates['x_c']},
              {'params': [y_c], 'lr': learning_rates['y_c']},
              {'params': [d_void_out], 'lr': learning_rates['d_void_out']},
              {'params': [t_shell], 'lr': learning_rates['t_shell']}]

optimizer = torch.optim.Adam(params=param_list)

sigma = sigma.requires_grad_(True)
gamma = gamma.requires_grad_(False)
x_c = x_c.requires_grad_(True)
y_c = y_c.requires_grad_(True)
d_void_out = d_void_out.requires_grad_(True)
t_shell = t_shell.requires_grad_(True)

loss_vals = []
iterations = 100
for i in tqdm(range(iterations)):
    optimizer.zero_grad()

    x_c_new = x_c
    y_c_new = y_c
    sigma_new = abs(sigma)
    gamma_new = abs(gamma)
    t_shell_new = t_shell

    r_void_out = d_void_out / 2  # Radius of SiO2 void
    SU8 = torch.zeros((Ny, Nx), dtype=torch.float32)  # Create 2D grid for SU8 material
    SU8[...] = SU8_thickness  # Thickness of the SU8 along the x-ray propagation direction
    arg_out = (r_void_out ** 2 - (x - x_c_new) ** 2 - (y - y_c_new) ** 2)  # Argument for a spherical bead
    arg_out[arg_out < 0] = 0
    t_void_out = 2.0 * torch.real(torch.sqrt(arg_out + 1e-12))  # Projection of spherical bead thickness to 2D
    r_void_in = (d_void_out - t_shell_new) / 2  # Inner radius of SiO2 void
    arg_in = (r_void_in ** 2 - (x - x_c_new) ** 2 - (y - y_c_new) ** 2)  # Argument for spherical bead
    arg_in[arg_in < 0] = 0
    t_in = 2.0 * torch.real(torch.sqrt(arg_in + 1e-12))  # Projection of spherical bead thickness to 2D
    t_void = t_void_out - t_in
    SU8 = SU8 - t_void_out

    print("d_void_out: ", d_void_out.item())
    print("t_shell: ", t_shell_new.item())
    print("sigma: ", sigma_new.item())
    print("gamma: ", gamma_new.item())
    print("x_c_new: ", x_c_new.item())
    print("y_c_new: ", y_c_new.item())

    # Phase map of SU8 with embedded SiO2 void (Projection approximation)
    phase = -k0 * (SU8 * delta_SU8 + t_void * delta_SiO2)  # Phase map
    mu = k0 * (SU8 * beta_SU8 + t_void * beta_SiO2)  # Attenuation map

    # If there is phase aliasing
    phase = resize_with_anti_aliasing(phase=phase, output_shape=(Ny, Nx), sigma=(3, 3))
    mu = resize_with_anti_aliasing(phase=mu, output_shape=(Ny, Nx), sigma=(3, 3))

    holo_sim, _ = simulate_hologram(fresnel_number=dx_eff ** 2 / (z_eff * lam),
                                    phase_image=phase,
                                    absorption_image=mu,
                                    beam=None,
                                    size_pad=[N_pad, N_pad],
                                    size_out=[Ny, Nx])

    holo_sim, _ = fadeoutImage(img=holo_sim,
                               fadeMethod='ellipse',
                               fadeToVal=1,
                               transitionLength=[20, 20],
                               ellipseSize=[0.85, 0.85])

    # May be required depending if the object is on the edge of the computational domain, manually adjust
    # holo_exp[0:1055, :] = 0
    # holo_sim[0:1055, :] = 0
    # holo_exp[2120:, :] = 0
    # holo_sim[2120:, :] = 0
    loss, conv_holo = func(sig=sigma_new,
                           gam=gamma_new,
                           simulated_hologram=holo_sim,
                           experimental_hologram=holo_exp)
    if plot:
        plt.figure()
        plt.imshow(conv_holo.detach().numpy(), cmap="Greys_r")
        plt.show()
        compare_plots_lineout(img1=holo_exp.detach(), img2=conv_holo.detach())
    imwrite("/Users/danielhodge/Desktop/sim.tiff", conv_holo.detach().numpy())
    imwrite("/Users/danielhodge/Desktop/exp.tiff", holo_exp.detach().numpy() * mask.detach().numpy())
    with torch.no_grad():
        loss_vals.append(loss)

    loss.backward()
    optimizer.step()
