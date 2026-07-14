import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time
from tifffile import imwrite
from skimage.transform import resize


save = True
plot_f1f2 = False
rescale = False

data_file = "/Users/danielhodge/Desktop/NanoDataSLAC_98.3fs.dat"

with open(data_file, 'r') as f:
    lines = f.readlines()

header_lines = lines[:12]
print("HEADER INFO:")
for line in header_lines:
    print(line.strip())

print("---------------------------------------------------------------------------------------------------------")

with open(data_file, 'r') as f:
    for line in f:
        if "Dimensions" in line:
            parts = line.strip().split(":")[-1].split()
            iy_len, iz_len, ix_len = map(int, parts)

        if "Sizes (cm)" in line:  # Sizes is the field of view, not voxel size
            parts = line.strip().split(":")[-1].split()
            fov_y, fov_z, fov_x = map(float, parts)

        if "Max electron density" in line:
            parts = line.strip().split(":")[-1].split()
            max_electron_density = np.array(parts, np.float64)[0]

print(f"iy_len = {iy_len}, iz_len = {iz_len}, ix_len = {ix_len}")
print(f"Y = {fov_y}, Z = {fov_z}, X = {fov_x}")
print(f"max_electron_density = {max_electron_density}")

print("---------------------------------------------------------------------------------------------------------")

data = np.loadtxt(data_file, skiprows=13)
iy, iz, ix = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(int)
electron_densities, ion1_densities, charge1, ion2_densities, charge2 = data[:, 3:].T  # Electron density in units atoms/cc

dx = fov_x / ix_len
dy = fov_y / iy_len
dz = fov_z / iz_len
extent_x = ix_len * dx * 1e7  # nm
extent_y = iy_len * dy * 1e7  # nm
extent_z = iz_len * dz * 1e7  # nm

E = 9000  # Energy in eV
r_e = 2.82e-13  # cm
lam_cm = 1240 / E * 1e-7  # cm
Ny = 2500  # Desired grid size
Nz = 2500  # Desired grid size


# rho_nickel = 8.902  # g/cc -- From CXRO --> density of nickel
# A = 58.6934  # g/mol -- Atomic weight of the atom --> See periodic table in Atwood book
# avo_num = 6.02214e23  # atoms/mol -- Avogadro's number
# n_a = rho_nickel * avo_num / A  # Page 20 Atwood
# print("Nickel atom density (atoms/cc):", n_a)


url = "https://henke.lbl.gov/optical_constants/asf.html"
element = "Ni"
chrome_options = Options()
driver = webdriver.Chrome(options=chrome_options)  # Opens Chrome

driver.get(url)
time.sleep(0.5)
driver.find_element(By.LINK_TEXT, element).click()
time.sleep(0.5)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
pre_tag = soup.find('pre')  # Obtains preformatted text which is to be presented exactly as written in the HTML file
data_text = pre_tag.get_text()

lines = data_text.splitlines(True)  # Splits string into a list
data = lines[1:]  # Skip the header (first line)
E_vals, f1_vals, f2_vals = np.loadtxt(data, unpack=True)

if plot_f1f2:
    plt.figure()
    plt.loglog(E_vals, f1_vals, color='blue', label='f1')
    plt.loglog(E_vals, f2_vals, color='red', label='f2')
    plt.title('Energies vs f1/f2 Values for ' + element, fontsize=18)
    plt.xlabel('Photon Energy [eV]', fontsize=14)
    plt.ylabel('f1/f2 Values', fontsize=14)
    plt.legend()
    plt.show()

# Interpolation to find the most accurate delta value for Ni corresponding to a specific beam energy
interp_f1 = interp1d(E_vals, f1_vals, kind='linear')
f1 = interp_f1(E)
interp_f2 = interp1d(E_vals, f2_vals, kind='linear')
f2 = interp_f2(E)

print("f1 for " + element + " at " + str(E) + " eV is:", f1)
print("f2 for " + element + " at " + str(E) + " eV is:", f2)

delta = electron_densities * r_e * lam_cm**2 * f1 / (2 * np.pi)
beta = electron_densities * r_e * lam_cm**2 * f2 / (2 * np.pi)

print("Real part of the refractive index (delta) is:", delta)
print("Imaginary part of the refractive index (beta) is:", beta)

delta = delta.reshape(ix_len, iy_len, iz_len)
beta = beta.reshape(ix_len, iy_len, iz_len)
delta = np.sum(delta, axis=0)
beta = np.sum(beta, axis=0)

# Rescale to a size corresponding to the detector size
if rescale:
    delta = resize(delta,
                   (Ny, Nz),
                   mode='constant',
                   order=3,
                   anti_aliasing=True,
                   anti_aliasing_sigma=1)

    beta = resize(beta,
                  (Ny, Nz),
                  mode='constant',
                  order=3,
                  anti_aliasing=True,
                  anti_aliasing_sigma=1)

plt.figure()
plt.imshow(delta, cmap='Greys_r', extent=(-extent_y / 2, extent_y / 2, -extent_z / 2, extent_z / 2))
plt.title("Phase")
plt.xlabel("Length [nm]")
plt.ylabel("Width [nm]")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(beta, cmap='Greys_r', extent=(-extent_y / 2, extent_y / 2, -extent_z / 2, extent_z / 2))
plt.title("Attenuation")
plt.xlabel("Length [nm]")
plt.ylabel("Width [nm]")
plt.colorbar()
plt.show()

if save:
    imwrite("/Users/danielhodge/Desktop/nw_delta.tiff", delta)
    imwrite("/Users/danielhodge/Desktop/nw_beta.tiff", beta)

driver.quit()

