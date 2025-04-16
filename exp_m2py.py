import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftshift, fft2, ifft2

# -- Radar System Operational Parameters
fBW = 8e9
fc = 79.6e9
wc = 2 * np.pi * fc
c = 3e8
theta_b = 60  # degrees

wBW = 2 * np.pi * fBW
RR = c / (2 * fBW)

# -- Fast-Time domain parameters and arrays
fs = 50e6
T = 48e-6
N = int(fs * T)
gamma = -wBW / T

dt = 1 / fs
Tp = (N - 1) * dt
t = np.linspace(-Tp / 2, Tp / 2, N)
range_arr = np.arange(0, RR * N, RR)

# -- Slow-Time domain parameters and arrays
spacing = 1.6e-3
xp = np.arange(-200e-3, 200e-3 + spacing, spacing)
yp = np.arange(-200e-3, 200e-3 + spacing, spacing)
zp = 0

Mx = len(xp)
My = len(yp)
M = Mx * My

# -- Load raw and background data
ch1 = np.fromfile('Imaging_raw_data/Mannequin_Measurement_1.ch0', dtype=np.float64)
ch1_bkg = np.fromfile('Imaging_raw_data/Bkg_Mannequin_Measurement_1.ch0', dtype=np.float64)

# -- Reshape and subtract background
valid_samples_per_trigger = int(fs * T)
ch1b = ch1.reshape(valid_samples_per_trigger, M, order='F')  # MATLAB uses column-major order

ch1_bkg = ch1_bkg.reshape((valid_samples_per_trigger, 1))  # Shape (2400, 1)
ch1_bkg = np.tile(ch1_bkg, (1, ch1b.shape[1]))             # Shape (2400, 63001)

sdata_time = ch1b - ch1_bkg

sdata_time = ch1b - ch1_bkg

# -- Calibration Step 2
del_lt = 19.1e-2
exp_shift = np.exp(-1j * (wc * 2 * del_lt / c - 2 * gamma * del_lt * t / c))[:, np.newaxis]
sdata_time2 = sdata_time * exp_shift

# -- Reformatting raster scan to XY grid
print("Reformatting raster scan to XY grid")
s_b = np.zeros((Mx, My, N), dtype=complex)
main_ind = 0

for Myind in range(My-1, -1, -1):
    if (Myind + 1) % 2 == 0:
        for Mxind in range(Mx):
            s_b[Mxind, Myind, :] = sdata_time2[:, main_ind]
            main_ind += 1
    else:
        for Mxind in range(Mx-1, -1, -1):
            s_b[Mxind, Myind, :] = sdata_time2[:, main_ind]
            main_ind += 1

# -- Precomputing K-space
Kt = gamma * t / c + wc / c
k_limit = (2 * wc / c) * np.sin(np.deg2rad(theta_b / 2))

dkx = 2 * np.pi / (2 * np.max(xp))
dky = 2 * np.pi / (2 * np.max(yp))
kx = dkx * np.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1)
ky = dky * np.arange(-(My - 1) / 2, (My - 1) / 2 + 1)

kx_trunc = kx[np.abs(kx) <= k_limit]
ky_trunc = ky[np.abs(ky) <= k_limit]
Mxn = len(kx_trunc)
Myn = len(ky_trunc)

kx3D = np.tile(kx_trunc[:, np.newaxis, np.newaxis], (1, Myn, N))
ky3D = np.tile(ky_trunc[np.newaxis, :, np.newaxis], (Mxn, 1, N))
Kt3D = np.tile(Kt[np.newaxis, np.newaxis, :], (Mxn, Myn, 1))
kz3D = np.sqrt(4 * Kt3D**2 - kx3D**2 - ky3D**2)

kz1D_uni = kz3D[Mxn // 2, Myn // 2, :]
kz3D_uni = np.tile(kz1D_uni[np.newaxis, np.newaxis, :], (Mxn, Myn, 1))

truncX_range = np.where(np.abs(kx) <= k_limit)[0]
truncY_range = np.where(np.abs(ky) <= k_limit)[0]

# -- Reconstruction - Algorithm
print("Starting algorithm. Est. time: 30 seconds")

# Step 1: 2D FFT
print("First 2-D FFT")
s_B = fftshift(fft2(s_b, axes=(0, 1)), axes=(0, 1))

# Step 2: Stolt Interpolation
print("Stolt Interpolation")
s_B2 = np.zeros((Mxn, Myn, N), dtype=complex)

for indX in range(Mxn):
    for indY in range(Myn):
        kz = kz3D[indX, indY, :]
        s_line = s_B[truncX_range[indX], truncY_range[indY], :]
        interpolator = interp1d(kz, s_line, kind='linear', bounds_error=False, fill_value=0)
        s_B2[indX, indY, :] = interpolator(kz1D_uni)

# Step 3: 3D IFFT
print("Last 3-D IFFT")
f_hat = fft( ifft2(fftshift(s_B2, axes=(0,1)), axes=(0,1)), axis=2 )

# -- Plotting Final image
dx = 2 * np.pi / (2 * np.max(kx_trunc))
dy = 2 * np.pi / (2 * np.max(ky_trunc))
dz = 2 * np.pi / (np.max(kz1D_uni) - np.min(kz1D_uni))

xIm = dx * np.arange(-(Mxn - 1) / 2, (Mxn - 1) / 2 + 1)
yIm = dy * np.arange(-(Myn - 1) / 2, (Myn - 1) / 2 + 1)
distZ = dz * np.arange(N)

X, Y = np.meshgrid(xIm * 100, yIm * 100, indexing='ij')
maxAll = np.max(np.abs(f_hat))

for zInd in range(26):
    f = np.abs(f_hat[:, :, zInd]) / maxAll
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(X, Y, f, shading='auto', cmap='gray')
    plt.clim(0, 1)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title(f'Z distance: {distZ[zInd]:.2f}')
    plt.axis('equal')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.colorbar()
    plt.pause(1)
    plt.close()