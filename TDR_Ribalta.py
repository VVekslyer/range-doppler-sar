import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1) DEFINE RADAR/SIMULATION PARAMETERS FROM YOUR TABLE
##############################################################################

c     = 3e8           # speed of light (m/s)

# From the table, e.g., sweepCenterFreq = 8.0E10 => 80 GHz
f0    = 8.0e10        # center frequency (Hz)

# sweepBandwidth = 2.0E10 => 20 GHz
B     = 2.0e10        # total FMCW sweep bandwidth

# sweepDuration = 0.006 => 6 ms
T     = 0.006         # pulse (chirp) length in seconds

# chirp rate K
K     = B / T         # ~ 3.333e12 Hz/s

# Some suitable PRF for demonstration
PRF   = 20            # pulse repetition frequency (Hz)
PRI   = 1.0 / PRF     # pulse repetition interval (sec)

# We'll do only 16 pulses just as a toy example
M     = 16

# For a 6 ms chirp, a real system might sample at 100 MSps or more
# => ~600,000 fast-time samples.  Here we use 512 to keep it small:
Nfast = 512
t_fast = np.linspace(0, T, Nfast)

# Radar trajectory: 1 cm/s in a simple linear path
v_mag = 0.01          # 0.01 m/s => 1 cm/s
theta = 0.0           # along x-axis
v_vec = v_mag * np.array([np.cos(theta), np.sin(theta)])
p0    = np.array([0.0, 0.0])  # initial position


##############################################################################
# 2) Define functions for distance, radar position, and phase
##############################################################################

def radar_position(m, t):
    """
    Returns the radar 2D position at the m-th pulse
    (plus t seconds into that pulse).
    """
    return p0 + v_vec*(m*PRI + t)

def distance(a, b):
    """ Euclidean distance in 2D. """
    return np.linalg.norm(a - b)

def phi(x, m, t):
    """
    The simplified phase function:
       phi(x; m, t) = [2/c * distance(x, radar_position(m, t))] * [f0 + K*t].
    """
    rng = distance(x, radar_position(m, t))
    return (2.0 * rng / c) * (f0 + K*t)

##############################################################################
# 3) Generate the raw phase history s_x(m,t) = exp{ j * 2π * phi(...) }
##############################################################################

# Fast-time samples
t_fast = np.linspace(0, T, Nfast)

# Allocate the raw data (phase history)
s_x = np.zeros((M, Nfast), dtype=np.complex128)

for m in range(M):
    for n, tn in enumerate(t_fast):
        s_x[m, n] = np.exp( 1j * 2.0*np.pi * phi(f0, m, tn) )

##############################################################################
# 4) Perform range compression
##############################################################################

# Generate the transmitted chirp (reference signal)
chirp_tx = np.exp(-1j * 2.0 * np.pi * (f0 * t_fast + 0.5 * K * t_fast**2))

# Apply matched filtering via FFT-based convolution
s_x_rc = np.zeros_like(s_x, dtype=np.complex128)
for m in range(M):
    s_x_rc[m, :] = np.fft.ifft(np.fft.fft(s_x[m, :]) * np.fft.fft(chirp_tx))

##############################################################################
# 5) Display the range-compressed signal
##############################################################################

plt.figure(figsize=(7,4))
plt.title("Magnitude of range-compressed signal")
plt.imshow(
    np.abs(s_x_rc),
    aspect='auto',
    extent=[0, Nfast, M, 0]
)
plt.xlabel("Fast-time sample index")
plt.ylabel("Pulse index (slow-time)")
plt.colorbar(label="Magnitude")
plt.show()
