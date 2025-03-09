# ENSC 461 Final Project: Range Doppler Algorithm

The Range Doppler Algorithm (RDA) is designed to achieve block processing efficiency, using frequency domain operations in both range and azimuth, while maintaining the simplicity of one-dimensional operations. It takes advantage of the approximate separability of processing in these two directions, allowed by the large difference in time scales of the range and azimuth data, and by the of range cell migration correction (RCMC) between the two one-dimensional operations.

From the book there's three implementations:

(a) Basic RDA, (b) RDA with accurate SRC, (c) RDA with approximate SRC

We will use **RDA with accurate SRC**.

Range cell migration correction ensures that the SAR image has better resolution and fewer distortions, resulting in clearer and more precise imagery.

They commonly share these steps:

1. Range compression is performed with a fast convolution when the data are in the zimuth time domain.

2. An zimuth FFT transforms the data into the range Doppler domain, where Doppler estimation and most of the subsequent operations are performed (see Chapter 12 for Doppler estimation).

3. RCMC performed in the range Doppler domain, to straighten out these trajectories so that they now run parallel to the azimuth frequency axis.

4. Azimuth matched filtering can be conveniently performed as a frequency domain matched filter multiply at each range gate.

5. The final step is azimuth IFFT to transform the data back to the time domain, resulting in a compressed complex image. Detection and look summation can be done at this stage, if desired.

<img src="file:///C:/Users/vital/Pictures/Typedown/3fa5961d-189a-4b2c-85d2-08081603fa54.png?msec=1741389241343" title="" alt="3fa5961d189a4b2c85d208081603fa54" data-align="center">

Specifically for RDA with accurate SRC we seem to have these steps:

1. Start: Raw radar data

2. Range Compression without the IFFT
   
   - The radar signal, which spreads out over time (delaying echoes from distant objects), is mathematically compressed back into sharp peaks, improving the resolution and clarity of objects along the radar's line of sight .

3. Azimuth FFT

4. SRC Option 2 and range IFFT

5. RCMC (Range Cell Migration Correction)
   
   - aligns reflected signals from a target to their correct range positions, ensuring the radar image is sharp and focused by compensating for **target motion effects***.

6. Azimuth Compression

7. Azimuth IFFT and Look Summation

8. End: Compressed data

# 1 Raw radar data

The data received from the radar system are referred to as "signal data" or "raw data." The data are first demodulated to baseband, so that the nominal center range frequency is zero. The demodulated radar signal, $s_0(\tau,η)$, received from a point target can be modeled as:

$$
\begin{aligned}        s_0(\tau,\eta) &= A_0 w_r[\tau - 2R(\eta)/c]w_a(η-η_c) \cdot \exp\{-jπK_r(\tau - 2R(η)/c)^2\ \\\\\text{where} \quad A_0 &= \text{an arbitrary complex constant} \\                  \tau &= \text{range time} \\                  \eta &= \text{azimuth time referenced to closest approach} \\                \eta_c &= \text{beam center offset time} \\             w_r(\tau) &= \text{range envelope (a rectangular function)} \\             w_a(\eta) &= \text{azimuth envelope (a sine-squared function)} \\                   f_0 &= \text{radar center frequency} \\                   K_r &= \text{range chirp FM rate} \\                  R(η) &= \text{instantaneous slant range}\end{aligned}
$$

A linear FM radar pulse is assumed, having an FM rate, $K_r$. The two $w$ terms model the magnitudes of the range and azimuth signals, and are often neglected in the signal analysis. The instantaneous slant range, $R(η)$ is given by

$$
R(η) = \sqrt{R_0^2 + V_r^2 η^2}
$$

where $R_0$ is the slant range of closest approach.

# 2 Range Compression

Let $S_0(f_\tau, \eta)$ be the range Fourier transform of $s_0 (\tau, \eta)$ of (6.1), and $G(f_\tau)$ be the frequency domain matched filter defined in (3.19). The output of the range matched filter can be expressed as

$\begin{aligned} s_{rc}(\tau,\eta) &= \text{IFFT}_\tau \{S_0 (f_\tau , \eta) G(f_\tau)\} \\ &= A_0 p_r[\tau - 2R(\eta)/c] w_a(\eta - \eta_c) \exp\{-j 4π f_0 R(η)/c\} \end{aligned}$

where the compressed pulse envelope, $p_r(\tau)$, is the IFFT of the window, $W_r(f_\tau)$. For a rectangular window, $p_r(\tau)$ is a sine function, and for a tapered window it is a sine-like function with lower sidelobes. The slant range resolution, derived in Section 4.4, is

$$
\rho_r = \frac{c}{2} \frac{0.886 γ_{w,r}}{|K_r|T_r}
$$

where $\gamma_{w,r}$ is the IRW broadening factor due to the tapering window, $|K_r|T_r$ is the chirp bandwidth, and the factor c/2 expresses the resolution in distance rather than time units. The broadening factor, $\gamma_{w,r}$, equals one when a rectangular window is used, and results in a PSLR of - 13 dB. Instead, a Kaiser window with roll-off coefficient of $\beta = 2.5$ is used in the simulation experiments. Then, $γ_{w,r} = 1.18$, corresponding to an 18% broadening of the IRW, and the resulting PSLR is -21 dB (recall Figure 2.12).



<img src="https://www.researchgate.net/publication/257877050/figure/fig2/AS:300952939712512@1448764180111/SAR-data-of-point-target-a-Real-part-of-SAR-raw-data-b-Compressive-SAR-images.png" title="" alt="SAR data of point target. (a) Real part of SAR raw data. (b)... | Download  Scientific Diagram" data-align="center">

<img src="file:///C:/Users/vital/Pictures/Typedown/53c31885-590c-4d76-86bc-18289d8081ac.png" title="" alt="53c31885-590c-4d76-86bc-18289d8081ac" data-align="center">

# 3 Azimuth Fourier Transform

In low squint cases, the antenna beam points close to the zero Doppler direction. The range equation can be approximated by the parabolic equation (5.1), if the aperture is not too large

$$
R(η) = \sqrt{R_0^2 + V_r^2\eta^2} \approx R_0 + \frac{V_r^2 η^2}{2R_0}
$$

Combining the range compression equation $s_{rc}(\tau,η)$ and the $R(η)$ equation, the range compressed signal can be expressed as

$$
s_{rc}(\tau,η) \approx A_0 p_r[\tau - 2\frac{R(η)}{c}] w_a(η-η_c) \cdot \exp\{-j \frac{4π f_0 R_0}{c}\} \exp\{-jπ \frac{2V_r^2}{\lambda R_0} η^2\}  \quad\quad (6.6)
$$

The azimuth phase modulation is now apparent in the second exponential phase term. Since the phase is a function of $η^2$ , the signal has linear FM characteristics, with the linear FM rate being

$$
K_a = \frac{2V_r^2}{\lambda R_0}
$$

The FM rate is derived in Section 4.5.5, resulting in $K_a = \frac{2}{\lambda} \frac{d^2R(η)}{dη^2}|_{η=η_c} = \frac{2V_r^2 \cos^2{θ_{r,c}}}{\lambda R(η_c)} = \frac{2V_r^2 \cos^3{θ_{r,c}}}{\lambda R_0}$, and if you assume that $\cos^3{θ_{r,c}}$

is equal to unity for small squint angles then you get $K_a = \frac{2V_r^2}{\lambda R_0}$.

An azimuth FFT is then performed on each range gate to transform the data into the range Doppler domain. In deriving the signal in this domain, only the second exponential phase term in (6.6) is important, as the first exponential term is a constant for a given target. By applying the POSP, the relationship between azimuth frequency and time is $f_\eta = -K_a \eta$. By subbing $\eta = - f_η/K_a$ into (6.6), the data after the zimuth FFT can be expressed as

$$
S_1(\tau,f_\eta) = \text{FFT}_η\{s_{rc}(\tau,\eta)\} = A_0 p_r[\tau - \frac{2 R_{rd}(f_\eta)}{c}] W_a(f_η-f_{η_c}) \cdot \exp\{-j \frac{4π f_0 R_0}{c}\} \exp\{jπ \frac{f_η^2}{K_a}\}
$$

The result should be:

<img src="file:///C:/Users/vital/Pictures/Typedown/00d7b824-8811-4362-9978-a52cd7059893.png" title="" alt="00d7b824-8811-4362-9978-a52cd7059893" data-align="center">


