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
   
   

![3fa5961d-189a-4b2c-85d2-08081603fa54](file:///C:/Users/vital/Pictures/Typedown/3fa5961d-189a-4b2c-85d2-08081603fa54.png)



Specifically for RDA with accurate RDA we seem to have these steps:

1) Start: Raw radar data

2) Range Compression without the IFFT

3) Azimuth FFT

4) SRC Option 2 and range IFFT

5) RCMC

6) Azimuth Compression

7) Azimuth IFFT and Look Summation

8) End: Compressed data
   
   

# 1  Raw radar data

The data received from the radar system are referred to as "signal data" or "raw data." The data are first demodulated to baseband, so that the nominal center range frequency is zero. The demodulated radar signal, $s_0(\tau,η)$, received from a point target can be modeled as:

$$
\begin{aligned}
        s_0(\tau,\eta) &= A_0 w_r[\tau - 2R(\eta)/c]w_a(η-η_c) \cdot \exp\{-jπK_r(\tau - 2R(η)/c)^2\ \\
\\
\text{where} \quad A_0 &= \text{an arbitrary complex constant} \\
                  \tau &= \text{range time} \\
                  \eta &= \text{azimuth time referenced to closest approach} \\
                \eta_c &= \text{beam center offset time} \\
             w_r(\tau) &= \text{range envelope (a rectangular function)} \\
             w_a(\eta) &= \text{azimuth envelope (a sine-squared function)} \\
                   f_0 &= \text{radar center frequency} \\
                   K_r &= \text{range chirp FM rate} \\
                  R(η) &= \text{instantaneous slant range}
\end{aligned}
$$

A linear FM radar pulse is assumed, having an FM rate, $K_r$. The two $w$ terms model the magnitudes of the range and azimuth signals, and are often neglected in the signal analysis. The instantaneous slant range, $R(η)$ is given by 

$$
R(η) = \sqrt{R_0^2 + V_r^2 η^2}
$$

where $R_0$ is the slant range of closest approach.
