# ENSC 461 Final Project: Range Doppler Algorithm

**DISCLAIMER:** These notes are entirely copy pasted from the Cummings & Wong "Digital Processing of Synthetic Aperture Data". These ideas and procedures are not my original ideas and they will be cited properly at the end of this project. The intention is to build a proper SAR processing algorithm for an applied purpose.

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

# 4 Range Cell Migration Correction (RCMC)

There are two ways to implement range cell migration correction (RCMC). In the first option, RCMC is performed by a range interpolation operation in the range Doppler domain. An interpolator based on the sine function can be conveniently implemented. The sine kernel is truncated and weighted by a tapering window, such as a Kaiser window.

The amount of RCM to correct is given by 

$$
\Delta R(f_η) = \frac{λ^2 R_0 f_η^2}{8 V_r^2}
$$

This equation represents the target displacement as a function of azimuth frequency, $f_η$. Note that $\Delta R(f_η)$ is also a function of $R_0$; that is, it is range variant. Since one of the dimensions in the data is range time, the RDA can correctly implement the range variation of the RCM in the range Doppler domain. Another RCMC implementation involves the assumption that the RCM is range invariant, at least over a finite range region. In this case, the RCMC can be implemented using an FFT, linear phase multiply, and IFFT technique. The phase multiplier, for a given $f_η$ is given by

$$
G_{rcmc}(f_\tau) = \exp\left\{j\frac{4\pi f_\tau \Delta R\left(f_{\eta}\right)}{c}\right\}
$$

To apply RCMC using this implementation, small range blocks can be used, with the correction amount held constant within each block. However, this implementation has the disadvantage that the blocks have to overlap in range, and the efficiency gain may not be worth the added complexity. From now on, the sine interpolation option is assumed. Assuming the RCMC interpolation is applied accurately, the signal becomes

$$
S_2(\tau,f_η) = A_0 p_r\left(\tau - \frac{2R_0}{c}\right) W_a(f_η - f_{η_c}) \cdot \exp\left\{-j\frac{4\pi f_{0}R_{0}}{c}\right\}\exp\left\{j\pi\frac{f_{\eta}^{2}}{K_{a}}\right\}
$$

Note that the range envelope, $p_r$, is now independent of azimuth frequency, showing that the RCM has been corrected. In addition, the energy is now centered at $\tau = 2R_0/c$, the range of closest approach.

There are three issues in the generation and application of the RCMC interpolator: the kernel length, the shift quantization, and the coefficient values ( the sine window). To perform the interpolation efficiently, the interpolation kernel can be tabulated at predefined subsample ( quantization) intervals. In this way, the sine function does not need to be generated at every interpolation point; only nearest neighbor indexing into the tabulated function is required. This introduces a maximum geometric distortion of $0.5 N_{\text{sub}}$, where $N_{\text{sub}}$ is the number of subsamples (typically 16). You can use generation of the kernel coefficients.

or efficiency considerations, the size of the interpolator kernel should be as short as possible. However, a short kernel has two drawbacks: the loss of radiometric and phase accuracy, and the introduction of radiometric artifacts known as paired echoes.

In practice, a compromise is made between no interpolation (nearest neighbor selection), and perfect interpolation (infinite length). Usually, a four- or eight-point interpolator is chosen to give reasonable accuracy. A quantization of 1/16 or 1/32 of a cell is used in practice. With a four-point interpolator, it is seen that the modulation is lower than say using Nearest Neighbor, and the paired echoes are reduced to 28 dB below the target peak.

The RCMC algorithm operates on the data in the range Doppler domain, which is the plot from the Azimuth FFT. After the RCMC interpolator has been applied, the data takes on a straightened form.

![a3efcdd7-967e-43f8-85f7-391920f105b2](file:///C:/Users/vital/Pictures/Typedown/a3efcdd7-967e-43f8-85f7-391920f105b2.png)

Image registration operations can be incorporated into the RCMC interpolation, namely, slant range to ground range (SRGR) conversion and scaling the sample spacing to a map grid. 

The natural slant range output sample spacing is given by $c/(2F_r \sin{θ_i})$, where $θ_i$ is the range-dependent incidence angle. Both factors require interpolation in the range direction, and therefore can be combined with the RCMC operation.



# 5  Azimuth Compression

In azimuth, there is usually some latitude in the choice of processed resolution. This is because the azimuth signal bandwidth is often greater than needed to make the azimuth resolution the same as the range resolution. "Full-resolution" ( or "single-look") processing can be done, using all the bandwidth, and achieving a resolution close to the theoretical limit of one-half the antenna length. On the other hand, "multilook" processing can be done, in which the data are processed to a resolution less than this limit, to obtain a less noisy image. One-look processing is discussed in this section.

After RCMC, a matched filter is applied to focus the data in the azimuth direction. Since the data after RCMC are in the range Doppler domain, it is convenient and efficient to implement the azimuth matched filter in this domain; that is, as a function of slant range, $R_0$, and azimuth frequency, $f_η$. The matched filter is the complex conjugate of the second exponential term in $S_2(\tau,f_η)$.

$$
H_{\text{az}}(f_η) = \exp\left\{-j\pi\frac{f_{\eta}^{2}}{K_{a}}\right\}
$$

in which $K_a$ is a function of $R_0$ like we did in $K_a \approx \frac{2V_r^2}{λR_0}$. This version of the matched filter is implemented. 

The Doppler centroid is an important parameter in generating the matched filter in much the same way as it is for RCMC. Recall that a point of discontinuity in the $f_η$ array must be selected, due to the wraparound in the frequency domain. This point is taken to be one-half of a PRF away from the Doppler centroid frequency.

The compressed result is registered to zero Doppler with this filter. The registration is correct because the phase of each target is canceled by the matched filter, except for a linear phase component that gives each target its unique position in the output array.

Weighting can be applied in the azimuth compression process. Because the azimuth beam profile already applies a significant amount of weighting in the single-look case, only a small amount of additional matched filter weighting is generally needed. As the magnitude of the two-way beam pattern at the edges of the processed beamwidth is approximately one-half of its peak value, the effective beam weighting is equivalent to a Kaiser window with a coefficient, $β$, equal to 1.8. Therefore, if a total weighting is desired that is equivalent to $β = 2.5$, only a light additional weighting need be applied with the matched filter. Because of the existing antenna weighting, either a moderate window can be used to give a small additional amount of tapering, or no window need be used. 



To perform azimuth compression, the data after RCMC, $S_2(\tau,f_η)$ are multipled by the frequency domain matched filter, $H_{\text{az}}(f_η)$. The result is

$$
S_3(\tau,f_η) = S_2(\tau,f_η) H_{\text{az}}(f_η)
$$

An IFFT then completes the compression

$$
S_ac(\tau,η) = \text{IFFT}_η\left\{S_{3}\left(\tau,f_{\eta}\right)\right\}
$$

The envelopes show that the target is now positioned at $\tau = 2R_0/c$ and $η = 0$. Recalling that 77 is relative to the time of closest approach when zero Doppler occurs for the given target, it is seen that the target is registered to its zero Doppler position.

It must be emphasized that the phase mentioned above is an approximation when the parabolic form of the range equation given in (6.5) or (6.10) is used. When this approximation is used, the processor may not be phase preserv.ing for nonzero squints. The current practice is to use the hyperbolic form of the phase for the matched filter for low squint cases when phase precision is needed.



Please provide four plots in this window (a) Compressed signal with Azimuth (samples) vs. Range time (samples), (b) Target C spectrum with Azimuth freq. (samples) vs. Range freq (samples), (c) Expanded Target C with Azimuth (samples) vs. Range (samples), (d) Expanded Target C contours with Azimuth (sample) vs. Range (samples) 


