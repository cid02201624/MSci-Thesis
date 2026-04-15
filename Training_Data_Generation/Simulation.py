"""
Author: Saskia Knight
Date: November 2025
Description: Simulated Data Generation Module
This module provides classes and functions to generate simulated gravitational-wave data (noise, signals, and glitches).
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import torch
import torch.nn as nn
import torch.nn.functional as F

# GW Software
from pycbc import catalog
from pycbc.frame import query_and_read_frame, read_frame
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import matched_filter, resample_to_delta_t
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.filter import sigma, sigmasq
from pycbc.psd import interpolate as interpolate_psd

from gwpy.timeseries import TimeSeries as GwpyTimeSeries

# params = {
# #    'axes.labelsize': x,
# #    'font.size': x, 
#     'font.family': 'Times New Roman'
# #    'legend.fontsize': x,
# #    'xtick.labelsize': x,
# #    'ytick.labelsize': x,
# #    'figure.figsize': [x, y]
#    } 
# plt.rcParams.update(params)


#%%
# NOISE GENERATION

class NoiseGenerator:
    """
    Generate time-domain noise of various types:
    - 'white': flat spectrum
    - 'pink':  1/f
    - 'red':   1/f^2 (Brownian)
    - 'blue':  f
    - 'pycbc': noise colored by a gravitational-wave PSD model (e.g. aLIGO)

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz (e.g. 4096.0).
    seglen : float
        Segment length in seconds.
    f_lower : float
        Lower cutoff frequency (Hz), relevant for PSD-based noise.
    psd_model : callable, optional
        Function returning a PyCBC PSD object. Default: aLIGOZeroDetHighPower.
    seed : int, optional
        Random seed for reproducibility.
    noise_type : str, optional
        Type of noise to generate: 'white', 'pink', 'red', 'blue', or 'pycbc'.
    """

    def __init__(self, sample_rate=4096.0, seglen=8.0, f_lower=20.0,
                 psd_model=aLIGOZeroDetHighPower, seed=None,
                 noise_type='pycbc'):
        self.sample_rate = sample_rate
        self.seglen = seglen
        self.f_lower = f_lower
        self.psd_model = psd_model
        self.noise_type = noise_type.lower()
        self.seed = seed if seed is not None else np.random.randint(1, 2**31 - 1)
        self.dt = 1.0 / sample_rate
        self.n = int(seglen * sample_rate)
        self.delta_f = 1.0 / seglen
        self.rng = np.random.default_rng(self.seed)

    def _make_psd(self):
        """Generate PSD for pycbc noise."""
        return self.psd_model(self.n // 2 + 1, self.delta_f, self.f_lower)

    def _coloured_noise(self, colour):
        """
        Generate coloured noise (white, pink, red, blue) using frequency-domain shaping.
        """
        # Frequency axis
        freqs = np.fft.rfftfreq(self.n, d=self.dt)
        freqs[0] = freqs[1]  # avoid division by zero

        # Generate white noise in frequency domain
        white = self.rng.normal(0, 1, len(freqs)) + 1j * self.rng.normal(0, 1, len(freqs))

        # Define spectral slope
        if colour == 'white':
            scale = np.ones_like(freqs)
        elif colour == 'pink':
            scale = 1.0 / np.sqrt(freqs)
        elif colour == 'red':
            scale = 1.0 / freqs
        elif colour == 'blue':
            scale = np.sqrt(freqs)
        else:
            raise ValueError(f"Unsupported noise colour: {colour}")

        # Apply scaling
        spectrum = white * scale
        # Symmetrise and IFFT to time domain
        time_series = np.fft.irfft(spectrum, n=self.n)
        # Normalise to unit variance
        time_series /= np.std(time_series)
        return TimeSeries(time_series, delta_t=self.dt)

    def generate(self, epoch=0.0):
        """
        Generate a PyCBC TimeSeries of the specified noise type.

        Parameters
        ----------
        epoch : float
            Start time of the noise segment.

        Returns
        -------
        noise : pycbc.types.TimeSeries
            The generated noise time series.
        """
        if self.noise_type == 'pycbc':
            psd = self._make_psd()
            noise = noise_from_psd(self.n, self.dt, psd, seed=int(self.rng.integers(1, 2**31 - 1)))
            return TimeSeries(noise.numpy(), delta_t=self.dt, epoch=epoch)

        elif self.noise_type in ['white', 'pink', 'red', 'blue']:
            noise = self._coloured_noise(self.noise_type)
            noise.start_time = epoch
            return noise

        else:
            raise ValueError(f"Invalid noise type: {self.noise_type}")

# %%
# SIGNAL GENERATION


class SignalGenerator:
    """
    Generate time-domain gravitational-wave signals using PyCBC.
    Produces strain time series for the two LIGO detectors (H1 and L1).

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz.
    seglen : float
        Segment length in seconds.
    f_lower : float
        Lower cutoff frequency (Hz).
    approximant : str
        Waveform model (e.g. 'IMRPhenomXPHM', 'SEOBNRv4', etc.)
    """

    def __init__(self, 
                 sample_rate=4096.0, 
                 seglen=4.0, 
                 f_lower=20.0,
                 approximant="IMRPhenomXPHM", #IMRPhenomD_NRTidalv2, IMRPhenomNSBH
                 source_class=None,   # "BBH", "BNS", "BHNS"
                 mass_ranges=None,     # optional override
                 tidal_enabled=True,
                 lambda_max=5000.0,    # rough prior cap for NS tidal deformability
                 snr_enabled=True, 
                 snr_min=2.0, 
                 snr_max=20.0, 
                 snr_dist="loguniform",
                 veto_enabled=True,     # Make sure min SNR is achieved
                 min_det_snr=5.0,
                 max_veto_tries=20,
                 resample=("sky", "pol", "time"),
                 seed = None,
                 ):

        self.sample_rate = sample_rate
        self.seglen = seglen
        self.f_lower = f_lower
        self.dt = 1.0 / sample_rate
        self.n = int(seglen * sample_rate)
        self.approximant = approximant

        self.H1 = Detector("H1")
        self.L1 = Detector("L1")

        self.snr_enabled = snr_enabled
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.snr_dist = snr_dist        # "loguniform" or "uniform"

        self.seed = seed if seed is not None else np.random.randint(1, 2**31 - 1)
        self.rng = np.random.default_rng(self.seed)
        self.veto_enabled = veto_enabled
        self.min_det_snr = float(min_det_snr)
        self.max_veto_tries = int(max_veto_tries)
        self.resample = set(resample)

        self.source_class = source_class
        self.tidal_enabled = bool(tidal_enabled)
        self.lambda_max = float(lambda_max)

        # Only these three approximants
        self.approximants = {
            "BBH":  "IMRPhenomXPHM",
            "BNS":  "IMRPhenomD_NRTidalv2",
            "BHNS": "IMRPhenomNSBH"}

        # Only these accept lambda1/lambda2 in your setup
        self.tidal_approximants = {"IMRPhenomD_NRTidalv2", "IMRPhenomNSBH"}

        # Mass ranges (Msun). Override by passing mass_ranges dict if you want.
        self.mass_ranges = mass_ranges or {
            "BBH":  {"m1": (10.0, 85.0), "m2": (5.0, 85.0)},
            "BNS":  {"m1": (1.0,  2.3),  "m2": (1.0,  2.3)},
            "BHNS": {"mbh": (5.0, 85.0), "mns": (1.0, 2.3)}}

    def generate(self,
                 source_class=None,
                 lambda1=None, lambda2=None,
                 mass1 = None, mass2 = None,
                 spin1z = None, spin2z = None,
                 ra = None,             # generated randomly (uniform in [0, 2pi])
                 dec = None,            # generated randomly (uniform in sin(dec))
                 inclination = None,
                 polarisation = None,
                 distance = 400.0,      # Mpc (this gets rescaled if SNR scaling is used!!!)
                 coa_phase = None,
                 coa_time = None,
                 epoch = 0.0,           # GPS time of start of segment
                 asd_H1=None, asd_L1=None, 
                 target_snr=None,
                 return_metadata=False,
                 ):
        """
        Generate a GW signal for H1 and L1 detectors.

        Returns
        -------
        hH1, hL1 : pycbc.types.TimeSeries
            Strain projected into H1 and L1.
        """
        last_meta = None
        ntries = self.max_veto_tries if self.veto_enabled else 1

        for attempt in range(ntries):

            # Resampling policy
            resample_sky  = ("sky" in self.resample) and (attempt > 0)
            resample_pol  = ("pol" in self.resample) and (attempt > 0)
            resample_time = ("time" in self.resample) and (attempt > 0)

            # Mass
            sc = source_class if source_class is not None else self.source_class
            if sc is None:
                sc = self.rng.choice(["BBH", "BNS", "BHNS"])

            approx = self.approximants[sc]

            # Masses consistent with BBH/BNS/BHNS
            if mass1 is None or mass2 is None:
                m1_s, m2_s = self.sample_masses(sc)
                mass1 = mass1 if mass1 is not None else m1_s
                mass2 = mass2 if mass2 is not None else m2_s

            # Sky location and orientation
            if (ra is None or dec is None or inclination is None or coa_phase is None) or resample_sky:
                ra_s, dec_s, inc_s, phase_s = self.sample_sky_and_orientation()
                if resample_sky:
                    ra, dec, inclination, coa_phase = ra_s, dec_s, inc_s, phase_s

                else:
                    ra = ra if ra is not None else ra_s
                    dec = dec if dec is not None else dec_s
                    inclination = inclination if inclination is not None else inc_s
                    coa_phase = coa_phase if coa_phase is not None else phase_s

            # Polarisation
            if (polarisation is None) or resample_pol:
                polarisation = self.rng.uniform(0, np.pi)

            # Geocentric coalescence time
            if coa_time == "middle":
                coa_time = epoch + self.seglen / 2.0
            elif (coa_time is None) or resample_time:
                coa_time = self._random_coalescence_time(epoch)

            
            # Aligned spins
            if spin1z is None or spin2z is None:
                s1z, s2z = self.sample_aligned_spins(source_class=sc, approximant=approx)
                spin1z = spin1z if spin1z is not None else s1z
                spin2z = spin2z if spin2z is not None else s2z


            # Tidal lambdas (only used for BNS/BHNS approximants)
            if (lambda1 is None) or (lambda2 is None):
                lam1_s, lam2_s = self.sample_tidal_lambdas(sc)
                lam1 = lam1_s if lambda1 is None else lambda1
                lam2 = lam2_s if lambda2 is None else lambda2
            else:
                lam1, lam2 = float(lambda1), float(lambda2)

            wf_kwargs = dict(
                approximant=approx,
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                delta_t=self.dt,
                f_lower=self.f_lower,
                distance=distance,
                inclination=inclination,
                coa_phase=coa_phase,
            )

            if approx in self.tidal_approximants:
                wf_kwargs["lambda1"] = float(lam1)
                wf_kwargs["lambda2"] = float(lam2)
            

            # if approx == "IMRPhenomNSBH":
            #     mBH, mNS = mass1, mass2
            #     q = mBH / mNS
            #     if not (1.0 <= mNS <= 3.0 and 1.0 <= q <= 100.0 and 0.0 <= lam2 <= 5000.0):
            #         continue  # resample this attempt
            #     spin2z = 0.0  # enforce chi_NS = 0

            if approx == "IMRPhenomNSBH":
                mBH, mNS = mass1, mass2
                q = mBH / mNS
                if not (1.0 <= mNS <= 3.0 and 1.0 <= q <= 100.0 and 0.0 <= lam2 <= 5000.0):
                    continue  # resample this attempt
                spin2z = 0.0  # enforce chi_NS = 0
                wf_kwargs["spin2z"] = 0.0

            # hp, hc = get_td_waveform(**wf_kwargs)
            try:
                hp, hc = get_td_waveform(**wf_kwargs)
            except Exception as e:
                last_meta = {"attempt": attempt, "err": repr(e), "approximant": approx}
                continue

            # Align waveform so coa_time is maximum amplitude
            hp_arr = hp.numpy()
            hc_arr = hc.numpy()

            amp = np.sqrt(hp_arr**2 + hc_arr**2)
            ipeak = int(np.argmax(amp))

            t_peak = float(hp.start_time) + ipeak * float(hp.delta_t)
            shift = float(coa_time) - t_peak

            hp.start_time = float(hp.start_time) + shift
            hc.start_time = float(hc.start_time) + shift

            # Now project into detectors
            hH1 = self.H1.project_wave(hp, hc, ra, dec, polarisation, method='lal')
            hL1 = self.L1.project_wave(hp, hc, ra, dec, polarisation, method='lal')

            # Compute time delays and store in metadata - optional
            dt_H1 = self.H1.time_delay_from_earth_center(ra, dec, coa_time)
            dt_L1 = self.L1.time_delay_from_earth_center(ra, dec, coa_time)

            chirp_mass = ((mass1 * mass2) ** (3.0 / 5.0)) / ((mass1 + mass2) ** (1.0 / 5.0))

            meta = {}
            meta.update({
                "attempt": attempt,
                "tc_geocenter": float(coa_time),
                "dt_H1": float(dt_H1),
                "dt_L1": float(dt_L1),
                "dt_H1_minus_L1": float(dt_H1 - dt_L1),
                "ra": float(ra),
                "dec": float(dec),
                "inclination": float(inclination),
                "coa_phase": float(coa_phase),
                "polarisation": float(polarisation),
                "spin1z": float(spin1z),
                "spin2z": float(spin2z),

                "source_class": str(sc),
                "approximant": str(approx),

                "mass1": float(mass1),
                "mass2": float(mass2),
                "chirp_mass": float(chirp_mass),

                "lambda1": float(lam1) if approx in self.tidal_approximants else 0.0,
                "lambda2": float(lam2) if approx in self.tidal_approximants else 0.0,
            })

            # Trim or pad to match desired segment length
            hH1 = self._fix_length(hH1, epoch)
            hL1 = self._fix_length(hL1, epoch)

            # Distance based SNR scaling using ASD (this is different to in the glitch class!!!!)
            if self.snr_enabled:
                desired = float(target_snr/100) if target_snr is not None else self._sample_target_snr()

                # network mode
                if (asd_H1 is None) or (asd_L1 is None):
                    raise ValueError("Missing asd_H1 and asd_L1.")

                rhoH0 = self._optimal_snr_from_asd(hH1, asd_H1, self.sample_rate)
                rhoL0 = self._optimal_snr_from_asd(hL1, asd_L1, self.sample_rate)
                rho_net0 = np.sqrt(rhoH0**2 + rhoL0**2)

                if rho_net0 > 0:
                    # compute new distance that would yield desired SNR
                    distance_new = distance * (rho_net0 / desired)

                    # Rescale strain by distance ratio (since h ∝ 1/d) (note! a new waveform is NOT generated - computationally expensive - but rescaling is the same thing)
                    amp_scale = distance / distance_new   # = desired / rho_net
                    hH1 = TimeSeries(hH1.numpy() * amp_scale, delta_t=self.dt, epoch=hH1.start_time)
                    hL1 = TimeSeries(hL1.numpy() * amp_scale, delta_t=self.dt, epoch=hL1.start_time)

                    # post-scale single-detector SNRs (to check veto later)
                    rhoH = float(rhoH0 * amp_scale)
                    rhoL = float(rhoL0 * amp_scale)
                    rho_net = float(np.sqrt(rhoH**2 + rhoL**2))

                    # store metadata
                    meta.update({
                        "target_snr": desired,
                        "rhoH_before": float(rhoH0),
                        "rhoL_before": float(rhoL0),
                        "rho_net_before": float(rho_net0),
                        "rhoH": rhoH, 
                        "rhoL": rhoL, 
                        "rho_net": rho_net,
                        "distance_before": float(distance),
                        "distance_after": float(distance_new),
                    })
                else:
                    meta.update({
                        "target_snr": desired,
                        "rhoH": float(rhoH0), 
                        "rhoL": float(rhoL0), 
                        "rho_net": float(rho_net0),
                        "distance_before": float(distance),
                        "distance_after": float(distance),
                    })


            if self.veto_enabled:
                # If SNR scaling is on, use meta["rhoH"], meta["rhoL"].
                # If scaling is off, compute directly (optional).
                if self.snr_enabled:
                    det_max = max(meta.get("rhoH", 0.0), meta.get("rhoL", 0.0))
                else:
                    if (asd_H1 is None) or (asd_L1 is None):
                        raise ValueError("Veto requires asd_H1/asd_L1 even if snr_enabled=False.")
                    rhoH = self._optimal_snr_from_asd(hH1, asd_H1, self.sample_rate)
                    rhoL = self._optimal_snr_from_asd(hL1, asd_L1, self.sample_rate)
                    det_max = max(float(rhoH), float(rhoL))
                    meta.update({"rhoH": float(rhoH), "rhoL": float(rhoL)})

                meta["det_snr_max"] = float(det_max)
                meta["min_det_snr"] = float(self.min_det_snr)

                if det_max < self.min_det_snr:
                    last_meta = meta
                    continue  # resample & try again

            if return_metadata:

                hH1 = pycbc_to_gwpy(hH1)
                hL1 = pycbc_to_gwpy(hL1)
                return hH1, hL1, meta

            hH1 = pycbc_to_gwpy(hH1)
            hL1 = pycbc_to_gwpy(hL1)
            return hH1, hL1

        raise RuntimeError(
            f"Failed to generate an injection with max(det SNR) >= {self.min_det_snr} "
            f"after {ntries} tries. Last attempt meta: {last_meta}")


    #-------------------------------------------------------------
    # Internal methodsssss
    #-------------------------------------------------------------
    
    def sample_aligned_spins(self, chi_max=0.99, source_class=None, approximant=None):
        # Default behaviour (BBH/BNS etc.)
        if approximant != "IMRPhenomNSBH":
            return (self.rng.uniform(-chi_max, chi_max),
                    self.rng.uniform(-chi_max, chi_max))

        # NSBH: single-spin model; enforce chi_NS = 0 and keep BH spin in calibrated-ish range
        chi_bh = self.rng.uniform(-0.5, 0.75)
        chi_ns = 0.0
        return chi_bh, chi_ns

    
    def sample_sky_and_orientation(self):
        ra = self.rng.uniform(0, 2*np.pi)
        u = self.rng.uniform(-1, 1)
        dec = np.arcsin(u)
        cosi = self.rng.uniform(-1, 1)
        inclination = np.arccos(cosi)
        coa_phase = self.rng.uniform(0, 2*np.pi)
        
        return ra, dec, inclination, coa_phase


    def _random_coalescence_time(self, epoch, margin=0.1):
        """
        Draw a random geocentric coalescence time.
        """
        tmin = epoch - margin * self.seglen          # I feel like there's no point in having the coalescence time before start???
        tmax = epoch + (1.0 + margin) * self.seglen

        # tmin = epoch + margin * self.seglen    
        # tmax = epoch + self.seglen
        return self.rng.uniform(tmin, tmax)


    @staticmethod
    def _optimal_snr_from_asd(ts, asd, fs):
        """
        ts: pycbc TimeSeries (or array-like)
        asd: ASD on rfft bins for this segment length
        fs: sample rate
        """
        x = ts.numpy() if hasattr(ts, "numpy") else np.asarray(ts)
        n = len(x)
        dt = 1.0 / float(fs)

        X = np.fft.rfft(x) * dt
        df = fs / n

        asd = np.asarray(asd)
        if len(asd) != len(X):
            raise ValueError(
                f"ASD length mismatch: len(asd)={len(asd)} vs len(rfft)={len(X)}. "
                "Interpolate ASD to rfft bins."
            )

        psd = np.maximum(asd**2, 1e-40)
        rho2 = 4.0 * np.sum((np.abs(X)**2) / psd) * df
        return np.sqrt(rho2)


    def _sample_target_snr(self):
        if self.snr_dist == "uniform":
            return self.rng.uniform(self.snr_min, self.snr_max) / 100
        if self.snr_dist == "loguniform":
            return 10 ** self.rng.uniform(np.log10(self.snr_min), np.log10(self.snr_max)) / 100
        raise ValueError("snr_dist must be 'uniform' or 'loguniform'")


    def _fix_length(self, ts, epoch, taper_injection_edges=True, fade_s=0.03, eps=1e-12):
        """
        Return a seglen-long TimeSeries starting at `epoch`.
        Any part of `ts` outside [epoch, epoch+seglen) is discarded.
        Any missing part inside the window is zero-padded.

        If taper_injection_edges=True, apply a short cosine-squared taper ONLY
        at the boundaries where the (pasted) waveform meets zero padding.
        This prevents step discontinuities (0->waveform, waveform->0) that
        produce broadband impulses after whitening / in q-transforms.

        Parameters
        ----------
        taper_injection_edges : bool
            If True, taper the nonzero-region edges after padding.
        fade_s : float
            Fade duration (seconds) at each edge of the nonzero region.
            Typical: 0.02–0.05 s.
        eps : float
            Threshold for defining "nonzero" samples.
        """
        out = np.zeros(self.n, dtype=np.float64)

        fs = float(self.sample_rate)
        dt = float(self.dt)

        # Desired output window
        t0 = float(epoch)
        t1 = float(epoch + self.seglen)

        # Available waveform window
        w0 = float(ts.start_time)
        w1 = float(ts.end_time)

        # Overlap in time
        a0 = max(t0, w0)
        a1 = min(t1, w1)

        if a1 > a0:
            # Convert overlap boundaries to sample indices
            out_i0 = int(np.round((a0 - t0) * fs))
            out_i1 = int(np.round((a1 - t0) * fs))

            ts_i0 = int(np.round((a0 - w0) * fs))
            ts_i1 = ts_i0 + (out_i1 - out_i0)

            x = ts.numpy()

            # Clamp indices just in case of rounding edge effects
            out_i0 = max(0, min(out_i0, self.n))
            out_i1 = max(0, min(out_i1, self.n))
            ts_i0 = max(0, min(ts_i0, len(x)))
            ts_i1 = max(0, min(ts_i1, len(x)))

            ncopy = min(out_i1 - out_i0, ts_i1 - ts_i0)
            if ncopy > 0:
                out[out_i0:out_i0 + ncopy] = x[ts_i0:ts_i0 + ncopy]

        # --- Injection-edge taper (only where waveform meets zero padding) ---
        if taper_injection_edges:
            nz = np.where(np.abs(out) > eps)[0]
            if nz.size > 0:
                i0 = int(nz[0])
                i1 = int(nz[-1]) + 1  # make i1 exclusive

                # fade length in samples (ensure it fits inside [i0, i1))
                nfade = int(round(fade_s * fs))
                nfade = max(1, nfade)
                # can't fade longer than half the nonzero region
                nfade = min(nfade, max(1, (i1 - i0) // 2))

                # Only taper if there is room to do so meaningfully
                if nfade >= 1 and (i1 - i0) >= 2:
                    # cosine-squared fade: 0->1 and 1->0 over nfade samples
                    t = np.linspace(0.0, np.pi / 2.0, nfade, endpoint=False)
                    fade = np.sin(t) ** 2  # starts at 0, approaches 1

                    # Apply fade-in
                    out[i0:i0 + nfade] *= fade

                    # Apply fade-out
                    out[i1 - nfade:i1] *= fade[::-1]

        return TimeSeries(out, delta_t=dt, epoch=epoch)

    def sample_masses(self, source_class):
        cfg = self.mass_ranges[source_class]

        if source_class == "BBH":
            m1 = self.rng.uniform(*cfg["m1"])
            m2 = self.rng.uniform(*cfg["m2"])
            if m2 > m1:
                m1, m2 = m2, m1
            return m1, m2

        if source_class == "BNS":
            m1 = self.rng.uniform(*cfg["m1"])
            m2 = self.rng.uniform(*cfg["m2"])
            if m2 > m1:
                m1, m2 = m2, m1
            return m1, m2

        if source_class == "BHNS":
            mbh = self.rng.uniform(*cfg["mbh"])
            mns = self.rng.uniform(*cfg["mns"])
            # enforce mass1=BH, mass2=NS so lambda1=0, lambda2>0 lines up
            return mbh, mns

        raise ValueError(f"Unknown source_class: {source_class}")


    def sample_tidal_lambdas(self, source_class):
        # default: no tides
        lam1, lam2 = 0.0, 0.0

        if not self.tidal_enabled:
            return lam1, lam2

        if source_class == "BNS":
            return (
                self.rng.uniform(0.0, self.lambda_max),
                self.rng.uniform(0.0, self.lambda_max),
            )

        if source_class == "BHNS":
            return (0.0, self.rng.uniform(0.0, self.lambda_max))

        return lam1, lam2  # BBH
    




# %%
# GLITCH GENERATION

class GlitchGenerator:
    """
    Generate time-domain instrumental glitches for two LIGO detectors (H1, L1).

    The glitches are synthetic, research-inspired approximations of common
    glitch morphologies used in the literature for simulation and machine
    learning studies (Gaussian, sine-Gaussian, ringdown, chirp-like, noise
    bursts, and scattered-light-like).

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz (e.g. 4096.0).
    seglen : float
        Segment length in seconds.
    seed : int, optional
        Random seed for reproducibility.
    glitch_types : list of str, optional
        Subset of glitch families to use. If None, defaults to
        ['gaussian', 'sine_gaussian', 'ringdown',
         'chirp', 'noise_burst', 'scattered'].

    Notes
    -----
    Output is two pycbc.types.TimeSeries objects with the same length and
    sampling as your NoiseGenerator and SignalGenerator classes, so you can
    directly add them:

        glitch_H1, glitch_L1 = glitch_gen.generate()
        noisy_H1 = noise_H1 + signal_H1 + glitch_H1
        noisy_L1 = noise_L1 + signal_L1 + glitch_L1
    """

    def __init__(self, 
                 sample_rate=4096.0, 
                 seglen=8.0,
                 seed=None,
                 glitch_types=None,
                 snr_min=5.0,
                 snr_max=50.0,
                 snr_dist="loguniform",  # "loguniform" or "uniform"
                 snr_enabled=True,
                 ):

        self.sample_rate = sample_rate
        self.seglen = seglen
        self.dt = 1.0 / sample_rate
        self.n = int(seglen * sample_rate)

        self.seed = seed if seed is not None else np.random.randint(1, 2**31 - 1)
        self.rng = np.random.default_rng(self.seed)

        # Time vector relative to the segment start
        self.time = np.linspace(0.0, self.seglen, self.n, endpoint=False)

        if glitch_types is None:
            glitch_types = [
                # 'gaussian',      # simple Gaussian bump - bad
                'sine_gaussian', # blip-like
                'ringdown',      # damped sinusoid
                'chirp',         # whistle-like
                'noise_burst',   # envelope-modulated noise
                'scattered',     # low-frequency scattered-light-like
            ]
        self.glitch_types = glitch_types

        if snr_min <= 0 or snr_max <= 0 or snr_max < snr_min:
            raise ValueError("Require snr_min > 0, snr_max > 0, and snr_max >= snr_min.")
        if snr_dist not in ("loguniform", "uniform"):
            raise ValueError("snr_dist must be 'loguniform' or 'uniform'.")

        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.snr_dist = snr_dist
        self.snr_enabled = bool(snr_enabled)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self,
                 epoch=0.0,
                 glitch_type=None,
                 asd_H1=None,
                 asd_L1=None,
                 target_snr=None,  # Optional, otherwise sampled from range
                 detector=None,
                 return_metadata=False,
                 **kwargs):
        """
        Generate a glitch time series for H1 and L1.

        Glitches ONLY ever appear in one detector at a time, with equal probability
        (50/50) of being in H1 vs L1.

        If ASD(s) are provided and snr_enabled=True, the glitch is automatically
        rescaled to a sampled target SNR (or to target_snr if you pass it).
        """
        if glitch_type is None:
            glitch_type = self.rng.choice(self.glitch_types)
        glitch_type = glitch_type.lower()

        # Always generate exactly one glitch realisation
        base = self._generate_single(glitch_type, **kwargs)

        # Decide which detector gets it
        if detector == "H1":
            glitch_in_H1 = True
        elif detector == "L1":
            glitch_in_H1 = False
        else:
            glitch_in_H1 = (self.rng.random() < 0.5)

        base_H1 = base if glitch_in_H1 else np.zeros_like(base)
        base_L1 = np.zeros_like(base) if glitch_in_H1 else base

        meta = {
            "glitch_type": str(glitch_type),
            "detector": "H1" if glitch_in_H1 else "L1",
            "target_snr": None,
            "final_snr": None,
        }

        # SNR scaling
        if self.snr_enabled:
            desired_snr = float(target_snr) if target_snr is not None else self._sample_target_snr()
            meta["target_snr"] = float(desired_snr)

            if glitch_in_H1:
                if asd_H1 is not None:
                    rho0 = self._optimal_snr_from_asd(base_H1, asd_H1, self.sample_rate)
                    if rho0 > 0:
                        scale = desired_snr / rho0
                        base_H1 = base_H1 * scale
                        meta["final_snr"] = float(rho0 * scale)
                    else:
                        meta["final_snr"] = 0.0
            else:
                if asd_L1 is not None:
                    rho0 = self._optimal_snr_from_asd(base_L1, asd_L1, self.sample_rate)
                    if rho0 > 0:
                        scale = desired_snr / rho0
                        base_L1 = base_L1 * scale
                        meta["final_snr"] = float(rho0 * scale)
                    else:
                        meta["final_snr"] = 0.0

        glitch_H1 = TimeSeries(base_H1, delta_t=self.dt, epoch=epoch)
        glitch_L1 = TimeSeries(base_L1, delta_t=self.dt, epoch=epoch)

        glitch_H1 = pycbc_to_gwpy(glitch_H1)
        glitch_L1 = pycbc_to_gwpy(glitch_L1)

        if return_metadata:
            return glitch_H1, glitch_L1, meta

        return glitch_H1, glitch_L1

    # ------------------------------------------------------------------
    # Internal: amplitude scalingg
    # ------------------------------------------------------------------
    @staticmethod
    def _optimal_snr_from_asd(glitch_ts, asd, fs):
        """
        Compute optimal SNR sqrt(4 ∫ |g(f)|^2 / S_n(f) df) using ASD.
        glitch_ts: 1D numpy array (time series)
        asd: ASD evaluated on rfft frequency bins (same length as rfft)
        fs: sample rate (Hz)
        """
        g = np.asarray(glitch_ts)
        n = len(g)
        G = np.fft.rfft(g)
        df = fs / n

        asd = np.asarray(asd)
        if len(asd) != len(G):
            raise ValueError(
                f"ASD length mismatch: len(asd)={len(asd)} vs len(rfft)={len(G)}. "
                "Make sure ASD is on rfft frequency bins for this segment length."
            )

        psd = np.maximum(asd**2, 1e-40)
        rho2 = 4.0 * np.sum((np.abs(G)**2) / psd) * df
        return np.sqrt(rho2)

    def _sample_target_snr(self):
        if self.snr_dist == "uniform":
            return self.rng.uniform(self.snr_min, self.snr_max)
        # log-uniform in [snr_min, snr_max]
        return 10 ** self.rng.uniform(np.log10(self.snr_min), np.log10(self.snr_max))


    # ------------------------------------------------------------------
    # Internal: generate a single underlying glitch waveform
    # ------------------------------------------------------------------
    def _generate_single(self, glitch_type, **overrides):
        t = self.time

        # Allow glitches anywhere in the segment (no edge bias)
        t0 = overrides.get('t0', self.rng.uniform(0.0, self.seglen))

        # Shared amplitude scale. RESCALE later.
        amplitude = overrides.get('amplitude', self.rng.uniform(1, 5.0))


        if glitch_type == 'gaussian':
            sigma = overrides.get('sigma', self.rng.uniform(0.02, 0.5) * self.seglen)  # 2–50% of segment
            
            g = self._gaussian(t, amplitude, t0, sigma)


        elif glitch_type == 'sine_gaussian':
            f0 = overrides.get('f0', self.rng.uniform(30.0, 400.0))  # blip-like band
            q = overrides.get('q', self.rng.uniform(3.0, 20.0))    # quality factor
            phase = overrides.get('phase', self.rng.uniform(0, 2 * np.pi))
            
            g = self._sine_gaussian(t, amplitude, t0, f0, q, phase)


        elif glitch_type == 'ringdown':
            f0 = overrides.get('f0', self.rng.uniform(60.0, 280.0))  # match plot frange
            tau = overrides.get('tau', self.rng.uniform(0.02, 0.2) * self.seglen)
            phase = overrides.get('phase', self.rng.uniform(0, 2*np.pi))
            duration = overrides.get('duration', 6.0 * tau)  # or fixed, e.g. 0.5s

            g = self._ringdown(t, amplitude, t0, f0, tau, phase, duration=duration, ramp_cycles=3)


        elif glitch_type == 'chirp':
            f_start = overrides.get('f_start', self.rng.uniform(50.0, 300.0))
            f_end = overrides.get('f_end', self.rng.uniform(f_start + 50.0, f_start + 600.0))
            duration = overrides.get('duration', self.rng.uniform(0.01, 0.3) * self.seglen)
            phase = overrides.get('phase', self.rng.uniform(0, 2 * np.pi))
            
            g = self._chirp(t, amplitude, t0, f_start, f_end, duration, phase)


        elif glitch_type == 'noise_burst':
            duration = overrides.get('duration', self.rng.uniform(0.1, 0.5) * self.seglen)
            g = self._noise_burst(t, amplitude, t0, duration)


        elif glitch_type == 'scattered':
            f0 = overrides.get('f0', self.rng.uniform(20.0, 200.0))   # low frequency
            f1 = overrides.get('f1', self.rng.uniform(f0 + 10.0, f0 + 80.0))
            duration = overrides.get('duration', self.rng.uniform(0.3, 0.9) * self.seglen)

            g = self._scattered_like(t, amplitude, t0, f0, f1, duration)


        else:
            raise ValueError(f"Unsupported glitch type: {glitch_type}")

        # Optional normalisation to unit variance before amplitude scaling
        if np.std(g) > 0:
            g = g / np.std(g) * amplitude

        return g



    # ------------------------------------------------------------------
    # Waveform primitives
    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian(t, amp, t0, sigma):
        return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


    @staticmethod
    def _sine_gaussian(t, amp, t0, f0, q, phase, n_tau=6):
        tau = q / (np.sqrt(2) * np.pi * f0)

        y = np.zeros_like(t)
        mask = np.abs(t - t0) <= (n_tau * tau)

        tt = t[mask] - t0
        envelope = np.exp(-0.5 * (tt / tau) ** 2)
        carrier = np.cos(2 * np.pi * f0 * tt + phase)
        y[mask] = amp * envelope * carrier
        return y

    @staticmethod
    def _ringdown(t, amp, t0, f0, tau, phase, duration=None, ramp_cycles=3):
        """
        Ringdown with smooth onset to avoid broadband step at t0.
        duration: if given, truncate after t0+duration
        ramp_cycles: onset ramp length in number of cycles at f0
        """
        y = np.zeros_like(t)

        if duration is None:
            # make it finite by default (otherwise it runs to end of segment)
            duration = 6.0 * tau  # ~e^-6 tail; adjust if you want

        t1 = t0
        t2 = t0 + duration
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            return y

        tt = t[mask] - t0

        envelope = np.exp(-tt / tau)
        carrier = np.cos(2 * np.pi * f0 * tt + phase)

        # Smooth onset ramp (raised cosine)
        ramp_time = max(ramp_cycles / f0, 0.0)  # seconds
        if ramp_time > 0:
            r = np.ones_like(tt)
            m = tt < ramp_time
            # goes 0 -> 1 smoothly
            r[m] = 0.5 * (1 - np.cos(np.pi * tt[m] / ramp_time))
        else:
            r = 1.0

        y[mask] = amp * r * envelope * carrier
        return y



    def _chirp(self, t, amp, t0, f_start, f_end, duration, phase):
        y = np.zeros_like(t)
        t1 = t0 - 0.5 * duration
        t2 = t0 + 0.5 * duration
        mask = (t >= t1) & (t <= t2)
        tt = t[mask] - t0

        # Linear frequency sweep
        k = (f_end - f_start) / duration
        inst_phase = 2 * np.pi * (f_start * tt + 0.5 * k * tt**2) + phase

        # Smooth Gaussian envelope across the duration
        sigma = duration / 4.0
        envelope = np.exp(-0.5 * (tt / sigma) ** 2)
        y[mask] = amp * envelope * np.cos(inst_phase)
        return y

    def _noise_burst(self, t, amp, t0, duration, f0=None, bandwidth=None):

        y = np.zeros_like(t)

        # Time window
        t1 = t0 - 0.5 * duration
        t2 = t0 + 0.5 * duration
        mask = (t >= t1) & (t <= t2)

        if not np.any(mask):
            return y

        tt = t[mask] - t0

        # Gaussian envelope
        sigma = duration / 4.0
        envelope = np.exp(-0.5 * (tt / sigma) ** 2)

        # Choose band parameters if not provided
        if f0 is None:
            f0 = self.rng.uniform(40.0, 300.0)

        if bandwidth is None:
            bandwidth = self.rng.uniform(30.0, 250.0)

        # Generate white noise
        n = mask.sum()
        noise = self.rng.normal(0.0, 1.0, size=n)

        # Band-limit in frequency domain (no wrap — local segment only)
        freqs = np.fft.rfftfreq(n, d=self.dt)
        spectrum = np.fft.rfft(noise)

        band = (freqs >= (f0 - bandwidth / 2.0)) & \
            (freqs <= (f0 + bandwidth / 2.0))

        spectrum[~band] = 0.0

        noise_band = np.fft.irfft(spectrum, n=n)

        # Normalize carrier to unit std so amp behaves consistently
        if np.std(noise_band) > 0:
            noise_band = noise_band / np.std(noise_band)

        y[mask] = amp * envelope * noise_band

        return y


    def _scattered_like(self, t, amp, t0, f0, f1, duration):
        """
        Very simplified scattered-light-like glitch: a long, low-frequency
        sinusoid with a slow frequency sweep and smooth envelope.
        """
        y = np.zeros_like(t)
        t1 = t0 - 0.5 * duration
        t2 = t0 + 0.5 * duration
        mask = (t >= t1) & (t <= t2)
        tt = t[mask] - t0

        # Slow chirp
        k = (f1 - f0) / duration
        inst_phase = 2 * np.pi * (f0 * tt + 0.5 * k * tt**2)

        # Broad envelope (almost flat in the middle)
        sigma = duration / 2.5
        envelope = np.exp(-0.5 * (tt / sigma) ** 2)
        sig = np.cos(inst_phase) + 0.5*np.cos(2*inst_phase) + 0.25*np.cos(3*inst_phase) #harmonicssss

        y[mask] = amp * envelope * sig #np.cos(inst_phase)
        return y


# %%
# ============================================
# Converter
# ============================================

def pycbc_to_gwpy(pycbc_ts, meta=None):
    ts = GwpyTimeSeries(
        pycbc_ts.numpy(),
        t0=float(pycbc_ts.start_time),
        dt=float(pycbc_ts.delta_t),
        unit=None,
    )
    return ts
