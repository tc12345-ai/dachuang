"""
Signal Measurements — 频率测量模块

THD, SNR, SFDR, harmonic analysis, peak detection, in-band power.
"""

import numpy as np
from scipy.fft import fft
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MeasurementResult:
    """Signal measurement results / 信号测量结果."""
    # Peak frequency
    fundamental_freq: float = 0.0
    fundamental_mag_db: float = -np.inf
    
    # Harmonics
    harmonic_freqs: List[float] = field(default_factory=list)
    harmonic_mags_db: List[float] = field(default_factory=list)
    
    # Distortion metrics
    thd_percent: float = 0.0
    thd_db: float = -np.inf
    
    # Noise metrics
    snr_db: float = 0.0
    sfdr_db: float = 0.0
    sinad_db: float = 0.0
    enob: float = 0.0                   # Effective number of bits
    
    # Power
    total_power_db: float = -np.inf
    signal_power_db: float = -np.inf
    noise_power_db: float = -np.inf
    inband_power_db: float = -np.inf
    noise_floor_db: float = -np.inf
    
    # Info
    info: str = ''


class SignalMeasurements:
    """
    Signal Measurements — 信号测量工具
    
    Provides comprehensive frequency-domain measurements.
    """
    
    def __init__(self, n_harmonics: int = 10):
        """
        Args:
            n_harmonics: Number of harmonics to analyze
        """
        self.n_harmonics = n_harmonics
    
    def analyze(self, data: np.ndarray, fs: float,
                nfft: int = None, window: str = 'hann',
                fundamental_hint: float = None,
                n_harmonics: int = None) -> MeasurementResult:
        """
        Perform comprehensive signal measurements.
        
        Args:
            data: Input signal
            fs: Sampling rate (Hz)
            nfft: FFT length
            window: Window function
            fundamental_hint: Expected fundamental frequency (Hz), None=auto
            n_harmonics: Override number of harmonics
        Returns:
            MeasurementResult with all measurements
        """
        result = MeasurementResult()
        
        x = np.asarray(data, dtype=np.float64).ravel()
        N = len(x)
        
        if nfft is None:
            nfft = N
        if n_harmonics is None:
            n_harmonics = self.n_harmonics
        
        # Remove DC
        x = x - np.mean(x)
        
        # Apply window
        from scipy.signal import get_window
        win = get_window(window, N)
        x_win = x * win
        
        # Coherent gain correction
        S1 = np.sum(win)
        S2 = np.sum(win ** 2)
        ENBW = N * S2 / (S1 ** 2)  # Equivalent noise bandwidth
        
        # FFT
        X = fft(x_win, n=nfft)
        n_half = nfft // 2 + 1
        X_half = X[:n_half]
        
        # Power spectrum (V^2)
        P = (np.abs(X_half) ** 2) / (S1 ** 2)
        P[1:-1] *= 2.0  # Double non-DC, non-Nyquist
        
        freq = np.linspace(0, fs / 2, n_half)
        df = fs / nfft  # Frequency resolution
        
        # Find fundamental
        # Skip DC bin (first few bins)
        dc_skip = max(1, int(20 / df))  # Skip below 20 Hz
        
        if fundamental_hint is not None:
            fund_bin = int(round(fundamental_hint / df))
            # Search around hint
            search_width = max(3, int(50 / df))
            lo = max(dc_skip, fund_bin - search_width)
            hi = min(n_half - 1, fund_bin + search_width)
            fund_bin = lo + np.argmax(P[lo:hi + 1])
        else:
            fund_bin = dc_skip + np.argmax(P[dc_skip:n_half])
        
        result.fundamental_freq = freq[fund_bin]
        result.fundamental_mag_db = 10.0 * np.log10(max(P[fund_bin], 1e-20))
        
        # Harmonic analysis
        harmonic_bins = []
        harmonic_powers = []
        bin_width = max(1, int(3 * ENBW))  # Bins to include around each harmonic
        
        for h in range(1, n_harmonics + 1):
            h_freq = result.fundamental_freq * h
            if h_freq >= fs / 2:
                break
            h_bin = int(round(h_freq / df))
            if h_bin >= n_half:
                break
            
            # Sum power in bins around harmonic
            lo = max(0, h_bin - bin_width)
            hi = min(n_half - 1, h_bin + bin_width)
            actual_bin = lo + np.argmax(P[lo:hi + 1])
            h_power = P[actual_bin]
            
            harmonic_bins.append(actual_bin)
            harmonic_powers.append(h_power)
            result.harmonic_freqs.append(freq[actual_bin])
            result.harmonic_mags_db.append(
                10.0 * np.log10(max(h_power, 1e-20)))
        
        # Signal power (fundamental)
        signal_power = harmonic_powers[0] if harmonic_powers else P[fund_bin]
        
        # Harmonic distortion power (exclude fundamental)
        harmonic_power = sum(harmonic_powers[1:]) if len(harmonic_powers) > 1 else 0.0
        
        # THD
        if signal_power > 0:
            thd_ratio = np.sqrt(harmonic_power / signal_power)
            result.thd_percent = thd_ratio * 100.0
            result.thd_db = 20.0 * np.log10(max(thd_ratio, 1e-12))
        
        # Total power
        total_power = np.sum(P[dc_skip:])
        result.total_power_db = 10.0 * np.log10(max(total_power, 1e-20))
        result.signal_power_db = 10.0 * np.log10(max(signal_power, 1e-20))
        
        # Noise power = total - signal - harmonics
        all_harmonic_power = sum(harmonic_powers)
        noise_power = max(total_power - all_harmonic_power, 1e-20)
        result.noise_power_db = 10.0 * np.log10(noise_power)
        
        # SNR
        result.snr_db = 10.0 * np.log10(max(signal_power / noise_power, 1e-12))
        
        # SINAD
        sinad_noise = noise_power + harmonic_power
        result.sinad_db = 10.0 * np.log10(
            max(signal_power / sinad_noise, 1e-12))
        
        # ENOB
        result.enob = (result.sinad_db - 1.76) / 6.02
        
        # SFDR
        # Find largest spurious component (any non-fundamental peak)
        P_copy = P.copy()
        # Zero out fundamental region
        if harmonic_bins:
            lo = max(0, harmonic_bins[0] - bin_width)
            hi = min(n_half - 1, harmonic_bins[0] + bin_width)
            P_copy[lo:hi + 1] = 0
        P_copy[:dc_skip] = 0
        
        max_spur = np.max(P_copy) if len(P_copy) > 0 else 1e-20
        result.sfdr_db = 10.0 * np.log10(
            max(signal_power / max(max_spur, 1e-20), 1e-12))
        
        # Noise floor estimate (median of spectrum)
        P_sorted = np.sort(P[dc_skip:])
        noise_floor = np.median(P_sorted[:len(P_sorted) // 2])
        result.noise_floor_db = 10.0 * np.log10(max(noise_floor, 1e-20))
        
        result.info = (f"Fundamental={result.fundamental_freq:.1f}Hz, "
                       f"THD={result.thd_percent:.3f}%, "
                       f"SNR={result.snr_db:.1f}dB, "
                       f"SFDR={result.sfdr_db:.1f}dB")
        
        return result
    
    def measure_inband_power(self, data: np.ndarray, fs: float,
                             f_low: float, f_high: float,
                             nfft: int = None,
                             window: str = 'hann') -> float:
        """
        Measure power in a specific frequency band.
        
        Args:
            data: Input signal
            fs: Sampling rate
            f_low: Lower band edge (Hz)
            f_high: Upper band edge (Hz)
            nfft: FFT length
            window: Window function
        Returns:
            In-band power in dB
        """
        x = np.asarray(data, dtype=np.float64).ravel()
        N = len(x)
        if nfft is None:
            nfft = N
        
        from scipy.signal import get_window
        win = get_window(window, N)
        X = fft(x * win, n=nfft)
        
        n_half = nfft // 2 + 1
        freq = np.linspace(0, fs / 2, n_half)
        P = np.abs(X[:n_half]) ** 2 / (np.sum(win) ** 2)
        
        mask = (freq >= f_low) & (freq <= f_high)
        inband = np.sum(P[mask])
        
        return 10.0 * np.log10(max(inband, 1e-20))
    
    def find_peaks(self, magnitude_db: np.ndarray, freq_hz: np.ndarray,
                   threshold_db: float = -60.0,
                   min_distance_hz: float = 10.0,
                   max_peaks: int = 20) -> List[Tuple[float, float]]:
        """
        Find spectral peaks.
        
        Args:
            magnitude_db: Magnitude spectrum in dB
            freq_hz: Frequency vector
            threshold_db: Minimum peak level
            min_distance_hz: Minimum distance between peaks
            max_peaks: Maximum peaks to return
        Returns:
            List of (frequency_hz, magnitude_db) tuples, sorted by magnitude
        """
        from scipy.signal import find_peaks as sp_find_peaks
        
        df = freq_hz[1] - freq_hz[0] if len(freq_hz) > 1 else 1.0
        min_distance = max(1, int(min_distance_hz / df))
        
        peak_indices, properties = sp_find_peaks(
            magnitude_db, height=threshold_db, distance=min_distance)
        
        if len(peak_indices) == 0:
            return []
        
        # Sort by magnitude (descending)
        sorted_idx = np.argsort(magnitude_db[peak_indices])[::-1]
        peak_indices = peak_indices[sorted_idx[:max_peaks]]
        
        peaks = [(float(freq_hz[i]), float(magnitude_db[i]))
                 for i in peak_indices]
        
        return peaks
