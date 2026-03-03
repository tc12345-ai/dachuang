"""
Spectrum Analysis Engine — 频谱分析引擎

Supports FFT, STFT, PSD estimation, and window function management.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class SpectrumResult:
    """Spectrum analysis result / 频谱分析结果."""
    # FFT results
    freq_hz: np.ndarray = None
    magnitude: np.ndarray = None        # Linear magnitude
    magnitude_db: np.ndarray = None     # dB magnitude
    phase_rad: np.ndarray = None        # Phase (radians)
    phase_deg: np.ndarray = None        # Phase (degrees)
    power: np.ndarray = None            # Power spectrum
    
    # PSD results
    psd_freq: np.ndarray = None
    psd: np.ndarray = None              # Power spectral density
    psd_db: np.ndarray = None           # PSD in dB/Hz
    
    # STFT results
    stft_times: np.ndarray = None
    stft_freqs: np.ndarray = None
    stft_magnitude: np.ndarray = None   # |STFT| matrix
    stft_magnitude_db: np.ndarray = None
    stft_phase: np.ndarray = None
    
    # Cepstrum
    cepstrum: np.ndarray = None
    quefrency: np.ndarray = None
    
    # Info
    fs: float = 1.0
    nfft: int = 0
    window_name: str = ''
    info: str = ''


class SpectrumAnalyzer:
    """
    Signal Spectrum Analyzer — 信号频谱分析器
    
    Provides FFT, STFT, PSD, and cepstrum analysis.
    """
    
    WINDOWS = {
        'Hann': 'hann',
        'Hamming': 'hamming',
        'Blackman': 'blackman',
        'Blackman-Harris': 'blackmanharris',
        'Kaiser (β=8)': ('kaiser', 8.0),
        'Kaiser (β=14)': ('kaiser', 14.0),
        'Kaiser (β=20)': ('kaiser', 20.0),
        'Rectangular': 'boxcar',
        'Bartlett': 'bartlett',
        'Flattop': 'flattop',
        'Nuttall': 'nuttall',
        'Tukey (α=0.5)': ('tukey', 0.5),
    }
    
    def __init__(self):
        pass
    
    def compute_fft(self, data: np.ndarray, fs: float,
                    nfft: int = None, window: str = 'Hann',
                    detrend: bool = False) -> SpectrumResult:
        """
        Compute FFT of signal.
        
        Args:
            data: Input signal (1D array)
            fs: Sampling rate (Hz)
            nfft: FFT length (None = len(data), or next power of 2)
            window: Window function name (key in WINDOWS dict)
            detrend: Remove DC offset before FFT
        Returns:
            SpectrumResult with frequency, magnitude, phase data
        """
        result = SpectrumResult(fs=fs)
        
        x = np.asarray(data, dtype=np.float64).ravel()
        N = len(x)
        
        if detrend:
            x = x - np.mean(x)
        
        if nfft is None:
            nfft = N
        nfft = max(nfft, N)
        result.nfft = nfft
        
        # Apply window
        win = self._get_window(window, N)
        result.window_name = window
        x_win = x * win
        
        # Window correction factor
        win_sum = np.sum(win)
        win_ss = np.sum(win ** 2)
        
        # FFT
        X = fft(x_win, n=nfft)
        
        # One-sided spectrum
        n_onesided = nfft // 2 + 1
        X_os = X[:n_onesided]
        
        # Frequency vector
        result.freq_hz = np.linspace(0, fs / 2, n_onesided)
        
        # Magnitude (corrected for window)
        result.magnitude = np.abs(X_os) * 2.0 / win_sum
        result.magnitude[0] /= 2.0  # DC component
        if n_onesided > 1:
            result.magnitude[-1] /= 2.0  # Nyquist
        
        result.magnitude_db = 20.0 * np.log10(np.maximum(result.magnitude, 1e-12))
        
        # Phase
        result.phase_rad = np.angle(X_os)
        result.phase_deg = np.degrees(result.phase_rad)
        
        # Power spectrum
        result.power = (np.abs(X_os) ** 2) / (win_ss * fs)
        result.power[0] /= 2.0
        if n_onesided > 1:
            result.power[-1] /= 2.0
        
        result.info = f"FFT: N={N}, NFFT={nfft}, Window={window}"
        return result
    
    def compute_psd(self, data: np.ndarray, fs: float,
                    nfft: int = 1024, window: str = 'Hann',
                    nperseg: int = None, noverlap: int = None,
                    detrend: str = 'constant') -> SpectrumResult:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            data: Input signal
            fs: Sampling rate (Hz)
            nfft: FFT length
            window: Window function name
            nperseg: Segment length (None = nfft)
            noverlap: Overlap samples (None = nperseg // 2)
            detrend: 'constant' (remove mean), 'linear', or False
        Returns:
            SpectrumResult with PSD data
        """
        result = SpectrumResult(fs=fs)
        
        x = np.asarray(data, dtype=np.float64).ravel()
        
        if nperseg is None:
            nperseg = min(nfft, len(x))
        if noverlap is None:
            noverlap = nperseg // 2
        
        win_name = self._get_scipy_window_name(window)
        
        f, Pxx = sig.welch(x, fs=fs, window=win_name, nperseg=nperseg,
                           noverlap=noverlap, nfft=nfft, detrend=detrend,
                           scaling='density')
        
        result.psd_freq = f
        result.psd = Pxx
        result.psd_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))
        result.nfft = nfft
        result.window_name = window
        result.info = f"PSD (Welch): nperseg={nperseg}, noverlap={noverlap}, NFFT={nfft}"
        
        return result
    
    def compute_stft(self, data: np.ndarray, fs: float,
                     nperseg: int = 256, noverlap: int = None,
                     nfft: int = None, window: str = 'Hann') -> SpectrumResult:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            data: Input signal
            fs: Sampling rate (Hz)
            nperseg: Segment length
            noverlap: Overlap (None = nperseg * 3 // 4)
            nfft: FFT length (None = nperseg)
            window: Window function name
        Returns:
            SpectrumResult with STFT spectrogram data
        """
        result = SpectrumResult(fs=fs)
        
        x = np.asarray(data, dtype=np.float64).ravel()
        
        if noverlap is None:
            noverlap = nperseg * 3 // 4
        if nfft is None:
            nfft = nperseg
        
        win_name = self._get_scipy_window_name(window)
        
        f, t, Zxx = sig.stft(x, fs=fs, window=win_name, nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft)
        
        result.stft_freqs = f
        result.stft_times = t
        result.stft_magnitude = np.abs(Zxx)
        result.stft_magnitude_db = 20.0 * np.log10(
            np.maximum(np.abs(Zxx), 1e-12))
        result.stft_phase = np.angle(Zxx)
        result.nfft = nfft
        result.window_name = window
        result.info = f"STFT: nperseg={nperseg}, noverlap={noverlap}, NFFT={nfft}"
        
        return result
    
    def compute_cepstrum(self, data: np.ndarray, fs: float) -> SpectrumResult:
        """
        Compute real cepstrum.
        
        Args:
            data: Input signal
            fs: Sampling rate (Hz)
        Returns:
            SpectrumResult with cepstrum data
        """
        result = SpectrumResult(fs=fs)
        
        x = np.asarray(data, dtype=np.float64).ravel()
        N = len(x)
        
        X = fft(x)
        log_X = np.log(np.maximum(np.abs(X), 1e-20))
        cepstrum = np.real(np.fft.ifft(log_X))
        
        result.cepstrum = cepstrum[:N // 2]
        result.quefrency = np.arange(N // 2) / fs
        result.info = f"Real Cepstrum, N={N}"
        
        return result
    
    def get_window_info(self, window_name: str, length: int = 256,
                        fs: float = 1.0) -> dict:
        """
        Get window function details for preview.
        
        Returns dict with:
            'samples': window samples
            'freq': frequency vector
            'magnitude_db': frequency response in dB
            'mainlobe_width': mainlobe width in bins
            'sidelobe_level': peak sidelobe level in dB
        """
        win = self._get_window(window_name, length)
        
        # Frequency response of window
        nfft = 4096
        W = fft(win, n=nfft)
        W_half = np.abs(W[:nfft // 2 + 1])
        W_half = W_half / np.max(W_half)
        W_db = 20.0 * np.log10(np.maximum(W_half, 1e-12))
        freq = np.linspace(0, 0.5, nfft // 2 + 1)
        
        # Mainlobe width (3dB)
        mainlobe_mask = W_db > -3.0
        mainlobe_width = np.sum(mainlobe_mask) * (0.5 / (nfft // 2 + 1))
        
        # Sidelobe level
        try:
            first_null = np.where(np.diff(mainlobe_mask.astype(int)) == -1)[0]
            if len(first_null) > 0:
                sidelobe_level = np.max(W_db[first_null[0]:])
            else:
                sidelobe_level = -100.0
        except Exception:
            sidelobe_level = -100.0
        
        return {
            'samples': win,
            'freq': freq,
            'magnitude_db': W_db,
            'mainlobe_width': mainlobe_width,
            'sidelobe_level': sidelobe_level,
        }
    
    def _get_window(self, window_name: str, length: int) -> np.ndarray:
        """Get window samples by name."""
        if window_name in self.WINDOWS:
            w = self.WINDOWS[window_name]
        else:
            w = 'hann'
        
        if isinstance(w, tuple):
            return sig.get_window(w, length)
        else:
            return sig.get_window(w, length)
    
    def _get_scipy_window_name(self, window_name: str):
        """Convert display name to scipy window parameter."""
        if window_name in self.WINDOWS:
            w = self.WINDOWS[window_name]
            return w
        return 'hann'
