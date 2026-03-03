"""
Utility helper functions for DSP Platform.
公用工具函数模块
"""

import numpy as np


def db_to_linear(db_value):
    """Convert dB to linear scale."""
    return 10 ** (db_value / 20.0)


def linear_to_db(linear_value):
    """Convert linear scale to dB."""
    return 20.0 * np.log10(np.maximum(linear_value, 1e-12))


def power_to_db(power_value):
    """Convert power to dB."""
    return 10.0 * np.log10(np.maximum(power_value, 1e-12))


def db_to_power(db_value):
    """Convert dB to power."""
    return 10 ** (db_value / 10.0)


def normalize_frequency(freq_hz, fs):
    """
    Normalize frequency to Nyquist (0-1 range for scipy).
    
    Args:
        freq_hz: Frequency in Hz
        fs: Sampling rate in Hz
    Returns:
        Normalized frequency (0 to 1, where 1 = Nyquist = fs/2)
    """
    return freq_hz / (fs / 2.0)


def hz_to_rad(freq_hz, fs):
    """Convert Hz to radians/sample."""
    return 2.0 * np.pi * freq_hz / fs


def generate_test_signal(signal_type, duration, fs, freq=1000.0, amplitude=1.0,
                         noise_level=0.0, harmonics=None):
    """
    Generate test signals for analysis.
    
    Args:
        signal_type: 'sine', 'square', 'sawtooth', 'chirp', 'impulse', 'step', 'noise'
        duration: Signal duration in seconds
        fs: Sampling rate in Hz
        freq: Fundamental frequency in Hz
        amplitude: Signal amplitude
        noise_level: Noise standard deviation (0 = no noise)
        harmonics: List of (harmonic_number, relative_amplitude) for multi-tone
    Returns:
        t: Time vector
        signal: Generated signal
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    if signal_type == 'sine':
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        if harmonics:
            for h_num, h_amp in harmonics:
                signal += amplitude * h_amp * np.sin(2 * np.pi * freq * h_num * t)
    elif signal_type == 'square':
        from scipy import signal as sig
        signal = amplitude * sig.square(2 * np.pi * freq * t)
    elif signal_type == 'sawtooth':
        from scipy import signal as sig
        signal = amplitude * sig.sawtooth(2 * np.pi * freq * t)
    elif signal_type == 'chirp':
        from scipy import signal as sig
        f1 = freq / 10.0
        signal = amplitude * sig.chirp(t, f0=f1, f1=freq, t1=duration)
    elif signal_type == 'impulse':
        signal = np.zeros(n_samples)
        signal[0] = amplitude
    elif signal_type == 'step':
        signal = np.zeros(n_samples)
        signal[n_samples // 4:] = amplitude
    elif signal_type == 'noise':
        signal = amplitude * np.random.randn(n_samples)
    elif signal_type == 'multi_tone':
        signal = np.zeros(n_samples)
        if harmonics:
            for h_num, h_amp in harmonics:
                signal += amplitude * h_amp * np.sin(2 * np.pi * freq * h_num * t)
        else:
            signal = amplitude * np.sin(2 * np.pi * freq * t)
    else:
        signal = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(n_samples)
    
    return t, signal


def find_nearest_index(array, value):
    """Find index of nearest value in array."""
    return int(np.argmin(np.abs(np.asarray(array) - value)))


def next_power_of_2(n):
    """Return next power of 2 >= n."""
    return 1 << (int(n) - 1).bit_length()


def format_freq(freq_hz):
    """Format frequency for display."""
    if abs(freq_hz) >= 1e6:
        return f"{freq_hz/1e6:.3f} MHz"
    elif abs(freq_hz) >= 1e3:
        return f"{freq_hz/1e3:.3f} kHz"
    else:
        return f"{freq_hz:.3f} Hz"


def format_db(db_value):
    """Format dB value for display."""
    return f"{db_value:.2f} dB"
