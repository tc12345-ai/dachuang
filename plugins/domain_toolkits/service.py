"""
Domain Toolkits — Service Layer
垂直行业专业套件

Vibration: Order tracking, envelope demodulation, 1/3 octave
BioMed:    ECG baseline, EEG rhythm, artifact removal
Acoustic:  Equal loudness, THD+N test
Comms:     Constellation, eye diagram, EVM, DDC
"""

import numpy as np
from scipy import signal as sig
from typing import Any, Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.event_bus import EventBus, Events, make_event
from core.protocols import PluginServiceBase


class Service(PluginServiceBase):
    plugin_id = 'domain_toolkits'

    def activate(self, bus: EventBus, ctx: Dict[str, Any]):
        self.bus = bus
        self.ctx = ctx

    # ═══ Vibration ═══

    def envelope_demodulation(self, x: np.ndarray, fs: float,
                               bp_low: float = 500,
                               bp_high: float = 5000) -> dict:
        """Envelope demodulation for bearing fault detection."""
        # Bandpass filter
        sos = sig.butter(4, [bp_low, bp_high], 'bandpass', fs=fs, output='sos')
        filtered = sig.sosfilt(sos, x)
        # Hilbert envelope
        analytic = sig.hilbert(filtered)
        envelope = np.abs(analytic)
        # FFT of envelope
        env_fft = np.abs(np.fft.rfft(envelope))
        env_freq = np.fft.rfftfreq(len(envelope), 1/fs)
        env_db = 20 * np.log10(np.maximum(env_fft, 1e-12))
        return {'envelope': envelope, 'env_freq': env_freq,
                'env_spectrum_db': env_db}

    def third_octave(self, x: np.ndarray, fs: float) -> dict:
        """1/3 octave band analysis."""
        center_freqs = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
                        250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                        2000, 2500, 3150, 4000, 5000, 6300, 8000,
                        10000, 12500, 16000, 20000]
        factor = 2 ** (1/6)
        levels = []
        valid_freqs = []
        for fc in center_freqs:
            fl = fc / factor
            fh = fc * factor
            if fh >= fs / 2:
                break
            if fl < 1:
                fl = 1
            try:
                sos = sig.butter(3, [fl, fh], 'bandpass', fs=fs, output='sos')
                filtered = sig.sosfilt(sos, x)
                rms = np.sqrt(np.mean(filtered ** 2))
                levels.append(20 * np.log10(max(rms, 1e-12)))
                valid_freqs.append(fc)
            except Exception:
                continue
        return {'center_freqs': valid_freqs, 'levels_db': levels}

    # ═══ BioMed ═══

    def ecg_baseline_removal(self, x: np.ndarray, fs: float,
                              cutoff: float = 0.5) -> np.ndarray:
        """Remove ECG baseline wander using highpass filter."""
        sos = sig.butter(2, cutoff, 'highpass', fs=fs, output='sos')
        return sig.sosfiltfilt(sos, x)

    def eeg_rhythm_extraction(self, x: np.ndarray, fs: float) -> dict:
        """Extract standard EEG rhythms (delta/theta/alpha/beta/gamma)."""
        bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, min(100, fs/2-1))
        }
        result = {}
        for name, (fl, fh) in bands.items():
            if fh >= fs / 2:
                continue
            sos = sig.butter(3, [fl, fh], 'bandpass', fs=fs, output='sos')
            filtered = sig.sosfiltfilt(sos, x)
            power = np.mean(filtered ** 2)
            result[name] = {'signal': filtered, 'power': power,
                           'power_db': 10*np.log10(max(power, 1e-20))}
        return result

    # ═══ Acoustic ═══

    def thd_plus_n(self, x: np.ndarray, fs: float,
                    fundamental: float = None) -> dict:
        """THD+N measurement per IEC 61000."""
        N = len(x)
        X = np.abs(np.fft.rfft(x * np.hanning(N)))
        freq = np.fft.rfftfreq(N, 1/fs)

        if fundamental is None:
            fundamental = float(freq[np.argmax(X[1:]) + 1])

        # Notch out fundamental
        f0_idx = np.argmin(np.abs(freq - fundamental))
        bw = max(3, int(N * 20 / fs))
        X_notched = X.copy()
        lo = max(0, f0_idx - bw)
        hi = min(len(X), f0_idx + bw)
        signal_power = np.sum(X[lo:hi] ** 2)
        X_notched[lo:hi] = 0

        noise_distortion_power = np.sum(X_notched ** 2)
        total_power = np.sum(X ** 2)
        thd_n = np.sqrt(noise_distortion_power / max(total_power, 1e-20))

        return {'thd_n_percent': thd_n * 100,
                'thd_n_db': 20 * np.log10(max(thd_n, 1e-12)),
                'fundamental_hz': fundamental}

    # ═══ Communications ═══

    def constellation_points(self, iq: np.ndarray) -> dict:
        """Extract constellation diagram data from IQ samples."""
        I = np.real(iq)
        Q = np.imag(iq)
        evm_values = np.abs(iq) / np.mean(np.abs(iq)) - 1
        evm_rms = np.sqrt(np.mean(evm_values ** 2)) * 100
        return {'I': I, 'Q': Q, 'evm_percent': evm_rms,
                'n_symbols': len(iq)}

    def eye_diagram(self, x: np.ndarray, sps: int) -> dict:
        """Generate eye diagram data."""
        n_symbols = len(x) // sps
        traces = []
        for i in range(min(n_symbols - 2, 200)):
            start = i * sps
            trace = x[start:start + 2 * sps]
            if len(trace) == 2 * sps:
                traces.append(trace)
        return {'traces': traces, 'sps': sps, 'n_traces': len(traces)}
