"""
AI-Powered Assistant — Service Layer
智能化辅助设计服务层

1. Smart filter recommendation (noise analysis → optimal filter cascade)
2. Anomaly detection & auto-diagnosis (spectral anomalies)
3. Chat-to-Signal (natural language → FilterSpec → design)
"""

import numpy as np
from scipy import signal as sig
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from core.event_bus import EventBus, Events, make_event
from core.protocols import PluginServiceBase
from core.models import Signal, AnomalyInfo, FilterStep, Pipeline


class Service(PluginServiceBase):
    """AI Assistant Service — 智能助手服务."""

    plugin_id = 'ai_assistant'

    def activate(self, bus: EventBus, ctx: Dict[str, Any]):
        self.bus = bus
        self.ctx = ctx
        bus.subscribe(Events.SPECTRUM_COMPUTED, self._on_spectrum,
                      subscriber_id=self.plugin_id)
        bus.subscribe(Events.CHAT_COMMAND, self._on_chat,
                      subscriber_id=self.plugin_id)

    def deactivate(self):
        pass

    # ═══ 1. Smart Filter Recommendation ═══

    def recommend_filter(self, noisy: np.ndarray, clean_segment: np.ndarray,
                         fs: float) -> List[Dict]:
        """
        Analyze noise profile and recommend optimal filter cascade.

        Args:
            noisy: Full noisy signal
            clean_segment: User-labeled clean segment
            fs: Sampling rate
        Returns:
            List of filter recommendation dicts
        """
        recs = []

        # Spectral analysis of noise = noisy - clean (aligned)
        n = min(len(noisy), len(clean_segment))
        noise_est = noisy[:n] - clean_segment[:n]

        noise_fft = np.abs(np.fft.rfft(noise_est * np.hanning(n)))
        freq = np.fft.rfftfreq(n, 1/fs)
        noise_db = 20 * np.log10(np.maximum(noise_fft, 1e-12))

        # Detect narrowband noise (peaks)
        from scipy.signal import find_peaks
        peaks, props = find_peaks(noise_db, height=-20, prominence=10, distance=5)

        for pk in peaks[:5]:
            f_peak = freq[pk]
            recs.append({
                'type': 'notch',
                'method': 'IIR Notch (iirnotch)',
                'params': {'freq': float(f_peak), 'Q': 30, 'fs': fs},
                'reason': f'Narrowband noise at {f_peak:.1f} Hz',
            })

        # Detect broadband noise level
        signal_fft = np.abs(np.fft.rfft(clean_segment[:n] * np.hanning(n)))
        signal_db = 20 * np.log10(np.maximum(signal_fft, 1e-12))

        # Find signal bandwidth (where signal > noise + 6dB)
        snr_local = signal_db - noise_db
        bw_mask = snr_local > 6
        if np.any(bw_mask):
            bw_freqs = freq[bw_mask]
            f_low = float(bw_freqs[0])
            f_high = float(bw_freqs[-1])

            if f_low > 10:
                recs.append({
                    'type': 'bandpass',
                    'method': 'FIR Bandpass (Parks-McClellan)',
                    'params': {'fp1': f_low, 'fp2': f_high,
                               'order': 'auto', 'fs': fs},
                    'reason': f'Signal bandwidth {f_low:.0f}-{f_high:.0f} Hz',
                })
            else:
                recs.append({
                    'type': 'lowpass',
                    'method': 'Butterworth LP (IIR)',
                    'params': {'fc': f_high, 'order': 4, 'fs': fs},
                    'reason': f'Signal bandwidth up to {f_high:.0f} Hz',
                })

        # Publish recommendation
        self.bus.publish(make_event(Events.FILTER_RECOMMENDED,
                                    source=self.plugin_id,
                                    recommendations=recs))
        return recs

    # ═══ 2. Anomaly Detection & Auto-Diagnosis ═══

    def detect_anomalies(self, freq_hz: np.ndarray,
                         magnitude_db: np.ndarray,
                         fs: float) -> List[AnomalyInfo]:
        """
        Detect spectral anomalies.

        Checks: harmonic families, resonance peaks, non-stationarity.
        """
        anomalies = []

        # 1) Find dominant peaks
        from scipy.signal import find_peaks
        peaks, props = find_peaks(magnitude_db, height=-40,
                                  prominence=15, distance=10)

        if len(peaks) < 2:
            return anomalies

        peak_freqs = freq_hz[peaks]
        peak_mags = magnitude_db[peaks]

        # 2) Harmonic family detection
        f0 = peak_freqs[np.argmax(peak_mags)]
        harmonics_found = []
        for f in peak_freqs:
            ratio = f / f0 if f0 > 0 else 0
            if ratio > 1.5 and abs(ratio - round(ratio)) < 0.08:
                harmonics_found.append(f)

        if len(harmonics_found) >= 2:
            anomalies.append(AnomalyInfo(
                anomaly_type='harmonic_distortion',
                frequency=f0,
                severity=min(1.0, len(harmonics_found) / 5),
                description=f'Harmonic family at f0={f0:.1f} Hz '
                            f'({len(harmonics_found)} harmonics)',
                suggestion='Consider notch filter cascade or check for '
                           'nonlinear distortion in the signal path'))

        # 3) Resonance detection (very sharp isolated peaks)
        for i, pk in enumerate(peaks):
            prom = props['prominences'][i]
            if prom > 30:
                anomalies.append(AnomalyInfo(
                    anomaly_type='resonance',
                    frequency=float(freq_hz[pk]),
                    severity=min(1.0, prom / 50),
                    description=f'Sharp resonance at {freq_hz[pk]:.1f} Hz '
                                f'(prominence {prom:.1f} dB)',
                    suggestion='Possible structural resonance or oscillation'))

        # 4) Noise floor unevenness → non-stationarity hint
        noise_region = magnitude_db[magnitude_db < np.median(magnitude_db)]
        if len(noise_region) > 10:
            noise_std = np.std(noise_region)
            if noise_std > 8:
                anomalies.append(AnomalyInfo(
                    anomaly_type='non_stationary',
                    frequency=0,
                    severity=min(1.0, noise_std / 15),
                    description=f'Non-stationary noise floor '
                                f'(std={noise_std:.1f} dB)',
                    suggestion='Signal may be time-varying; '
                               'consider STFT or wavelet analysis'))

        self.bus.publish(make_event(Events.ANOMALY_DETECTED,
                                    source=self.plugin_id,
                                    anomalies=[{'type': a.anomaly_type,
                                                'freq': a.frequency,
                                                'severity': a.severity,
                                                'desc': a.description}
                                               for a in anomalies]))
        return anomalies

    # ═══ 3. Chat-to-Signal ═══

    def parse_chat(self, text: str, fs: float = 8000) -> List[Dict]:
        """
        Parse natural language filter request.

        Examples:
          "50Hz notch + bandpass 100-3000Hz"
          "lowpass 1kHz order 16 FIR"
          "remove 60Hz hum and highpass at 20Hz"
        """
        import re
        text = text.lower().strip()
        steps = []

        # Notch patterns
        notch_pat = re.findall(
            r'(?:notch|remove|reject|消除|去除)\s*(?:at\s*)?(\d+)\s*(?:hz)?', text)
        for f in notch_pat:
            steps.append({
                'type': 'notch', 'freq': float(f), 'Q': 30, 'fs': fs,
                'desc': f'Notch filter at {f} Hz'})

        # Lowpass
        lp_pat = re.findall(
            r'(?:lowpass|low-pass|低通|lp)\s*(?:at\s*)?(\d+\.?\d*)\s*(?:k?hz)?', text)
        for f in lp_pat:
            fc = float(f)
            if 'k' in text[text.find(f):text.find(f)+len(f)+3]:
                fc *= 1000
            steps.append({
                'type': 'lowpass', 'fc': fc, 'order': 8, 'fs': fs,
                'desc': f'Lowpass at {fc:.0f} Hz'})

        # Highpass
        hp_pat = re.findall(
            r'(?:highpass|high-pass|高通|hp)\s*(?:at\s*)?(\d+\.?\d*)\s*(?:k?hz)?', text)
        for f in hp_pat:
            fc = float(f)
            steps.append({
                'type': 'highpass', 'fc': fc, 'order': 4, 'fs': fs,
                'desc': f'Highpass at {fc:.0f} Hz'})

        # Bandpass
        bp_pat = re.findall(
            r'(?:bandpass|band-pass|带通|bp)\s*(\d+\.?\d*)\s*[-–to到]\s*(\d+\.?\d*)\s*(?:k?hz)?',
            text)
        for fl, fh in bp_pat:
            steps.append({
                'type': 'bandpass', 'fp1': float(fl), 'fp2': float(fh),
                'order': 8, 'fs': fs,
                'desc': f'Bandpass {fl}-{fh} Hz'})

        # Order override
        order_pat = re.findall(r'(?:order|阶数)\s*(\d+)', text)
        if order_pat and steps:
            steps[-1]['order'] = int(order_pat[0])

        # FIR/IIR preference
        if 'fir' in text:
            for s in steps:
                s['class'] = 'FIR'
        elif 'iir' in text:
            for s in steps:
                s['class'] = 'IIR'

        self.bus.publish(make_event(Events.CHAT_RESPONSE,
                                    source=self.plugin_id,
                                    text=f'Parsed {len(steps)} filter(s)',
                                    actions=steps))
        return steps

    # ─── EventBus handlers ───

    def _on_spectrum(self, event):
        """Auto-diagnose on spectrum computation."""
        p = event.payload
        if 'freq_hz' in p and 'magnitude_db' in p:
            self.detect_anomalies(
                np.asarray(p['freq_hz']),
                np.asarray(p['magnitude_db']),
                p.get('fs', 8000))

    def _on_chat(self, event):
        """Handle chat command."""
        text = event.payload.get('text', '')
        fs = event.payload.get('fs', 8000)
        if text:
            self.parse_chat(text, fs)
