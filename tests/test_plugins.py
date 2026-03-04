"""
Tests for Plugin Services — 插件服务测试

Tests AI Assistant, HIL, Stress Testing, and Domain Toolkit services.
"""

import pytest
import numpy as np
from core.event_bus import EventBus


class TestAIAssistant:
    def _make_svc(self, bus):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'plugins', 'ai_assistant'))
        from plugins.ai_assistant.service import Service
        svc = Service()
        svc.activate(bus, {})
        return svc

    def test_chat_parse_notch(self, bus):
        svc = self._make_svc(bus)
        steps = svc.parse_chat("remove 50Hz", fs=8000)
        assert len(steps) >= 1
        assert steps[0]['type'] == 'notch'
        assert steps[0]['freq'] == 50

    def test_chat_parse_bandpass(self, bus):
        svc = self._make_svc(bus)
        steps = svc.parse_chat("bandpass 100-3000Hz", fs=8000)
        assert any(s['type'] == 'bandpass' for s in steps)

    def test_chat_parse_combo(self, bus):
        svc = self._make_svc(bus)
        steps = svc.parse_chat("notch 50Hz + lowpass 4000Hz", fs=8000)
        assert len(steps) >= 2

    def test_anomaly_detection(self, bus):
        svc = self._make_svc(bus)
        # Create spectrum with harmonics at 50, 100, 150, 200 Hz
        freq = np.linspace(0, 4000, 4000)
        mag = np.full_like(freq, -80.0)
        for f0 in [50, 100, 150, 200, 250]:
            idx = int(f0)
            mag[idx] = -10
        anomalies = svc.detect_anomalies(freq, mag, 8000)
        # Should detect harmonic family
        types = [a.anomaly_type for a in anomalies]
        assert 'harmonic_distortion' in types or 'resonance' in types

    def test_recommend_filter(self, bus, sample_signal):
        svc = self._make_svc(bus)
        x, fs = sample_signal
        # Clean = the sine part; noisy = x
        clean = np.sin(2 * np.pi * 440 * np.arange(len(x)) / fs)
        recs = svc.recommend_filter(x, clean, fs)
        assert isinstance(recs, list)


class TestHIL:
    def _make_svc(self, bus):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'plugins', 'hil_twin'))
        from plugins.hil_twin.service import Service
        svc = Service()
        svc.activate(bus, {})
        return svc

    def test_mock_device(self, bus, sample_fir, sample_signal):
        svc = self._make_svc(bus)
        b, a, fs = sample_fir
        x, _ = sample_signal
        resp = svc.push_and_test(b, a, fs, x)
        assert resp is not None
        assert len(resp) == len(x)

    def test_resource_cortex_m4(self, bus, sample_fir):
        svc = self._make_svc(bus)
        b, a, fs = sample_fir
        result = svc.estimate_resources(b, a, fs)
        cm4 = result['cortex_m4']
        assert cm4.macs_per_sample == len(b)
        assert cm4.memory_bytes > 0

    def test_resource_zynq(self, bus, sample_iir):
        svc = self._make_svc(bus)
        b, a, fs = sample_iir
        result = svc.estimate_resources(b, a, fs)
        zynq = result['zynq']
        assert zynq.dsp_slices > 0
        assert zynq.lut_count > 0


class TestStressTesting:
    def _make_svc(self, bus):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'plugins', 'stress_testing'))
        from plugins.stress_testing.service import Service
        svc = Service()
        svc.activate(bus, {})
        return svc

    def test_monte_carlo_fir(self, bus, sample_fir):
        svc = self._make_svc(bus)
        b, a, fs = sample_fir
        report = svc.monte_carlo_sensitivity(b, a, fs, n_trials=50)
        assert report.method == 'monte_carlo'
        assert report.n_trials == 50
        assert 0 <= report.pass_rate <= 1
        assert report.max_deviation_db >= 0

    def test_monte_carlo_iir(self, bus, sample_iir):
        svc = self._make_svc(bus)
        b, a, fs = sample_iir
        report = svc.monte_carlo_sensitivity(b, a, fs, n_trials=50)
        assert report.pass_rate > 0

    def test_fixedpoint_sweep(self, bus, sample_fir):
        svc = self._make_svc(bus)
        b, a, fs = sample_fir
        report = svc.fixedpoint_sweep(b, a, fs, max_error_db=1.0)
        assert report.method == 'fixedpoint_sweep'
        assert report.optimal_q  # Should find a solution

    def test_jitter(self, bus, sample_signal):
        svc = self._make_svc(bus)
        x, fs = sample_signal
        y = svc.simulate_jitter(x, fs, jitter_ppm=100)
        assert len(y) == len(x)

    def test_clipping(self, bus, sample_signal):
        svc = self._make_svc(bus)
        x, fs = sample_signal
        y = svc.simulate_clipping(x, clip_level=0.5)
        assert np.max(np.abs(y)) <= np.max(np.abs(x)) * 0.5 + 1e-10

    def test_packet_loss(self, bus, sample_signal):
        svc = self._make_svc(bus)
        x, fs = sample_signal
        y = svc.simulate_packet_loss(x, loss_rate=0.1)
        assert len(y) == len(x)
        # Some packets should be zeroed
        assert np.any(y == 0)


class TestDomainToolkits:
    def _make_svc(self, bus):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'plugins', 'domain_toolkits'))
        from plugins.domain_toolkits.service import Service
        svc = Service()
        svc.activate(bus, {})
        return svc

    def test_third_octave(self, bus, sample_signal):
        svc = self._make_svc(bus)
        x, fs = sample_signal
        r = svc.third_octave(x, fs)
        assert len(r['center_freqs']) > 0
        assert len(r['levels_db']) == len(r['center_freqs'])

    def test_ecg_baseline(self, bus):
        svc = self._make_svc(bus)
        # Simulate ECG with baseline wander
        fs = 250
        t = np.arange(fs * 5) / fs
        ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        clean = svc.ecg_baseline_removal(ecg, fs)
        assert len(clean) == len(ecg)
        # Baseline should be reduced
        low_power_before = np.mean(ecg[:fs] ** 2)
        low_power_after = np.mean(clean[:fs] ** 2)
        # Not a strict test, just verifies it runs

    def test_eeg_rhythm(self, bus):
        svc = self._make_svc(bus)
        fs = 256
        t = np.arange(fs * 2) / fs
        eeg = (np.sin(2*np.pi*2*t) +     # delta
               np.sin(2*np.pi*10*t) +     # alpha
               0.1 * np.random.randn(len(t)))
        r = svc.eeg_rhythm_extraction(eeg, fs)
        assert 'delta' in r
        assert 'alpha' in r
        assert r['alpha']['power'] > 0

    def test_thdn(self, bus):
        svc = self._make_svc(bus)
        fs = 48000
        t = np.arange(fs) / fs
        # Pure tone → very low THD+N
        x = np.sin(2 * np.pi * 1000 * t)
        r = svc.thd_plus_n(x, fs)
        assert r['thd_n_percent'] < 1  # Should be very low for pure tone
        assert abs(r['fundamental_hz'] - 1000) < 10
