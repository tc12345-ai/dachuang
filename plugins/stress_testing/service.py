"""
Stress Testing — Service Layer
算法鲁棒性与边界测试

1. Non-stationary simulation: jitter, packet loss, clipping
2. Monte Carlo sensitivity: random coefficient perturbation
3. Fixed-point auto-optimization: minimum bit-width search
"""

import numpy as np
from scipy import signal as sig
from scipy.signal import freqz
from typing import Any, Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.event_bus import EventBus, Events, make_event
from core.protocols import PluginServiceBase
from core.models import StressReport


class Service(PluginServiceBase):
    """Stress Testing Service — 鲁棒性测试服务."""
    plugin_id = 'stress_testing'

    def activate(self, bus: EventBus, ctx: Dict[str, Any]):
        self.bus = bus
        self.ctx = ctx

    # ═══ 1. Non-Stationary Simulation ═══

    def simulate_jitter(self, x: np.ndarray, fs: float,
                        jitter_ppm: float = 100) -> np.ndarray:
        """Simulate sample rate jitter via non-uniform resampling."""
        n = len(x)
        # Jittered sample indices
        jitter = 1 + jitter_ppm * 1e-6 * np.random.randn(n)
        t_jittered = np.cumsum(jitter) / fs
        t_ideal = np.arange(n) / fs
        # Resample
        return np.interp(t_ideal, t_jittered, x)

    def simulate_packet_loss(self, x: np.ndarray,
                              loss_rate: float = 0.01,
                              packet_size: int = 64) -> np.ndarray:
        """Simulate packet loss with zero-fill."""
        y = x.copy()
        n_packets = len(x) // packet_size
        n_lost = int(n_packets * loss_rate)
        lost_idx = np.random.choice(n_packets, n_lost, replace=False)
        for i in lost_idx:
            y[i*packet_size:(i+1)*packet_size] = 0
        return y

    def simulate_clipping(self, x: np.ndarray,
                           clip_level: float = 0.8) -> np.ndarray:
        """Simulate amplitude clipping/saturation."""
        peak = np.max(np.abs(x))
        threshold = peak * clip_level
        return np.clip(x, -threshold, threshold)

    # ═══ 2. Monte Carlo Sensitivity ═══

    def monte_carlo_sensitivity(self, b: np.ndarray, a: np.ndarray,
                                 fs: float,
                                 n_trials: int = 500,
                                 perturbation_std: float = 1e-4
                                 ) -> StressReport:
        """
        Monte Carlo: perturb coefficients, measure frequency response deviation.

        Returns StressReport with max_deviation_db and stability_margin.
        """
        # Reference response
        w_ref, H_ref = freqz(b, a, worN=512, fs=fs)
        H_ref_db = 20 * np.log10(np.maximum(np.abs(H_ref), 1e-12))

        max_devs = []
        unstable_count = 0

        for _ in range(n_trials):
            b_p = b + np.random.randn(len(b)) * perturbation_std
            a_p = a.copy()
            if len(a) > 1:
                a_p[1:] += np.random.randn(len(a)-1) * perturbation_std

            # Stability check
            if len(a_p) > 1:
                poles = np.roots(a_p)
                if np.any(np.abs(poles) >= 1.0):
                    unstable_count += 1
                    continue

            w, H = freqz(b_p, a_p, worN=512, fs=fs)
            H_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
            max_devs.append(float(np.max(np.abs(H_db - H_ref_db))))

        if not max_devs:
            max_devs = [999.0]

        report = StressReport(
            method='monte_carlo',
            n_trials=n_trials,
            pass_rate=1.0 - unstable_count / n_trials,
            max_deviation_db=float(np.max(max_devs)),
            stability_margin=float(np.percentile(max_devs, 95)),
            details={
                'mean_dev_db': float(np.mean(max_devs)),
                'p95_dev_db': float(np.percentile(max_devs, 95)),
                'unstable_trials': unstable_count,
                'perturbation_std': perturbation_std,
            })

        self.bus.publish(make_event(Events.STRESS_TEST_DONE,
                                    source=self.plugin_id,
                                    report=report.__dict__))
        return report

    # ═══ 3. Fixed-Point Auto-Optimization ═══

    def fixedpoint_sweep(self, b: np.ndarray, a: np.ndarray,
                         fs: float,
                         max_error_db: float = 0.5,
                         min_bits: int = 8,
                         max_bits: int = 32) -> StressReport:
        """
        Find minimum bit-width (fractional bits) that keeps
        frequency response error below threshold.
        """
        w_ref, H_ref = freqz(b, a, worN=512, fs=fs)
        H_ref_db = 20 * np.log10(np.maximum(np.abs(H_ref), 1e-12))

        optimal_wl = max_bits
        optimal_frac = max_bits - 2

        for total_bits in range(min_bits, max_bits + 1):
            for frac_bits in range(total_bits - 4, total_bits):
                if frac_bits < 1:
                    continue
                scale = 2 ** frac_bits
                b_q = np.round(b * scale) / scale
                a_q = np.round(a * scale) / scale if len(a) > 1 else a

                w, H = freqz(b_q, a_q, worN=512, fs=fs)
                H_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
                error = float(np.max(np.abs(H_db - H_ref_db)))

                if error <= max_error_db:
                    optimal_wl = total_bits
                    optimal_frac = frac_bits
                    return StressReport(
                        method='fixedpoint_sweep',
                        n_trials=(max_bits - min_bits + 1),
                        pass_rate=1.0,
                        max_deviation_db=error,
                        optimal_q=f'Q{total_bits - frac_bits - 1}.{frac_bits}',
                        details={
                            'word_length': total_bits,
                            'frac_bits': frac_bits,
                            'error_db': error,
                            'threshold_db': max_error_db,
                        })

        return StressReport(
            method='fixedpoint_sweep',
            pass_rate=0.0,
            max_deviation_db=999,
            optimal_q=f'Q{max_bits}',
            details={'note': 'No solution found within bit range'})

    # ═══ Combined Stress Test Runner ═══

    def run_full_stress(self, b, a, fs, signal=None,
                        n_mc_trials=200) -> Dict:
        """Run all stress tests and return combined report."""
        results = {}

        # Monte Carlo
        results['monte_carlo'] = self.monte_carlo_sensitivity(
            b, a, fs, n_mc_trials).__dict__

        # Fixed-point sweep
        results['fixedpoint'] = self.fixedpoint_sweep(
            b, a, fs).__dict__

        # Signal degradation tests
        if signal is not None:
            from scipy.signal import lfilter
            y_ref = lfilter(b, a, signal)

            for name, func in [
                ('jitter', lambda x: self.simulate_jitter(x, fs)),
                ('packet_loss', lambda x: self.simulate_packet_loss(x)),
                ('clipping', lambda x: self.simulate_clipping(x)),
            ]:
                x_deg = func(signal)
                y_deg = lfilter(b, a, x_deg)
                snr = 10*np.log10(np.sum(y_ref**2) /
                                   max(np.sum((y_deg-y_ref)**2), 1e-20))
                results[name] = {'snr_db': float(snr)}

        return results
