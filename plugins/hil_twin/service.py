"""
HIL & Digital Twin — Service Layer
硬件在环与数字孪生服务层

1. MockDevice adapter (extensible to Serial/TCP)
2. Coefficient push & response capture
3. ARM Cortex-M4 / Xilinx Zynq resource estimation
"""

import numpy as np
import time
import threading
from typing import Any, Dict, List, Optional
from scipy.signal import lfilter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.event_bus import EventBus, Events, make_event
from core.protocols import PluginServiceBase
from core.models import ResourceEstimate


# ═══════════════════════════════════════════════════
#  Device Adapter Protocol
# ═══════════════════════════════════════════════════

class DeviceAdapter:
    """Base class for HIL device connection."""
    name: str = 'unknown'
    connected: bool = False

    def connect(self, **kwargs) -> bool:
        raise NotImplementedError

    def disconnect(self):
        pass

    def push_coefficients(self, b: np.ndarray, a: np.ndarray,
                          fs: float) -> bool:
        raise NotImplementedError

    def send_stimulus(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Send input signal, return output signal."""
        raise NotImplementedError


class MockDevice(DeviceAdapter):
    """
    Mock device for testing — 模拟设备.
    Simulates a DSP/FPGA applying the pushed filter with
    optional quantization noise and latency.
    """
    name = 'MockDevice'

    def __init__(self, latency_ms: float = 1.0, snr_db: float = 60):
        self._b = np.array([1.0])
        self._a = np.array([1.0])
        self._fs = 8000
        self.latency_ms = latency_ms
        self.snr_db = snr_db
        self.connected = False

    def connect(self, **kwargs) -> bool:
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def push_coefficients(self, b, a, fs) -> bool:
        self._b = np.array(b)
        self._a = np.array(a)
        self._fs = fs
        return True

    def send_stimulus(self, signal, fs):
        # Simulate processing latency
        time.sleep(self.latency_ms / 1000)
        # Apply filter
        y = lfilter(self._b, self._a, signal)
        # Add quantization noise
        noise_power = np.mean(y ** 2) / (10 ** (self.snr_db / 10))
        y += np.random.randn(len(y)) * np.sqrt(max(noise_power, 1e-20))
        return y


# TODO: class SerialDevice(DeviceAdapter): ...
# TODO: class TcpDevice(DeviceAdapter): ...


# ═══════════════════════════════════════════════════
#  Resource Estimator
# ═══════════════════════════════════════════════════

class ResourceEstimator:
    """Hardware resource/power estimation — 资源功耗估算."""

    @staticmethod
    def estimate_cortex_m4(b: np.ndarray, a: np.ndarray,
                           fs: float, word_bits: int = 16) -> ResourceEstimate:
        """Estimate for ARM Cortex-M4 (single-MAC, 168 MHz)."""
        nb, na = len(b), len(a)
        is_fir = na <= 1

        macs = nb  # FIR: N multiplies per sample
        if not is_fir:
            macs += na - 1  # IIR feedback

        # Cortex-M4: ~1 cycle per MAC (with pipeline), + overhead
        cycles = macs * 2 + 10  # conservative
        max_fs = 168e6 / cycles  # Max achievable sample rate

        mem = (nb + na) * (word_bits // 8)  # Coefficient storage
        mem += nb * (word_bits // 8)        # Delay buffer
        if not is_fir:
            mem += na * (word_bits // 8)

        # Power: ~30 mW/MHz * utilization
        utilization = fs * cycles / 168e6
        power = 30 * utilization  # rough mW

        return ResourceEstimate(
            target='cortex_m4',
            macs_per_sample=macs,
            memory_bytes=mem,
            cycles_per_sample=cycles,
            power_mw=power,
            notes=f"Max Fs={max_fs:.0f}Hz, util={utilization:.1%}")

    @staticmethod
    def estimate_zynq(b: np.ndarray, a: np.ndarray,
                      fs: float, word_bits: int = 16) -> ResourceEstimate:
        """Estimate for Xilinx Zynq-7020 (Artix-7 fabric)."""
        nb, na = len(b), len(a)
        is_fir = na <= 1

        # DSP48E1 slices: 1 per multiplier (or shared with pipelining)
        dsp = nb if is_fir else nb + na - 1

        # LUTs: ~50 per tap for control + routing
        lut = dsp * 50 + 200  # overhead

        # BRAM: 1 block (18Kb) per ~1K coefficients
        coeff_bits = (nb + na) * word_bits
        bram = max(1, coeff_bits // 18432 + 1)

        # Available on Zynq-7020: 220 DSP48, 53200 LUT, 140 BRAM
        notes = (f"DSP48: {dsp}/220 ({dsp/220:.0%}), "
                 f"LUT: {lut}/53200 ({lut/53200:.1%}), "
                 f"BRAM: {bram}/140")

        return ResourceEstimate(
            target='zynq_7020',
            macs_per_sample=dsp,
            lut_count=lut,
            bram_blocks=bram,
            dsp_slices=dsp,
            notes=notes)


# ═══════════════════════════════════════════════════
#  Plugin Service
# ═══════════════════════════════════════════════════

class Service(PluginServiceBase):
    """HIL & Digital Twin Service."""

    plugin_id = 'hil_twin'

    def activate(self, bus: EventBus, ctx: Dict[str, Any]):
        self.bus = bus
        self.ctx = ctx
        self.device = MockDevice()
        self.estimator = ResourceEstimator()
        bus.subscribe(Events.FILTER_DESIGNED, self._on_filter,
                      subscriber_id=self.plugin_id)

    def deactivate(self):
        if self.device:
            self.device.disconnect()

    def push_and_test(self, b, a, fs, test_signal=None):
        """Push coefficients to device and run test."""
        self.device.connect()
        self.device.push_coefficients(b, a, fs)

        self.bus.publish(make_event(Events.HIL_COEFF_PUSHED,
                                    source=self.plugin_id,
                                    device=self.device.name,
                                    coeffs_b=b.tolist(), coeffs_a=a.tolist()))

        if test_signal is not None:
            response = self.device.send_stimulus(test_signal, fs)
            self.bus.publish(make_event(Events.HIL_RESPONSE_RECV,
                                        source=self.plugin_id,
                                        device=self.device.name,
                                        response=response.tolist()))
            return response
        return None

    def estimate_resources(self, b, a, fs, word_bits=16):
        """Estimate resources for both Cortex-M4 and Zynq."""
        cm4 = self.estimator.estimate_cortex_m4(b, a, fs, word_bits)
        zynq = self.estimator.estimate_zynq(b, a, fs, word_bits)

        self.bus.publish(make_event(Events.RESOURCE_ESTIMATED,
                                    source=self.plugin_id,
                                    cortex_m4=cm4.__dict__,
                                    zynq=zynq.__dict__))
        return {'cortex_m4': cm4, 'zynq': zynq}

    def _on_filter(self, event):
        """Auto-estimate resources when filter is designed."""
        p = event.payload
        if 'b' in p and 'a' in p:
            b = np.asarray(p['b'])
            a = np.asarray(p['a'])
            fs = p.get('fs', 8000)
            self.estimate_resources(b, a, fs)
