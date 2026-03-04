"""
Data Models — 统一数据模型

Canonical dataclasses shared across all plugins and the core.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    """Canonical signal container — 标准信号容器."""
    samples: np.ndarray                  # (N,) or (N, C) for multi-channel
    fs: float                            # Sampling rate
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.fs

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]

    def channel(self, idx: int = 0) -> np.ndarray:
        if self.samples.ndim == 1:
            return self.samples
        return self.samples[:, idx]


@dataclass
class FilterStep:
    """One step in a processing pipeline — 流水线步骤."""
    name: str                            # Human-readable
    kind: str                            # 'fir', 'iir', 'notch', 'custom', ...
    params: Dict[str, Any] = field(default_factory=dict)
    coeffs_b: Optional[np.ndarray] = None
    coeffs_a: Optional[np.ndarray] = None

    def is_fir(self) -> bool:
        return self.coeffs_a is None or (
            len(self.coeffs_a) == 1 and self.coeffs_a[0] == 1.0)


@dataclass
class Pipeline:
    """Ordered chain of FilterSteps — 滤波器链."""
    steps: List[FilterStep] = field(default_factory=list)
    name: str = 'default'

    def add(self, step: FilterStep):
        self.steps.append(step)

    def apply(self, signal: Signal) -> Signal:
        """Apply all steps sequentially."""
        from scipy.signal import lfilter
        y = signal.samples.copy()
        for step in self.steps:
            b = step.coeffs_b if step.coeffs_b is not None else np.array([1.0])
            a = step.coeffs_a if step.coeffs_a is not None else np.array([1.0])
            y = lfilter(b, a, y)
        return Signal(samples=y, fs=signal.fs,
                      meta={**signal.meta, 'pipeline': self.name})


@dataclass
class AnomalyInfo:
    """Detected anomaly in spectrum / signal — 异常信息."""
    anomaly_type: str       # 'harmonic_distortion', 'resonance', 'non_stationary', ...
    frequency: float        # Hz (0 if time-domain)
    severity: float         # 0..1
    description: str = ''
    suggestion: str = ''


@dataclass
class ResourceEstimate:
    """Hardware resource estimate — 硬件资源估算."""
    target: str             # 'cortex_m4', 'zynq_7020', ...
    macs_per_sample: int = 0
    memory_bytes: int = 0
    cycles_per_sample: int = 0
    lut_count: int = 0      # FPGA
    bram_blocks: int = 0    # FPGA
    dsp_slices: int = 0     # FPGA
    power_mw: float = 0.0
    notes: str = ''


@dataclass
class StressReport:
    """Result of stress / sensitivity testing — 压力测试报告."""
    method: str             # 'monte_carlo', 'jitter', 'fixedpoint_sweep'
    n_trials: int = 0
    pass_rate: float = 0.0
    max_deviation_db: float = 0.0
    stability_margin: float = 0.0
    optimal_q: str = ''     # e.g. 'Q3.13'
    details: Dict[str, Any] = field(default_factory=dict)
