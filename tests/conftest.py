"""
conftest.py — Shared pytest fixtures for DSP Platform tests.
"""

import sys, os
import pytest
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus singleton between tests."""
    from core.event_bus import EventBus
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def bus():
    from core.event_bus import EventBus
    return EventBus.instance()


@pytest.fixture
def sample_signal():
    """1 second sine + noise at 8kHz."""
    fs = 8000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(fs)
    return x, fs


@pytest.fixture
def sample_fir():
    """Simple FIR lowpass (order 32, fc=1kHz, fs=8kHz)."""
    from scipy.signal import firwin
    b = firwin(33, 1000, fs=8000)
    a = np.array([1.0])
    return b, a, 8000


@pytest.fixture
def sample_iir():
    """Butterworth IIR lowpass (order 4, fc=1kHz, fs=8kHz)."""
    from scipy.signal import butter
    b, a = butter(4, 1000, fs=8000)
    return b, a, 8000


@pytest.fixture
def plugins_dir():
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'plugins')
