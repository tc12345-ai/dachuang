"""
Tests for PluginManager — 插件管理器测试
"""

import pytest
import os
from core.plugin_manager import PluginManager
from core.event_bus import EventBus


class TestPluginManager:
    def test_discover(self, bus, plugins_dir):
        pm = PluginManager(bus)
        manifests = pm.discover(plugins_dir)
        assert len(manifests) >= 5
        ids = [m.id for m in manifests]
        assert 'ai_assistant' in ids
        assert 'hil_twin' in ids
        assert 'stress_testing' in ids

    def test_manifest_fields(self, bus, plugins_dir):
        pm = PluginManager(bus)
        manifests = pm.discover(plugins_dir)
        ai = [m for m in manifests if m.id == 'ai_assistant'][0]
        assert ai.version == '0.1.0'
        assert ai.category == 'ai'
        assert 'FilterRecommended' in ai.events_published

    def test_list_plugins(self, bus, plugins_dir):
        pm = PluginManager(bus)
        pm.discover(plugins_dir)
        listing = pm.list_plugins()
        assert all('id' in p for p in listing)
        assert all('active' in p for p in listing)

    def test_load_unload_lifecycle(self, bus, plugins_dir):
        """Test load + unload emits correct events."""
        events = []
        bus.subscribe('PluginLoaded', lambda e: events.append(('loaded', e.payload)))
        bus.subscribe('PluginUnloaded', lambda e: events.append(('unloaded', e.payload)))

        pm = PluginManager(bus)
        pm.discover(plugins_dir)
        pm.load_plugin('stress_testing')
        assert pm.is_loaded('stress_testing')

        pm.unload_plugin('stress_testing')
        assert not pm.is_loaded('stress_testing')
        assert any(t == 'loaded' for t, _ in events)
        assert any(t == 'unloaded' for t, _ in events)


class TestModels:
    def test_signal(self):
        import numpy as np
        from core.models import Signal
        s = Signal(np.zeros(1000), 8000.0)
        assert s.duration == 0.125
        assert s.n_samples == 1000
        assert s.n_channels == 1

    def test_pipeline_apply(self, sample_signal):
        import numpy as np
        from core.models import Signal, FilterStep, Pipeline
        x, fs = sample_signal
        sig = Signal(x, fs)
        step = FilterStep('LP', 'fir',
                          coeffs_b=np.array([0.25, 0.5, 0.25]),
                          coeffs_a=np.array([1.0]))
        pipe = Pipeline(steps=[step])
        out = pipe.apply(sig)
        assert len(out.samples) == len(x)
        assert out.fs == fs

    def test_stress_report(self):
        from core.models import StressReport
        r = StressReport('monte_carlo', n_trials=100, pass_rate=0.95,
                         max_deviation_db=0.3)
        assert r.pass_rate == 0.95
