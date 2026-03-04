"""Stress Testing — API."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginAPIBase

class Api(PluginAPIBase):
    plugin_id = 'stress_testing'
    def get_routes(self):
        return [('POST', '/api/stress/monte_carlo', self._handle_mc),
                ('POST', '/api/stress/fixedpoint', self._handle_fp)]

    def _handle_mc(self, body, query):
        """POST /api/stress/monte_carlo {b,a,fs,n_trials,perturbation_std}"""
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        b = np.array(body.get('b', [1.0]))
        a = np.array(body.get('a', [1.0]))
        r = svc.monte_carlo_sensitivity(
            b, a, body.get('fs', 8000),
            body.get('n_trials', 500), body.get('perturbation_std', 1e-4))
        return 200, r.__dict__

    def _handle_fp(self, body, query):
        """POST /api/stress/fixedpoint {b,a,fs,max_error_db}"""
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        b = np.array(body.get('b', [1.0]))
        a = np.array(body.get('a', [1.0]))
        r = svc.fixedpoint_sweep(b, a, body.get('fs', 8000),
                                  body.get('max_error_db', 0.5))
        return 200, r.__dict__
