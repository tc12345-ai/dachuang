"""HIL & Digital Twin — API Endpoints."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginAPIBase

class Api(PluginAPIBase):
    plugin_id = 'hil_twin'
    def get_routes(self):
        return [
            ('POST', '/api/hil/push', self._handle_push),
            ('POST', '/api/hil/estimate', self._handle_estimate),
        ]

    def _handle_push(self, body, query):
        """POST /api/hil/push {b, a, fs, test_signal?} -> {response?}"""
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        b = np.array(body.get('b', [1.0]))
        a = np.array(body.get('a', [1.0]))
        fs = body.get('fs', 8000)
        ts = body.get('test_signal')
        resp = svc.push_and_test(b, a, fs, np.array(ts) if ts else None)
        return 200, {'response': resp.tolist() if resp is not None else None}

    def _handle_estimate(self, body, query):
        """POST /api/hil/estimate {b, a, fs, word_bits?} -> {cortex_m4, zynq}"""
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        b = np.array(body.get('b', [1.0]))
        a = np.array(body.get('a', [1.0]))
        fs = body.get('fs', 8000)
        wb = body.get('word_bits', 16)
        result = svc.estimate_resources(b, a, fs, wb)
        return 200, {k: v.__dict__ for k, v in result.items()}
