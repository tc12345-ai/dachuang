"""Domain Toolkits — API Endpoints."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginAPIBase

class Api(PluginAPIBase):
    plugin_id = 'domain_toolkits'
    def get_routes(self):
        return [
            ('POST', '/api/domain/envelope', self._handle_envelope),
            ('POST', '/api/domain/third_octave', self._handle_octave),
            ('POST', '/api/domain/ecg_baseline', self._handle_ecg),
            ('POST', '/api/domain/thd_n', self._handle_thdn),
        ]

    def _handle_envelope(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        x = np.array(body.get('data', [])); fs = body.get('fs', 8000)
        r = svc.envelope_demodulation(x, fs)
        return 200, {'env_freq': r['env_freq'].tolist(),
                     'env_spectrum_db': r['env_spectrum_db'].tolist()}

    def _handle_octave(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        x = np.array(body.get('data', [])); fs = body.get('fs', 8000)
        r = svc.third_octave(x, fs)
        return 200, r

    def _handle_ecg(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        x = np.array(body.get('data', [])); fs = body.get('fs', 250)
        y = svc.ecg_baseline_removal(x, fs)
        return 200, {'filtered': y.tolist()}

    def _handle_thdn(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        x = np.array(body.get('data', [])); fs = body.get('fs', 48000)
        r = svc.thd_plus_n(x, fs)
        return 200, r
