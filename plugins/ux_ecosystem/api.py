"""UX & Ecosystem — API."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginAPIBase

class Api(PluginAPIBase):
    plugin_id = 'ux_ecosystem'
    def get_routes(self):
        return [
            ('GET',  '/api/ecosystem/registry', self._handle_registry),
            ('POST', '/api/ecosystem/3d', self._handle_3d),
            ('POST', '/api/cloud/submit', self._handle_submit),
            ('GET',  '/api/cloud/status', self._handle_status),
        ]

    def _handle_registry(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        return 200, {'plugins': svc.get_registry()}

    def _handle_3d(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        x = np.array(body.get('data', []))
        fs = body.get('fs', 8000)
        r = svc.compute_3d_spectrogram(x, fs)
        return 200, {'freq': r['freq'].tolist(),
                     'time': r['time'].tolist(),
                     'shape': list(r['magnitude_db'].shape)}

    def _handle_submit(self, body, query):
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        jid = svc.submit_job(body.get('func', 'test'), body.get('args', {}))
        return 200, {'job_id': jid}

    def _handle_status(self, body, query):
        jid = query.get('job_id', [''])[0]
        from service import Service
        from core.event_bus import EventBus
        svc = Service(); svc.activate(EventBus.instance(), {})
        return 200, svc.get_job_status(jid)
