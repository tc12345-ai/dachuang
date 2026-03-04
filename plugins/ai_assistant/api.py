"""AI Assistant — API Endpoints."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginAPIBase


class Api(PluginAPIBase):
    plugin_id = 'ai_assistant'

    def get_routes(self):
        return [
            ('POST', '/api/ai/chat', self._handle_chat),
            ('POST', '/api/ai/recommend', self._handle_recommend),
            ('POST', '/api/ai/anomalies', self._handle_anomalies),
        ]

    def _handle_chat(self, body, query):
        """POST /api/ai/chat  {text, fs} -> {actions}"""
        from service import Service
        svc = Service()
        from core.event_bus import EventBus
        svc.activate(EventBus.instance(), {})
        text = body.get('text', '')
        fs = body.get('fs', 8000)
        actions = svc.parse_chat(text, fs)
        return 200, {'actions': actions}

    def _handle_recommend(self, body, query):
        """POST /api/ai/recommend  {noisy, clean, fs} -> {recommendations}"""
        from service import Service
        svc = Service()
        from core.event_bus import EventBus
        svc.activate(EventBus.instance(), {})
        noisy = np.array(body.get('noisy', []))
        clean = np.array(body.get('clean', []))
        fs = body.get('fs', 8000)
        recs = svc.recommend_filter(noisy, clean, fs)
        return 200, {'recommendations': recs}

    def _handle_anomalies(self, body, query):
        """POST /api/ai/anomalies  {freq_hz, magnitude_db, fs} -> {anomalies}"""
        from service import Service
        svc = Service()
        from core.event_bus import EventBus
        svc.activate(EventBus.instance(), {})
        freq = np.array(body.get('freq_hz', []))
        mag = np.array(body.get('magnitude_db', []))
        fs = body.get('fs', 8000)
        anomalies = svc.detect_anomalies(freq, mag, fs)
        return 200, {'anomalies': [
            {'type': a.anomaly_type, 'freq': a.frequency,
             'severity': a.severity, 'desc': a.description,
             'suggestion': a.suggestion} for a in anomalies]}
