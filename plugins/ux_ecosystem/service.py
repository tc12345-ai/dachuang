"""
UX & Ecosystem — Service Layer
交互体验与生态扩展

1. 3D time-frequency analysis (STFT → 3D surface/waterfall)
2. Plugin marketplace (local registry.json, install/uninstall)
3. Cloud compute pool (local multiprocessing simulation)
"""

import numpy as np
from scipy import signal as sig
from concurrent.futures import ProcessPoolExecutor
import json, os, time, uuid
from typing import Any, Dict, List
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.event_bus import EventBus, Events, make_event
from core.protocols import PluginServiceBase


class Service(PluginServiceBase):
    plugin_id = 'ux_ecosystem'

    def activate(self, bus: EventBus, ctx: Dict[str, Any]):
        self.bus = bus
        self.ctx = ctx
        self._jobs: Dict[str, dict] = {}

    # ═══ 1. 3D Time-Frequency ═══

    def compute_3d_spectrogram(self, x: np.ndarray, fs: float,
                                nperseg: int = 256,
                                noverlap: int = None,
                                window: str = 'hann') -> dict:
        """
        Compute 3D spectrogram data.

        Returns dict with time, freq, magnitude arrays for 3D plotting.
        """
        if noverlap is None:
            noverlap = nperseg * 3 // 4

        f, t, Zxx = sig.stft(x, fs=fs, nperseg=nperseg,
                             noverlap=noverlap, window=window)
        magnitude = np.abs(Zxx)
        mag_db = 20 * np.log10(np.maximum(magnitude, 1e-12))

        self.bus.publish(make_event(Events.SPECTRUM_3D_READY,
                                    source=self.plugin_id,
                                    freq=f.tolist(), time=t.tolist(),
                                    magnitude_db='ndarray'))
        return {
            'freq': f, 'time': t,
            'magnitude': magnitude, 'magnitude_db': mag_db,
            'fs': fs, 'nperseg': nperseg,
        }

    # ═══ 2. Plugin Marketplace ═══

    def get_registry(self, registry_path: str = None) -> List[dict]:
        """Load local plugin registry."""
        if registry_path is None:
            registry_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'registry.json')

        if not os.path.exists(registry_path):
            # Create default registry
            default = [
                {'id': 'ai_assistant', 'name': 'AI-Powered Assistant',
                 'version': '0.1.0', 'installed': True,
                 'description': 'Smart filter recommendation & chat'},
                {'id': 'hil_twin', 'name': 'HIL & Digital Twin',
                 'version': '0.1.0', 'installed': True,
                 'description': 'Virtual instruments & resource estimation'},
                {'id': 'domain_toolkits', 'name': 'Domain Toolkits',
                 'version': '0.1.0', 'installed': True,
                 'description': 'Vibration/BioMed/Acoustic/Comms'},
                {'id': 'stress_testing', 'name': 'Stress Testing',
                 'version': '0.1.0', 'installed': True,
                 'description': 'Monte Carlo & fixed-point sweep'},
                {'id': 'ux_ecosystem', 'name': 'UX & Ecosystem',
                 'version': '0.1.0', 'installed': True,
                 'description': '3D spectrogram & cloud compute'},
            ]
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            return default

        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def verify_signature(self, plugin_path: str) -> bool:
        """Stub: verify plugin signature."""
        # TODO: Implement actual signature verification
        return True

    # ═══ 3. Cloud Compute Pool ═══

    def submit_job(self, func_name: str, args: dict) -> str:
        """
        Submit a compute job (simulated with local multiprocessing).

        Returns job_id.
        """
        job_id = str(uuid.uuid4())[:8]
        self._jobs[job_id] = {
            'status': 'pending', 'func': func_name,
            'args': args, 'result': None, 'submitted': time.time()
        }

        # Simulate async processing
        import threading
        def run_job():
            self._jobs[job_id]['status'] = 'running'
            time.sleep(0.5)  # Simulate compute time
            self._jobs[job_id]['status'] = 'done'
            self._jobs[job_id]['result'] = {'note': 'computed locally'}

        t = threading.Thread(target=run_job, daemon=True)
        t.start()
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        return self._jobs.get(job_id, {'status': 'not_found'})

    def get_job_result(self, job_id: str) -> dict:
        """Get job result."""
        job = self._jobs.get(job_id, {})
        if job.get('status') == 'done':
            return job.get('result', {})
        return {'error': 'Job not ready'}
