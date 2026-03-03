"""
REST/SDK API Server — REST/SDK API 服务器

Provides HTTP REST API for external integration with
production test systems and other applications.
Built-in lightweight server using http.server (no external deps).
"""

import json
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filter_design import FilterDesigner, FilterSpec
from core.spectrum_analysis import SpectrumAnalyzer
from core.measurements import SignalMeasurements
from core.code_generator import CodeGenerator


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class DSPApiHandler(BaseHTTPRequestHandler):
    """HTTP request handler for DSP API."""
    
    # Shared instances
    designer = FilterDesigner()
    analyzer = SpectrumAnalyzer()
    measurer = SignalMeasurements()
    codegen = CodeGenerator()
    
    def log_message(self, format, *args):
        """Suppress default logging to console."""
        pass
    
    def _send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        body = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
        self.wfile.write(body.encode('utf-8'))
    
    def _send_error(self, message, status=400):
        """Send error response."""
        self._send_json({'error': message}, status)
    
    def _read_body(self):
        """Read and parse JSON body."""
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        body = self.rfile.read(length).decode('utf-8')
        return json.loads(body)
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        if path == '/api/status':
            self._send_json({
                'status': 'running',
                'version': '1.0',
                'name': 'DSP Platform API',
                'endpoints': [
                    'GET /api/status',
                    'POST /api/filter/design',
                    'POST /api/filter/codegen',
                    'POST /api/spectrum/fft',
                    'POST /api/spectrum/psd',
                    'POST /api/spectrum/measure',
                ]
            })
        
        elif path == '/api/filter/methods':
            self._send_json({
                'fir_methods': FilterDesigner.FIR_METHODS,
                'iir_methods': FilterDesigner.IIR_METHODS,
                'filter_types': FilterDesigner.FILTER_TYPES,
                'windows': FilterDesigner.WINDOWS,
            })
        
        elif path == '/api/spectrum/windows':
            self._send_json({
                'windows': list(SpectrumAnalyzer.WINDOWS.keys())
            })
        
        else:
            self._send_error('Not found', 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            body = self._read_body()
        except Exception as e:
            self._send_error(f'Invalid JSON: {e}')
            return
        
        try:
            if path == '/api/filter/design':
                self._handle_filter_design(body)
            elif path == '/api/filter/codegen':
                self._handle_codegen(body)
            elif path == '/api/spectrum/fft':
                self._handle_fft(body)
            elif path == '/api/spectrum/psd':
                self._handle_psd(body)
            elif path == '/api/spectrum/measure':
                self._handle_measure(body)
            else:
                self._send_error('Not found', 404)
        except Exception as e:
            self._send_error(f'Server error: {str(e)}', 500)
    
    def _handle_filter_design(self, body):
        """Design filter via API."""
        spec = FilterSpec(
            filter_class=body.get('filter_class', 'FIR'),
            design_method=body.get('design_method', 'window'),
            filter_type=body.get('filter_type', 'lowpass'),
            order=body.get('order', 32),
            fs=body.get('fs', 8000),
            fp1=body.get('fp1', 1000),
            fp2=body.get('fp2', 2000),
            fs1=body.get('fs1', 1500),
            fs2=body.get('fs2', 500),
            passband_ripple=body.get('passband_ripple', 1.0),
            stopband_atten=body.get('stopband_atten', 60.0),
            window_type=body.get('window_type', 'hamming'),
        )
        
        result = self.designer.design(spec)
        
        response = {
            'info': result.info,
            'order': result.order,
            'is_stable': result.is_stable,
            'b': result.b,
            'a': result.a,
            'gain': result.gain,
        }
        
        if body.get('include_response', False):
            response['freq_hz'] = result.freq_hz
            response['magnitude_db'] = result.magnitude_db
            response['phase_deg'] = result.phase_deg
        
        if body.get('include_zpk', False):
            response['zeros_real'] = np.real(result.zeros)
            response['zeros_imag'] = np.imag(result.zeros)
            response['poles_real'] = np.real(result.poles)
            response['poles_imag'] = np.imag(result.poles)
        
        if result.sos is not None:
            response['sos'] = result.sos
        
        self._send_json(response)
    
    def _handle_codegen(self, body):
        """Generate code via API."""
        b = np.array(body.get('b', [1.0]))
        a = np.array(body.get('a', [1.0]))
        fs = body.get('fs', 8000)
        language = body.get('language', 'C')
        name = body.get('filter_name', 'api_filter')
        
        code = self.codegen.generate(b, a, fs, language,
                                      filter_name=name,
                                      filter_info='Generated via API')
        
        self._send_json({'language': language, 'code': code})
    
    def _handle_fft(self, body):
        """FFT analysis via API."""
        data = np.array(body.get('data', []))
        fs = body.get('fs', 8000)
        nfft = body.get('nfft', None)
        window = body.get('window', 'Hann')
        
        result = self.analyzer.compute_fft(data, fs, nfft=nfft, window=window)
        
        self._send_json({
            'freq_hz': result.freq_hz,
            'magnitude_db': result.magnitude_db,
            'phase_deg': result.phase_deg,
            'info': result.info,
        })
    
    def _handle_psd(self, body):
        """PSD analysis via API."""
        data = np.array(body.get('data', []))
        fs = body.get('fs', 8000)
        nfft = body.get('nfft', 1024)
        
        result = self.analyzer.compute_psd(data, fs, nfft=nfft)
        
        self._send_json({
            'freq_hz': result.psd_freq,
            'psd_db': result.psd_db,
            'info': result.info,
        })
    
    def _handle_measure(self, body):
        """Signal measurements via API."""
        data = np.array(body.get('data', []))
        fs = body.get('fs', 8000)
        
        result = self.measurer.analyze(data, fs)
        
        self._send_json({
            'fundamental_freq': result.fundamental_freq,
            'thd_percent': result.thd_percent,
            'thd_db': result.thd_db,
            'snr_db': result.snr_db,
            'sfdr_db': result.sfdr_db,
            'sinad_db': result.sinad_db,
            'enob': result.enob,
            'harmonic_freqs': result.harmonic_freqs,
            'harmonic_mags_db': result.harmonic_mags_db,
            'noise_floor_db': result.noise_floor_db,
            'info': result.info,
        })


class DSPApiServer:
    """
    DSP API Server — DSP API 服务器
    
    Manages the HTTP server lifecycle.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
    
    def start(self):
        """Start the API server in a background thread."""
        self._server = HTTPServer((self.host, self.port), DSPApiHandler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                         daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"
    
    def stop(self):
        """Stop the API server."""
        if self._server:
            self._server.shutdown()
            self._server = None
    
    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
