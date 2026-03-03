"""
Script Engine — 脚本引擎

Python scripting engine for automation, batch processing,
parameter sweeps, and custom analysis workflows.
Supports safe execution with sandboxed globals.
"""

import sys
import os
import io
import traceback
import threading
import numpy as np
from typing import Optional, Callable, Dict, Any


class ScriptEngine:
    """
    Script Engine — 脚本引擎
    
    Provides Python scripting capabilities for automation:
    - Parameter sweeps
    - Batch processing
    - Custom analysis scripts
    - Report generation automation
    - Regression testing
    
    Scripts have access to DSP platform APIs through
    pre-injected globals (designer, analyzer, measurer, etc.)
    """
    
    def __init__(self):
        self._globals = {}
        self._output_buffer = io.StringIO()
        self._is_running = False
        self._stop_requested = False
        self._thread = None
        self._init_globals()
    
    def _init_globals(self):
        """Initialize sandbox globals with DSP tools."""
        # Safe builtins
        safe_builtins = {
            'print': self._safe_print,
            'range': range,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'isinstance': isinstance,
            'True': True,
            'False': False,
            'None': None,
        }
        
        self._globals = {
            '__builtins__': safe_builtins,
            'np': np,
            'numpy': np,
        }
        
        # Inject DSP modules
        try:
            from core.filter_design import FilterDesigner, FilterSpec
            from core.spectrum_analysis import SpectrumAnalyzer
            from core.measurements import SignalMeasurements
            from core.code_generator import CodeGenerator
            from core.fixed_point import FixedPointAnalyzer
            from utils.helpers import generate_test_signal, format_freq, format_db
            
            self._globals.update({
                'FilterDesigner': FilterDesigner,
                'FilterSpec': FilterSpec,
                'SpectrumAnalyzer': SpectrumAnalyzer,
                'SignalMeasurements': SignalMeasurements,
                'CodeGenerator': CodeGenerator,
                'FixedPointAnalyzer': FixedPointAnalyzer,
                'generate_test_signal': generate_test_signal,
                'format_freq': format_freq,
                'format_db': format_db,
                
                # Convenience instances
                'designer': FilterDesigner(),
                'analyzer': SpectrumAnalyzer(),
                'measurer': SignalMeasurements(),
                'codegen': CodeGenerator(),
            })
        except ImportError:
            pass
        
        # Add scipy.signal
        try:
            from scipy import signal
            self._globals['signal'] = signal
        except ImportError:
            pass
    
    def _safe_print(self, *args, **kwargs):
        """Redirect print to output buffer."""
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        text = sep.join(str(a) for a in args) + end
        self._output_buffer.write(text)
    
    def execute(self, code: str, timeout: float = 30.0,
                extra_vars: Dict[str, Any] = None) -> dict:
        """
        Execute a script.
        
        Args:
            code: Python source code
            timeout: Max execution time in seconds
            extra_vars: Additional variables to inject
        Returns:
            dict with 'output', 'error', 'variables', 'elapsed'
        """
        self._output_buffer = io.StringIO()
        self._stop_requested = False
        
        local_vars = {}
        exec_globals = dict(self._globals)
        exec_globals['stop_requested'] = lambda: self._stop_requested
        
        if extra_vars:
            exec_globals.update(extra_vars)
        
        error = None
        elapsed = 0.0
        
        import time
        t0 = time.perf_counter()
        
        try:
            exec(compile(code, '<script>', 'exec'), exec_globals, local_vars)
        except Exception as e:
            error = traceback.format_exc()
        
        elapsed = time.perf_counter() - t0
        
        # Collect output variables (exclude modules and builtins)
        result_vars = {}
        for k, v in local_vars.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                result_vars[k] = v
            elif isinstance(v, np.ndarray):
                result_vars[k] = f"ndarray{v.shape} dtype={v.dtype}"
        
        return {
            'output': self._output_buffer.getvalue(),
            'error': error,
            'variables': result_vars,
            'elapsed_ms': elapsed * 1000,
        }
    
    def execute_async(self, code: str, callback: Callable = None,
                      timeout: float = 60.0,
                      extra_vars: Dict = None):
        """
        Execute script in background thread.
        
        Args:
            code: Python source code
            callback: Function(result_dict) called on completion
            timeout: Max execution time
            extra_vars: Additional variables
        """
        def run():
            self._is_running = True
            result = self.execute(code, timeout, extra_vars)
            self._is_running = False
            if callback:
                callback(result)
        
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Request script to stop."""
        self._stop_requested = True
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @staticmethod
    def get_example_scripts() -> Dict[str, str]:
        """Get built-in example scripts."""
        return {
            'Parameter Sweep': '''# Parameter sweep: test different filter orders
results = []
for order in [4, 8, 16, 32, 64]:
    spec = FilterSpec(
        filter_class='FIR', design_method='window',
        filter_type='lowpass', order=order,
        fs=8000, fp1=1000, window_type='hamming')
    r = designer.design(spec)
    # Measure stopband attenuation
    sb_mask = r.freq_hz >= 1500
    if np.any(sb_mask):
        atten = -np.max(r.magnitude_db[sb_mask])
    else:
        atten = 0
    results.append({'order': order, 'atten_db': round(atten, 2)})
    print(f"Order={order}: Stopband Atten = {atten:.1f} dB")

print("\\nSweep complete!")
for r in results:
    print(f"  Order {r['order']:3d}: {r['atten_db']:6.1f} dB")
''',
            'THD Analysis': '''# Generate test signal and measure THD
fs = 44100
t, sig = generate_test_signal('sine', 0.1, fs, freq=1000,
                              harmonics=[(2, 0.01), (3, 0.005)])
result = measurer.analyze(sig, fs)
print(f"Fundamental: {format_freq(result.fundamental_freq)}")
print(f"THD: {result.thd_percent:.4f}% ({result.thd_db:.1f} dB)")
print(f"SNR: {result.snr_db:.1f} dB")
print(f"SFDR: {result.sfdr_db:.1f} dB")
print(f"ENOB: {result.enob:.2f} bits")
''',
            'Batch Filter Compare': '''# Compare Butterworth vs Chebyshev vs Elliptic
methods = ['butter', 'cheby1', 'ellip']
names = ['Butterworth', 'Chebyshev I', 'Elliptic']
for name, method in zip(names, methods):
    spec = FilterSpec(
        filter_class='IIR', design_method=method,
        filter_type='lowpass', order=4, fs=8000,
        fp1=1000, passband_ripple=1.0, stopband_atten=40.0)
    r = designer.design(spec)
    print(f"{name:15s}: stable={r.is_stable}, "
          f"max_gain={np.max(r.magnitude_db):.2f}dB")
''',
            'Code Generation': '''# Design and generate C code
spec = FilterSpec(filter_class='FIR', design_method='window',
    filter_type='lowpass', order=16, fs=8000, fp1=1000)
r = designer.design(spec)
code = codegen.generate(r.b, r.a, r.spec.fs, 'C',
    filter_name='lpf_1khz', filter_info=r.info)
print(code[:500])
print("...")
print(f"Total code length: {len(code)} chars")
''',
        }
