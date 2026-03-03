"""
Filter Design Engine — 滤波器设计引擎

Supports FIR and IIR filter design with multiple methods:
- FIR: Window method, Frequency sampling, Parks-McClellan (equiripple)
- IIR: Butterworth, Chebyshev I/II, Elliptic, Bessel
- Types: Lowpass, Highpass, Bandpass, Bandstop, Notch
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class FilterSpec:
    """Filter specification / 滤波器规格."""
    filter_class: str = 'FIR'          # 'FIR' or 'IIR'
    design_method: str = 'window'       # 'window','freq_sampling','equiripple','butter','cheby1','cheby2','ellip','bessel'
    filter_type: str = 'lowpass'        # 'lowpass','highpass','bandpass','bandstop','notch'
    order: int = 32                     # Filter order
    fs: float = 8000.0                  # Sampling rate (Hz)
    
    # Frequency specs (Hz)
    fp1: float = 1000.0                 # Passband edge 1
    fp2: float = 2000.0                 # Passband edge 2 (for bandpass/bandstop)
    fs1: float = 1500.0                 # Stopband edge 1
    fs2: float = 500.0                  # Stopband edge 2 (for bandpass/bandstop)
    
    # Performance specs (dB)
    passband_ripple: float = 1.0        # Max passband ripple (dB)
    stopband_atten: float = 60.0        # Min stopband attenuation (dB)
    
    # FIR-specific
    window_type: str = 'hamming'        # Window function
    kaiser_beta: float = 5.0            # Kaiser window beta
    
    # Notch specific
    notch_freq: float = 50.0            # Notch center frequency (Hz)
    notch_bw: float = 10.0              # Notch bandwidth (Hz)


@dataclass
class FilterResult:
    """Filter design result / 滤波器设计结果."""
    b: np.ndarray = None                # Numerator coefficients
    a: np.ndarray = None                # Denominator coefficients (1.0 for FIR)
    sos: Optional[np.ndarray] = None    # Second-order sections
    zeros: np.ndarray = None            # Zeros
    poles: np.ndarray = None            # Poles
    gain: float = 1.0                   # Gain
    order: int = 0                      # Actual order
    
    # Frequency response
    freq_hz: np.ndarray = None          # Frequency vector (Hz)
    magnitude_db: np.ndarray = None     # Magnitude response (dB)
    phase_rad: np.ndarray = None        # Phase response (radians)
    phase_deg: np.ndarray = None        # Phase response (degrees)
    group_delay: np.ndarray = None      # Group delay (samples)
    gd_freq_hz: np.ndarray = None      # Group delay frequency vec (Hz)
    
    # Time responses
    impulse_response: np.ndarray = None
    step_response: np.ndarray = None
    impulse_t: np.ndarray = None
    step_t: np.ndarray = None
    
    # Status
    is_stable: bool = True
    spec: FilterSpec = None
    info: str = ""


class FilterDesigner:
    """
    Digital Filter Designer — 数字滤波器设计器
    
    Supports comprehensive FIR/IIR filter design with multiple methods.
    """
    
    # Available window functions
    WINDOWS = ['hamming', 'hann', 'blackman', 'bartlett', 'kaiser',
               'rectangular', 'flattop', 'nuttall', 'blackmanharris']
    
    # Available design methods
    FIR_METHODS = ['window', 'freq_sampling', 'equiripple']
    IIR_METHODS = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    
    FILTER_TYPES = ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch']
    
    def __init__(self, n_freq_points=2048):
        """
        Initialize designer.
        
        Args:
            n_freq_points: Number of frequency points for response calculation
        """
        self.n_freq_points = n_freq_points
    
    def design(self, spec: FilterSpec) -> FilterResult:
        """
        Design a filter based on specification.
        
        Args:
            spec: FilterSpec with design parameters
        Returns:
            FilterResult with coefficients, responses, and analysis
        """
        result = FilterResult(spec=spec)
        
        try:
            if spec.filter_type == 'notch':
                self._design_notch(spec, result)
            elif spec.filter_class == 'FIR':
                self._design_fir(spec, result)
            elif spec.filter_class == 'IIR':
                self._design_iir(spec, result)
            else:
                raise ValueError(f"Unknown filter class: {spec.filter_class}")
            
            # Compute responses
            self._compute_frequency_response(spec, result)
            self._compute_time_responses(spec, result)
            self._compute_zpk(result)
            self._check_stability(result)
            
        except Exception as e:
            result.info = f"Design error: {str(e)}"
            # Provide trivial pass-through on error
            result.b = np.array([1.0])
            result.a = np.array([1.0])
            self._compute_frequency_response(spec, result)
        
        return result
    
    def _design_fir(self, spec: FilterSpec, result: FilterResult):
        """Design FIR filter."""
        nyq = spec.fs / 2.0
        order = spec.order
        
        # Determine cutoff frequencies
        if spec.filter_type in ('lowpass', 'highpass'):
            cutoff = spec.fp1 / nyq
            cutoff = np.clip(cutoff, 0.001, 0.999)
        elif spec.filter_type in ('bandpass', 'bandstop'):
            cutoff = [spec.fp1 / nyq, spec.fp2 / nyq]
            cutoff = np.clip(cutoff, 0.001, 0.999)
        
        if spec.design_method == 'window':
            # Window method using firwin
            window = spec.window_type
            if window == 'kaiser':
                window = ('kaiser', spec.kaiser_beta)
            elif window == 'rectangular':
                window = 'boxcar'
            
            try:
                if spec.filter_type in ('lowpass', 'highpass'):
                    pass_zero = (spec.filter_type == 'lowpass')
                    b = signal.firwin(order + 1, cutoff, window=window,
                                      pass_zero=pass_zero, fs=spec.fs)
                else:
                    pass_zero = (spec.filter_type == 'bandstop')
                    b = signal.firwin(order + 1, [spec.fp1, spec.fp2],
                                      window=window, pass_zero=pass_zero,
                                      fs=spec.fs)
            except Exception:
                # Fallback: ensure valid numtaps
                numtaps = order + 1
                if spec.filter_type in ('bandpass',) and numtaps % 2 == 0:
                    numtaps += 1
                if spec.filter_type in ('lowpass', 'highpass'):
                    pass_zero = (spec.filter_type == 'lowpass')
                    b = signal.firwin(numtaps, cutoff, window=window,
                                      pass_zero=pass_zero, fs=spec.fs)
                else:
                    pass_zero = (spec.filter_type == 'bandstop')
                    b = signal.firwin(numtaps, [spec.fp1, spec.fp2],
                                      window=window, pass_zero=pass_zero,
                                      fs=spec.fs)
            
            result.b = b
            result.a = np.array([1.0])
            result.order = len(b) - 1
            result.info = f"FIR Window ({spec.window_type}), Order={result.order}"
            
        elif spec.design_method == 'freq_sampling':
            # Frequency sampling using firwin2
            if spec.filter_type == 'lowpass':
                freq = [0, spec.fp1/(nyq), spec.fs1/(nyq), 1.0]
                freq = np.clip(freq, 0, 1)
                gain = [1, 1, 0, 0]
            elif spec.filter_type == 'highpass':
                freq = [0, spec.fs1/(nyq), spec.fp1/(nyq), 1.0]
                freq = np.clip(freq, 0, 1)
                gain = [0, 0, 1, 1]
            elif spec.filter_type == 'bandpass':
                f1n = max(0.001, spec.fs2 / nyq)
                f2n = min(0.999, spec.fp1 / nyq)
                f3n = min(0.999, spec.fp2 / nyq)
                f4n = min(0.999, spec.fs1 / nyq)
                freq = [0, f1n, f2n, f3n, f4n, 1.0]
                gain = [0, 0, 1, 1, 0, 0]
            elif spec.filter_type == 'bandstop':
                f1n = max(0.001, spec.fp1 / nyq)
                f2n = min(0.999, spec.fs2 / nyq)
                f3n = min(0.999, spec.fs1 / nyq)
                f4n = min(0.999, spec.fp2 / nyq)
                freq = [0, f1n, f2n, f3n, f4n, 1.0]
                gain = [1, 1, 0, 0, 1, 1]
            
            # Ensure frequencies are monotonically increasing
            freq = np.array(freq)
            for i in range(1, len(freq)):
                if freq[i] <= freq[i-1]:
                    freq[i] = freq[i-1] + 0.001
            freq = np.clip(freq, 0, 1)
            
            numtaps = order + 1
            if numtaps < 2:
                numtaps = 2
            b = signal.firwin2(numtaps, freq, gain)
            result.b = b
            result.a = np.array([1.0])
            result.order = len(b) - 1
            result.info = f"FIR Freq Sampling, Order={result.order}"
            
        elif spec.design_method == 'equiripple':
            # Parks-McClellan (remez)
            numtaps = order + 1
            if numtaps < 4:
                numtaps = 4
            
            if spec.filter_type == 'lowpass':
                bands = [0, spec.fp1, spec.fs1, nyq]
                desired = [1, 0]
            elif spec.filter_type == 'highpass':
                bands = [0, spec.fs1, spec.fp1, nyq]
                desired = [0, 1]
                if numtaps % 2 == 0:
                    numtaps += 1
            elif spec.filter_type == 'bandpass':
                bands = [0, spec.fs2, spec.fp1, spec.fp2, spec.fs1, nyq]
                desired = [0, 1, 0]
            elif spec.filter_type == 'bandstop':
                bands = [0, spec.fp1, spec.fs2, spec.fs1, spec.fp2, nyq]
                desired = [1, 0, 1]
                if numtaps % 2 == 0:
                    numtaps += 1
            
            # Validate bands
            bands = np.array(bands)
            for i in range(1, len(bands)):
                if bands[i] <= bands[i-1]:
                    bands[i] = bands[i-1] + 1.0
            
            # Weight: passband vs stopband
            weight_ratio = 10 ** (spec.stopband_atten / 20.0)
            weights = []
            for d in desired:
                if d > 0:
                    weights.append(1.0)
                else:
                    weights.append(1.0 / weight_ratio)
            
            b = signal.remez(numtaps, bands, desired, weight=weights, fs=spec.fs)
            result.b = b
            result.a = np.array([1.0])
            result.order = len(b) - 1
            result.info = f"FIR Equiripple (Parks-McClellan), Order={result.order}"
    
    def _design_iir(self, spec: FilterSpec, result: FilterResult):
        """Design IIR filter."""
        nyq = spec.fs / 2.0
        
        # Determine filter type string for scipy
        if spec.filter_type in ('lowpass', 'highpass'):
            Wn = spec.fp1 / nyq
            Wn = np.clip(Wn, 0.001, 0.999)
            btype = spec.filter_type
        elif spec.filter_type in ('bandpass', 'bandstop'):
            Wn = [spec.fp1 / nyq, spec.fp2 / nyq]
            Wn = np.clip(Wn, 0.001, 0.999)
            btype = 'bandpass' if spec.filter_type == 'bandpass' else 'bandstop'
        
        order = max(1, spec.order)
        
        if spec.design_method == 'butter':
            b, a = signal.butter(order, Wn, btype=btype)
            try:
                sos = signal.butter(order, Wn, btype=btype, output='sos')
                result.sos = sos
            except Exception:
                pass
            result.info = f"IIR Butterworth, Order={order}"
            
        elif spec.design_method == 'cheby1':
            rp = max(0.01, spec.passband_ripple)
            b, a = signal.cheby1(order, rp, Wn, btype=btype)
            try:
                sos = signal.cheby1(order, rp, Wn, btype=btype, output='sos')
                result.sos = sos
            except Exception:
                pass
            result.info = f"IIR Chebyshev Type I, Order={order}, Rp={rp}dB"
            
        elif spec.design_method == 'cheby2':
            rs = max(1.0, spec.stopband_atten)
            b, a = signal.cheby2(order, rs, Wn, btype=btype)
            try:
                sos = signal.cheby2(order, rs, Wn, btype=btype, output='sos')
                result.sos = sos
            except Exception:
                pass
            result.info = f"IIR Chebyshev Type II, Order={order}, Rs={rs}dB"
            
        elif spec.design_method == 'ellip':
            rp = max(0.01, spec.passband_ripple)
            rs = max(1.0, spec.stopband_atten)
            b, a = signal.ellip(order, rp, rs, Wn, btype=btype)
            try:
                sos = signal.ellip(order, rp, rs, Wn, btype=btype, output='sos')
                result.sos = sos
            except Exception:
                pass
            result.info = f"IIR Elliptic, Order={order}, Rp={rp}dB, Rs={rs}dB"
            
        elif spec.design_method == 'bessel':
            b, a = signal.bessel(order, Wn, btype=btype, norm='phase')
            result.info = f"IIR Bessel, Order={order}"
        
        else:
            raise ValueError(f"Unknown IIR method: {spec.design_method}")
        
        result.b = b
        result.a = a
        result.order = order
    
    def _design_notch(self, spec: FilterSpec, result: FilterResult):
        """Design notch filter."""
        w0 = spec.notch_freq / (spec.fs / 2.0)
        w0 = np.clip(w0, 0.001, 0.999)
        Q = spec.notch_freq / max(spec.notch_bw, 0.1)
        
        b, a = signal.iirnotch(w0, Q)
        result.b = b
        result.a = a
        result.order = max(len(b), len(a)) - 1
        result.info = f"Notch Filter, f0={spec.notch_freq}Hz, BW={spec.notch_bw}Hz, Q={Q:.1f}"
    
    def _compute_frequency_response(self, spec: FilterSpec, result: FilterResult):
        """Compute frequency response."""
        w, h = signal.freqz(result.b, result.a, worN=self.n_freq_points, fs=spec.fs)
        
        result.freq_hz = w
        mag = np.abs(h)
        mag = np.maximum(mag, 1e-12)
        result.magnitude_db = 20.0 * np.log10(mag)
        result.phase_rad = np.unwrap(np.angle(h))
        result.phase_deg = np.degrees(result.phase_rad)
        
        # Group delay
        try:
            gd_w, gd = signal.group_delay((result.b, result.a),
                                           w=self.n_freq_points, fs=spec.fs)
            result.group_delay = gd
            result.gd_freq_hz = gd_w
        except Exception:
            result.group_delay = np.zeros(self.n_freq_points)
            result.gd_freq_hz = result.freq_hz
    
    def _compute_time_responses(self, spec: FilterSpec, result: FilterResult):
        """Compute impulse and step responses."""
        n_points = max(128, result.order * 4)
        
        # Impulse response
        imp = np.zeros(n_points)
        imp[0] = 1.0
        try:
            if result.sos is not None:
                result.impulse_response = signal.sosfilt(result.sos, imp)
            else:
                result.impulse_response = signal.lfilter(result.b, result.a, imp)
        except Exception:
            result.impulse_response = np.zeros(n_points)
        result.impulse_t = np.arange(n_points) / spec.fs
        
        # Step response
        step_input = np.ones(n_points)
        try:
            if result.sos is not None:
                result.step_response = signal.sosfilt(result.sos, step_input)
            else:
                result.step_response = signal.lfilter(result.b, result.a, step_input)
        except Exception:
            result.step_response = np.zeros(n_points)
        result.step_t = np.arange(n_points) / spec.fs
    
    def _compute_zpk(self, result: FilterResult):
        """Compute zeros, poles, gain."""
        try:
            z, p, k = signal.tf2zpk(result.b, result.a)
            result.zeros = z
            result.poles = p
            result.gain = k
        except Exception:
            result.zeros = np.array([])
            result.poles = np.array([])
            result.gain = 1.0
    
    def _check_stability(self, result: FilterResult):
        """Check filter stability."""
        if result.poles is not None and len(result.poles) > 0:
            max_pole_mag = np.max(np.abs(result.poles))
            result.is_stable = max_pole_mag < 1.0
            if not result.is_stable:
                result.info += f" | UNSTABLE (max|pole|={max_pole_mag:.4f})"
        else:
            result.is_stable = True
    
    @staticmethod
    def estimate_order(spec: FilterSpec) -> int:
        """
        Estimate required filter order for given specs.
        
        Returns:
            Estimated minimum order
        """
        nyq = spec.fs / 2.0
        
        if spec.filter_class == 'FIR':
            if spec.design_method == 'equiripple':
                try:
                    if spec.filter_type in ('lowpass', 'highpass'):
                        wp = spec.fp1 / nyq
                        ws = spec.fs1 / nyq
                    else:
                        wp = spec.fp1 / nyq
                        ws = spec.fs1 / nyq
                    delta_p = 10 ** (-spec.passband_ripple / 20)
                    delta_s = 10 ** (-spec.stopband_atten / 20)
                    n = signal.remezord([spec.fp1, spec.fs1],
                                        [1, 0],
                                        [1 - delta_p, delta_s],
                                        fs=spec.fs)[0]
                    return max(4, int(n))
                except Exception:
                    pass
            # Kaiser formula estimate
            delta_s = 10 ** (-spec.stopband_atten / 20.0)
            A = -20.0 * np.log10(delta_s)
            df = abs(spec.fs1 - spec.fp1) / spec.fs
            if df > 0:
                n = int(np.ceil((A - 7.95) / (14.36 * df)))
                return max(4, n)
            return 32
            
        elif spec.filter_class == 'IIR':
            try:
                if spec.filter_type in ('lowpass', 'highpass'):
                    wp = spec.fp1 / nyq
                    ws = spec.fs1 / nyq
                else:
                    wp = [spec.fp1 / nyq, spec.fp2 / nyq]
                    ws = [spec.fs2 / nyq, spec.fs1 / nyq]
                
                rp = spec.passband_ripple
                rs = spec.stopband_atten
                
                if spec.design_method == 'butter':
                    n, _ = signal.buttord(wp, ws, rp, rs)
                elif spec.design_method == 'cheby1':
                    n, _ = signal.cheb1ord(wp, ws, rp, rs)
                elif spec.design_method == 'cheby2':
                    n, _ = signal.cheb2ord(wp, ws, rp, rs)
                elif spec.design_method == 'ellip':
                    n, _ = signal.ellipord(wp, ws, rp, rs)
                else:
                    n = spec.order
                return max(1, int(n))
            except Exception:
                return spec.order
        
        return spec.order
    
    def apply_filter(self, data: np.ndarray, result: FilterResult) -> np.ndarray:
        """
        Apply designed filter to data.
        
        Args:
            data: Input signal
            result: FilterResult from design()
        Returns:
            Filtered signal
        """
        if result.sos is not None:
            return signal.sosfilt(result.sos, data)
        else:
            return signal.lfilter(result.b, result.a, data)
    
    def apply_filtfilt(self, data: np.ndarray, result: FilterResult) -> np.ndarray:
        """
        Apply zero-phase filtering (forward-backward).
        
        Args:
            data: Input signal
            result: FilterResult from design()
        Returns:
            Zero-phase filtered signal
        """
        if result.sos is not None:
            return signal.sosfiltfilt(result.sos, data)
        else:
            return signal.filtfilt(result.b, result.a, data)
