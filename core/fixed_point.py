"""
Fixed-Point Analysis — 定点量化工具

Q-format conversion, coefficient quantization, noise estimation, SOS scaling.
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class FixedPointResult:
    """Fixed-point analysis result / 定点分析结果."""
    # Original coefficients
    b_float: np.ndarray = None
    a_float: np.ndarray = None
    
    # Quantized coefficients
    b_quant: np.ndarray = None
    a_quant: np.ndarray = None
    
    # Q-format info
    q_format: str = ''            # e.g., 'Q1.15', 'Q3.13'
    word_length: int = 16
    frac_bits: int = 15
    int_bits: int = 0
    
    # Error analysis
    coeff_error: np.ndarray = None       # Per-coefficient error
    max_coeff_error: float = 0.0
    mean_coeff_error: float = 0.0
    
    # Frequency response comparison
    freq_hz: np.ndarray = None
    mag_float_db: np.ndarray = None
    mag_quant_db: np.ndarray = None
    mag_error_db: np.ndarray = None
    
    # SOS
    sos_float: np.ndarray = None
    sos_quant: np.ndarray = None
    
    # Quantization noise
    quant_noise_power: float = 0.0
    quant_snr_db: float = 0.0
    
    info: str = ''


class FixedPointAnalyzer:
    """
    Fixed-Point Analyzer — 定点量化分析器
    
    Provides Q-format analysis, coefficient quantization, and error estimation.
    """
    
    COMMON_FORMATS = {
        'Q1.7':  (8, 7),
        'Q1.15': (16, 15),
        'Q1.31': (32, 31),
        'Q3.13': (16, 13),
        'Q5.11': (16, 11),
        'Q8.8':  (16, 8),
        'Q16.16': (32, 16),
    }
    
    def __init__(self):
        pass
    
    def suggest_q_format(self, coefficients: np.ndarray,
                         word_length: int = 16) -> Tuple[int, int, str]:
        """
        Suggest optimal Q-format for given coefficients.
        
        Args:
            coefficients: Filter coefficients
            word_length: Total word length in bits
        Returns:
            (int_bits, frac_bits, format_string)
        """
        max_val = np.max(np.abs(coefficients))
        if max_val == 0:
            max_val = 1.0
        
        # Need enough integer bits to represent max value + sign
        int_bits = max(1, int(np.ceil(np.log2(max_val + 1))) + 1)
        frac_bits = word_length - int_bits
        
        if frac_bits < 0:
            frac_bits = 0
            int_bits = word_length
        
        format_str = f"Q{int_bits}.{frac_bits}"
        return int_bits, frac_bits, format_str
    
    def quantize_coefficients(self, coefficients: np.ndarray,
                              word_length: int = 16,
                              frac_bits: int = None,
                              method: str = 'round') -> Tuple[np.ndarray, str]:
        """
        Quantize coefficients to fixed-point.
        
        Args:
            coefficients: Float coefficients
            word_length: Total bits
            frac_bits: Fractional bits (None = auto)
            method: 'round', 'truncate', or 'ceil'
        Returns:
            (quantized_coefficients, q_format_string)
        """
        if frac_bits is None:
            _, frac_bits, _ = self.suggest_q_format(coefficients, word_length)
        
        int_bits = word_length - frac_bits
        scale = 2.0 ** frac_bits
        
        # Max/min values
        max_val = (2 ** (word_length - 1) - 1) / scale
        min_val = -(2 ** (word_length - 1)) / scale
        
        # Quantize
        scaled = coefficients * scale
        if method == 'round':
            quantized_int = np.round(scaled)
        elif method == 'truncate':
            quantized_int = np.floor(scaled)
        elif method == 'ceil':
            quantized_int = np.ceil(scaled)
        else:
            quantized_int = np.round(scaled)
        
        # Clip to range
        quantized_int = np.clip(quantized_int,
                                -(2 ** (word_length - 1)),
                                2 ** (word_length - 1) - 1)
        
        quantized = quantized_int / scale
        format_str = f"Q{int_bits}.{frac_bits}"
        
        return quantized, format_str
    
    def analyze(self, b: np.ndarray, a: np.ndarray,
                fs: float = 1.0,
                word_length: int = 16,
                frac_bits: int = None,
                method: str = 'round',
                n_freq: int = 2048) -> FixedPointResult:
        """
        Comprehensive fixed-point analysis.
        
        Args:
            b: Numerator coefficients
            a: Denominator coefficients
            fs: Sampling rate
            word_length: Word length in bits
            frac_bits: Fractional bits (None = auto)
            method: Quantization method
            n_freq: Frequency response points
        Returns:
            FixedPointResult
        """
        result = FixedPointResult()
        result.b_float = b.copy()
        result.a_float = a.copy()
        result.word_length = word_length
        
        # Quantize b
        b_q, fmt_b = self.quantize_coefficients(b, word_length, frac_bits, method)
        result.b_quant = b_q
        
        # Quantize a
        a_q, fmt_a = self.quantize_coefficients(a, word_length, frac_bits, method)
        result.a_quant = a_q
        
        # Use suggested format for the larger of the two
        all_coeffs = np.concatenate([b, a])
        int_bits, fb, fmt = self.suggest_q_format(all_coeffs, word_length)
        result.q_format = fmt
        result.frac_bits = fb
        result.int_bits = int_bits
        
        # Coefficient errors
        b_err = np.abs(b - b_q)
        a_err = np.abs(a - a_q)
        result.coeff_error = np.concatenate([b_err, a_err])
        result.max_coeff_error = float(np.max(result.coeff_error))
        result.mean_coeff_error = float(np.mean(result.coeff_error))
        
        # Frequency response comparison
        w_f, h_f = signal.freqz(b, a, worN=n_freq, fs=fs)
        w_q, h_q = signal.freqz(b_q, a_q, worN=n_freq, fs=fs)
        
        result.freq_hz = w_f
        result.mag_float_db = 20.0 * np.log10(np.maximum(np.abs(h_f), 1e-12))
        result.mag_quant_db = 20.0 * np.log10(np.maximum(np.abs(h_q), 1e-12))
        result.mag_error_db = result.mag_quant_db - result.mag_float_db
        
        # Quantization noise estimation
        # Noise power = (LSB^2) / 12 per coefficient
        lsb = 1.0 / (2.0 ** fb)
        n_coeffs = len(b) + len(a) - 1  # -1 for a[0]
        result.quant_noise_power = n_coeffs * (lsb ** 2) / 12.0
        
        # Quantization SNR estimate
        signal_power = np.mean(np.abs(b) ** 2)
        if signal_power > 0 and result.quant_noise_power > 0:
            result.quant_snr_db = 10.0 * np.log10(
                signal_power / result.quant_noise_power)
        
        result.info = (f"{fmt}, {word_length}-bit, "
                       f"Max Err={result.max_coeff_error:.2e}, "
                       f"QSNR={result.quant_snr_db:.1f}dB")
        
        return result
    
    def decompose_sos(self, b: np.ndarray, a: np.ndarray,
                      word_length: int = 16,
                      frac_bits: int = None) -> FixedPointResult:
        """
        Decompose to second-order sections and quantize.
        
        Args:
            b: Numerator coefficients
            a: Denominator coefficients
            word_length: Word length
            frac_bits: Fractional bits
        Returns:
            FixedPointResult with SOS data
        """
        result = FixedPointResult()
        result.word_length = word_length
        result.b_float = b.copy()
        result.a_float = a.copy()
        
        try:
            sos = signal.tf2sos(b, a)
            result.sos_float = sos
            
            # Quantize each section
            sos_q = np.zeros_like(sos)
            for i in range(sos.shape[0]):
                b_sec = sos[i, :3]  # b0, b1, b2
                a_sec = sos[i, 3:]  # a0(=1), a1, a2
                
                b_sec_q, _ = self.quantize_coefficients(
                    b_sec, word_length, frac_bits)
                a_sec_q, _ = self.quantize_coefficients(
                    a_sec, word_length, frac_bits)
                
                sos_q[i, :3] = b_sec_q
                sos_q[i, 3:] = a_sec_q
            
            result.sos_quant = sos_q
            
            n_sections = sos.shape[0]
            _, fb, fmt = self.suggest_q_format(sos.ravel(), word_length)
            result.q_format = fmt
            result.frac_bits = fb
            result.int_bits = word_length - fb
            
            result.info = f"SOS: {n_sections} sections, {fmt}"
            
        except Exception as e:
            result.info = f"SOS decomposition failed: {str(e)}"
        
        return result
    
    def format_coefficients_hex(self, coefficients: np.ndarray,
                                 word_length: int = 16,
                                 frac_bits: int = 15) -> List[str]:
        """
        Format quantized coefficients as hex strings.
        
        Args:
            coefficients: Float coefficients
            word_length: Word length
            frac_bits: Fractional bits
        Returns:
            List of hex strings
        """
        scale = 2.0 ** frac_bits
        n_hex = (word_length + 3) // 4
        
        hex_vals = []
        for c in coefficients:
            val = int(np.round(c * scale))
            if val < 0:
                val = val + (1 << word_length)
            hex_vals.append(f"0x{val:0{n_hex}X}")
        
        return hex_vals
