"""
File Import — 数据导入模块

Supports WAV, CSV, MAT, BIN file import with auto-detection.
"""

import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class SignalData:
    """Imported signal data container / 信号数据容器."""
    data: np.ndarray = None          # Signal data (samples x channels)
    fs: float = 1.0                   # Sampling rate (Hz)
    n_channels: int = 1
    n_samples: int = 0
    bit_depth: int = 16
    duration: float = 0.0             # Duration in seconds
    filename: str = ''
    file_format: str = ''
    metadata: Dict = field(default_factory=dict)
    
    @property
    def channel_data(self) -> np.ndarray:
        """Get data as 1D array (first channel or mono)."""
        if self.data is None:
            return np.array([])
        if self.data.ndim == 1:
            return self.data
        return self.data[:, 0]


class FileImporter:
    """
    File Importer — 文件导入器
    
    Auto-detects file format and loads signal data.
    """
    
    SUPPORTED_FORMATS = {
        'WAV': ['.wav'],
        'CSV': ['.csv', '.txt', '.tsv'],
        'MAT': ['.mat'],
        'BIN': ['.bin', '.raw', '.dat'],
    }
    
    def __init__(self):
        pass
    
    def load(self, filepath: str, fs: float = None,
             dtype: str = 'float64', channels: int = 1,
             byte_order: str = 'little',
             delimiter: str = ',',
             skip_header: int = 0) -> SignalData:
        """
        Load signal data from file.
        
        Args:
            filepath: Path to file
            fs: Sampling rate (None = auto-detect from file)
            dtype: Data type for BIN files ('int16', 'float32', 'float64')
            channels: Number of channels for BIN files
            byte_order: Byte order for BIN ('little' or 'big')
            delimiter: Delimiter for CSV files
            skip_header: Lines to skip in CSV
        Returns:
            SignalData object
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext in self.SUPPORTED_FORMATS['WAV']:
            return self._load_wav(filepath, fs)
        elif ext in self.SUPPORTED_FORMATS['CSV']:
            return self._load_csv(filepath, fs, delimiter, skip_header)
        elif ext in self.SUPPORTED_FORMATS['MAT']:
            return self._load_mat(filepath, fs)
        elif ext in self.SUPPORTED_FORMATS['BIN']:
            return self._load_bin(filepath, fs, dtype, channels, byte_order)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_wav(self, filepath: str, fs: float = None) -> SignalData:
        """Load WAV file."""
        from scipy.io import wavfile
        
        sample_rate, data = wavfile.read(filepath)
        
        result = SignalData()
        result.filename = os.path.basename(filepath)
        result.file_format = 'WAV'
        result.fs = fs if fs is not None else float(sample_rate)
        
        # Convert to float64
        if data.dtype == np.int16:
            result.data = data.astype(np.float64) / 32768.0
            result.bit_depth = 16
        elif data.dtype == np.int32:
            result.data = data.astype(np.float64) / 2147483648.0
            result.bit_depth = 32
        elif data.dtype == np.uint8:
            result.data = (data.astype(np.float64) - 128.0) / 128.0
            result.bit_depth = 8
        else:
            result.data = data.astype(np.float64)
            result.bit_depth = 0
        
        if result.data.ndim == 1:
            result.n_channels = 1
            result.n_samples = len(result.data)
        else:
            result.n_channels = result.data.shape[1]
            result.n_samples = result.data.shape[0]
        
        result.duration = result.n_samples / result.fs
        result.metadata['original_dtype'] = str(data.dtype)
        result.metadata['original_sample_rate'] = sample_rate
        
        return result
    
    def _load_csv(self, filepath: str, fs: float = None,
                  delimiter: str = ',', skip_header: int = 0) -> SignalData:
        """Load CSV/TXT file."""
        result = SignalData()
        result.filename = os.path.basename(filepath)
        result.file_format = 'CSV'
        
        try:
            data = np.loadtxt(filepath, delimiter=delimiter,
                              skiprows=skip_header)
        except Exception:
            # Try with tab delimiter
            try:
                data = np.loadtxt(filepath, delimiter='\t',
                                  skiprows=skip_header)
            except Exception:
                data = np.genfromtxt(filepath, delimiter=delimiter,
                                     skip_header=skip_header,
                                     filling_values=0.0)
        
        result.data = data.astype(np.float64)
        
        if result.data.ndim == 1:
            result.n_channels = 1
            result.n_samples = len(result.data)
        else:
            # Check if first column is time
            if result.data.shape[1] >= 2:
                # Heuristic: if first column is monotonically increasing, treat as time
                col0 = result.data[:, 0]
                if np.all(np.diff(col0) > 0) and fs is None:
                    dt = np.mean(np.diff(col0))
                    fs = 1.0 / dt if dt > 0 else 1.0
                    result.data = result.data[:, 1:]  # Remove time column
            
            if result.data.ndim == 1:
                result.n_channels = 1
                result.n_samples = len(result.data)
            else:
                result.n_channels = result.data.shape[1]
                result.n_samples = result.data.shape[0]
        
        result.fs = fs if fs is not None else 1.0
        result.duration = result.n_samples / result.fs
        
        return result
    
    def _load_mat(self, filepath: str, fs: float = None) -> SignalData:
        """Load MATLAB .mat file."""
        from scipy.io import loadmat
        
        result = SignalData()
        result.filename = os.path.basename(filepath)
        result.file_format = 'MAT'
        
        mat_data = loadmat(filepath)
        
        # Find the main data variable (skip meta variables)
        data_var = None
        fs_var = None
        for key, value in mat_data.items():
            if key.startswith('__'):
                continue
            if isinstance(value, np.ndarray):
                if key.lower() in ('fs', 'sample_rate', 'sr', 'samplerate'):
                    fs_var = float(value.ravel()[0])
                elif data_var is None or value.size > data_var.size:
                    data_var = value
        
        if data_var is not None:
            result.data = data_var.astype(np.float64)
            if result.data.ndim == 1:
                result.n_channels = 1
                result.n_samples = len(result.data)
            else:
                # Ensure samples are along axis 0
                if result.data.shape[0] < result.data.shape[1]:
                    result.data = result.data.T
                result.n_channels = result.data.shape[1]
                result.n_samples = result.data.shape[0]
        
        if fs is not None:
            result.fs = fs
        elif fs_var is not None:
            result.fs = fs_var
        else:
            result.fs = 1.0
        
        result.duration = result.n_samples / result.fs
        result.metadata['mat_keys'] = [k for k in mat_data.keys()
                                        if not k.startswith('__')]
        
        return result
    
    def _load_bin(self, filepath: str, fs: float = None,
                  dtype: str = 'float64', channels: int = 1,
                  byte_order: str = 'little') -> SignalData:
        """Load raw binary file."""
        result = SignalData()
        result.filename = os.path.basename(filepath)
        result.file_format = 'BIN'
        
        # Map dtype string
        dt_map = {
            'int8': np.int8, 'uint8': np.uint8,
            'int16': np.int16, 'uint16': np.uint16,
            'int32': np.int32, 'uint32': np.uint32,
            'float32': np.float32, 'float64': np.float64,
        }
        np_dtype = dt_map.get(dtype, np.float64)
        
        if byte_order == 'big':
            np_dtype = np.dtype(np_dtype).newbyteorder('>')
        
        raw = np.fromfile(filepath, dtype=np_dtype)
        
        # Normalize integer types
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            result.data = raw.astype(np.float64) / float(info.max)
            result.bit_depth = info.bits
        else:
            result.data = raw.astype(np.float64)
        
        if channels > 1:
            n = len(result.data)
            n_trim = n - (n % channels)
            result.data = result.data[:n_trim].reshape(-1, channels)
            result.n_channels = channels
            result.n_samples = result.data.shape[0]
        else:
            result.n_channels = 1
            result.n_samples = len(result.data)
        
        result.fs = fs if fs is not None else 1.0
        result.duration = result.n_samples / result.fs
        result.metadata['original_dtype'] = dtype
        
        return result
    
    @staticmethod
    def get_supported_extensions() -> str:
        """Get file dialog filter string."""
        return (
            "All Supported (*.wav *.csv *.txt *.mat *.bin *.raw *.dat);;"
            "WAV Files (*.wav);;"
            "CSV/Text Files (*.csv *.txt *.tsv);;"
            "MATLAB Files (*.mat);;"
            "Binary Files (*.bin *.raw *.dat);;"
            "All Files (*.*)"
        )
