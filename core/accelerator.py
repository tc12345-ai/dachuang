"""
GPU/Multi-thread Accelerator — GPU/多线程加速模块

Provides accelerated FFT, filtering, and visualization using
thread pools, multiprocessing, and optional CuPy GPU acceleration.
"""

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple
import time


class Accelerator:
    """
    Computation Accelerator — 计算加速器
    
    Provides multi-threaded and optional GPU-accelerated DSP operations.
    Falls back to NumPy/SciPy if GPU libraries are not available.
    """
    
    def __init__(self, max_workers: int = None, use_gpu: bool = True):
        """
        Args:
            max_workers: Max thread pool size (None = auto)
            use_gpu: Try to use GPU if available
        """
        import os
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._gpu_available = False
        self._cupy = None
        
        if use_gpu:
            self._try_init_gpu()
    
    def _try_init_gpu(self):
        """Try to initialize GPU acceleration."""
        try:
            import cupy as cp
            # Quick test
            x = cp.array([1, 2, 3])
            _ = cp.fft.fft(x)
            self._cupy = cp
            self._gpu_available = True
            print("GPU acceleration enabled (CuPy)")
        except ImportError:
            self._gpu_available = False
        except Exception:
            self._gpu_available = False
    
    @property
    def gpu_available(self) -> bool:
        return self._gpu_available
    
    def get_info(self) -> dict:
        """Get accelerator information."""
        info = {
            'max_workers': self.max_workers,
            'gpu_available': self._gpu_available,
            'backend': 'CuPy' if self._gpu_available else 'NumPy/SciPy',
        }
        if self._gpu_available:
            try:
                device = self._cupy.cuda.Device()
                info['gpu_name'] = device.attributes.get('name', 'Unknown')
                info['gpu_memory'] = f"{device.mem_info[1] / 1e9:.1f} GB"
            except Exception:
                pass
        return info
    
    # === Accelerated FFT ===
    
    def fft(self, data: np.ndarray, n: int = None) -> np.ndarray:
        """
        Accelerated FFT.
        
        Args:
            data: Input signal
            n: FFT length
        Returns:
            FFT result (complex array)
        """
        if self._gpu_available:
            try:
                cp = self._cupy
                d_gpu = cp.asarray(data)
                result = cp.fft.fft(d_gpu, n=n)
                return cp.asnumpy(result)
            except Exception:
                pass
        
        from scipy.fft import fft
        return fft(data, n=n)
    
    def ifft(self, data: np.ndarray, n: int = None) -> np.ndarray:
        """Accelerated inverse FFT."""
        if self._gpu_available:
            try:
                cp = self._cupy
                d_gpu = cp.asarray(data)
                result = cp.fft.ifft(d_gpu, n=n)
                return cp.asnumpy(result)
            except Exception:
                pass
        
        from scipy.fft import ifft
        return ifft(data, n=n)
    
    def rfft(self, data: np.ndarray, n: int = None) -> np.ndarray:
        """Accelerated real FFT."""
        if self._gpu_available:
            try:
                cp = self._cupy
                d_gpu = cp.asarray(data)
                result = cp.fft.rfft(d_gpu, n=n)
                return cp.asnumpy(result)
            except Exception:
                pass
        
        from scipy.fft import rfft
        return rfft(data, n=n)
    
    # === Parallel Processing ===
    
    def parallel_fft_batch(self, signals: List[np.ndarray],
                            n: int = None) -> List[np.ndarray]:
        """
        Compute FFT on multiple signals in parallel.
        
        Args:
            signals: List of signals
            n: FFT length
        Returns:
            List of FFT results
        """
        from scipy.fft import fft
        
        def compute_one(sig):
            return fft(sig, n=n)
        
        futures = [self.thread_pool.submit(compute_one, s) for s in signals]
        return [f.result() for f in futures]
    
    def parallel_filter(self, signals: List[np.ndarray],
                        b: np.ndarray, a: np.ndarray) -> List[np.ndarray]:
        """
        Apply filter to multiple signals in parallel.
        
        Args:
            signals: List of input signals
            b, a: Filter coefficients
        Returns:
            List of filtered signals
        """
        from scipy.signal import lfilter
        
        def filter_one(sig):
            return lfilter(b, a, sig)
        
        futures = [self.thread_pool.submit(filter_one, s) for s in signals]
        return [f.result() for f in futures]
    
    def parallel_psd(self, signals: List[np.ndarray], fs: float,
                     nperseg: int = 1024) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute PSD for multiple signals in parallel.
        
        Args:
            signals: List of input signals
            fs: Sampling rate
            nperseg: Segment length
        Returns:
            List of (freq, psd) tuples
        """
        from scipy.signal import welch
        
        def psd_one(sig):
            return welch(sig, fs=fs, nperseg=nperseg)
        
        futures = [self.thread_pool.submit(psd_one, s) for s in signals]
        return [f.result() for f in futures]
    
    # === Batch File Processing ===
    
    def batch_process(self, file_list: List[str],
                      process_func: Callable,
                      progress_callback: Callable = None) -> List:
        """
        Process multiple files in parallel.
        
        Args:
            file_list: List of file paths
            process_func: Function(filepath) -> result
            progress_callback: Function(completed, total) for progress
        Returns:
            List of results
        """
        results = [None] * len(file_list)
        completed = [0]
        lock = threading.Lock()
        
        def process_one(i, path):
            result = process_func(path)
            results[i] = result
            with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], len(file_list))
        
        futures = []
        for i, path in enumerate(file_list):
            f = self.thread_pool.submit(process_one, i, path)
            futures.append(f)
        
        for f in futures:
            f.result()  # Wait for completion
        
        return results
    
    # === Timed Execution ===
    
    def benchmark(self, func: Callable, *args, n_runs: int = 10,
                  **kwargs) -> dict:
        """
        Benchmark a function.
        
        Args:
            func: Function to benchmark
            *args: Arguments
            n_runs: Number of runs
        Returns:
            Dict with timing statistics
        """
        times = []
        result = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'n_runs': n_runs,
            'result': result,
        }
    
    def shutdown(self):
        """Shutdown thread pool."""
        self.thread_pool.shutdown(wait=False)
