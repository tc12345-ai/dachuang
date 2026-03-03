"""
Real-time Streaming Engine — 实时流采集模块

Supports soundcard (PyAudio), DAQ, serial port, and SDR input.
Ring buffer, frame-based processing, low-latency mode.
"""

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List


@dataclass
class StreamConfig:
    """Stream configuration / 流配置."""
    source_type: str = 'soundcard'      # 'soundcard', 'serial', 'simulated'
    sample_rate: float = 44100.0
    channels: int = 1
    chunk_size: int = 1024              # Samples per frame
    buffer_size: int = 16               # Number of chunks in ring buffer
    bit_depth: int = 16
    device_index: int = 0
    
    # Serial port settings
    serial_port: str = 'COM3'
    serial_baud: int = 115200
    serial_dtype: str = 'int16'
    
    # SDR settings (placeholder)
    sdr_freq: float = 100e6             # Center frequency Hz
    sdr_gain: float = 20.0              # Gain dB


class RingBuffer:
    """
    Thread-safe ring buffer for real-time audio processing.
    环形缓冲区
    """
    
    def __init__(self, max_frames: int, frame_size: int, channels: int = 1):
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.channels = channels
        self.buffer = np.zeros((max_frames * frame_size, channels))
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
        self.frames_available = 0
    
    def write(self, data: np.ndarray):
        """Write a frame of data into the buffer."""
        with self.lock:
            n = len(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            total = self.buffer.shape[0]
            end = self.write_pos + n
            
            if end <= total:
                self.buffer[self.write_pos:end, :data.shape[1]] = data
            else:
                first = total - self.write_pos
                self.buffer[self.write_pos:total, :data.shape[1]] = data[:first]
                remainder = n - first
                self.buffer[:remainder, :data.shape[1]] = data[first:]
            
            self.write_pos = end % total
            self.frames_available = min(self.frames_available + 1, self.max_frames)
    
    def read(self, n_samples: int) -> Optional[np.ndarray]:
        """Read n_samples from the buffer."""
        with self.lock:
            if self.frames_available <= 0:
                return None
            
            total = self.buffer.shape[0]
            end = self.read_pos + n_samples
            
            if end <= total:
                data = self.buffer[self.read_pos:end].copy()
            else:
                first = total - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:total],
                    self.buffer[:n_samples - first]
                ])
            
            self.read_pos = end % total
            self.frames_available -= 1
            return data
    
    def get_latest(self, n_samples: int) -> np.ndarray:
        """Get the most recent n_samples without consuming."""
        with self.lock:
            total = self.buffer.shape[0]
            start = (self.write_pos - n_samples) % total
            if start + n_samples <= total:
                return self.buffer[start:start + n_samples].copy()
            else:
                first = total - start
                return np.concatenate([
                    self.buffer[start:total],
                    self.buffer[:n_samples - first]
                ])
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer[:] = 0
            self.write_pos = 0
            self.read_pos = 0
            self.frames_available = 0


class RealtimeStream:
    """
    Real-time Stream Manager — 实时流管理器
    
    Manages audio/data capture from various sources with
    ring buffering and frame-based callback processing.
    """
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self.ring_buffer = None
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=32)
        self._callbacks: List[Callable] = []
        self._pyaudio_stream = None
        self._pyaudio_instance = None
        self._serial_conn = None
    
    def add_callback(self, callback: Callable):
        """
        Add a frame processing callback.
        
        Args:
            callback: Function(frame_data: np.ndarray, fs: float) -> None
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start(self):
        """Start streaming."""
        if self.is_running:
            return
        
        self.ring_buffer = RingBuffer(
            self.config.buffer_size,
            self.config.chunk_size,
            self.config.channels)
        
        self._stop_event.clear()
        self.is_running = True
        
        if self.config.source_type == 'soundcard':
            self._start_soundcard()
        elif self.config.source_type == 'serial':
            self._start_serial()
        elif self.config.source_type == 'simulated':
            self._start_simulated()
        
        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop streaming."""
        self._stop_event.set()
        self.is_running = False
        
        # Stop source
        if self._pyaudio_stream:
            try:
                self._pyaudio_stream.stop_stream()
                self._pyaudio_stream.close()
            except Exception:
                pass
            self._pyaudio_stream = None
        
        if self._pyaudio_instance:
            try:
                self._pyaudio_instance.terminate()
            except Exception:
                pass
            self._pyaudio_instance = None
        
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_latest_data(self, n_samples: int = None) -> np.ndarray:
        """Get latest data from ring buffer."""
        if self.ring_buffer is None:
            return np.array([])
        if n_samples is None:
            n_samples = self.config.chunk_size * 4
        return self.ring_buffer.get_latest(n_samples)
    
    def _start_soundcard(self):
        """Start soundcard capture using PyAudio."""
        try:
            import pyaudio
            self._pyaudio_instance = pyaudio.PyAudio()
            
            fmt_map = {8: pyaudio.paInt8, 16: pyaudio.paInt16,
                       24: pyaudio.paInt24, 32: pyaudio.paFloat32}
            fmt = fmt_map.get(self.config.bit_depth, pyaudio.paInt16)
            
            def audio_callback(in_data, frame_count, time_info, status):
                if self.config.bit_depth == 32:
                    data = np.frombuffer(in_data, dtype=np.float32)
                else:
                    data = np.frombuffer(in_data, dtype=np.int16)
                    data = data.astype(np.float64) / 32768.0
                
                if self.config.channels > 1:
                    data = data.reshape(-1, self.config.channels)
                
                self.ring_buffer.write(data)
                self._frame_queue.put(data.copy(), block=False)
                
                import pyaudio as pa
                return (None, pa.paContinue)
            
            self._pyaudio_stream = self._pyaudio_instance.open(
                format=fmt,
                channels=self.config.channels,
                rate=int(self.config.sample_rate),
                input=True,
                input_device_index=self.config.device_index if self.config.device_index >= 0 else None,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=audio_callback)
            
            self._pyaudio_stream.start_stream()
            
        except ImportError:
            print("Warning: PyAudio not installed. Using simulated stream.")
            print("Install with: pip install pyaudio")
            self._start_simulated()
        except Exception as e:
            print(f"Soundcard error: {e}. Falling back to simulated stream.")
            self._start_simulated()
    
    def _start_serial(self):
        """Start serial port capture."""
        try:
            import serial
            self._serial_conn = serial.Serial(
                self.config.serial_port,
                self.config.serial_baud,
                timeout=0.1)
            
            def serial_reader():
                dt_map = {'int8': np.int8, 'int16': np.int16,
                          'float32': np.float32}
                dtype = dt_map.get(self.config.serial_dtype, np.int16)
                bytes_per_sample = np.dtype(dtype).itemsize
                chunk_bytes = self.config.chunk_size * bytes_per_sample
                
                while not self._stop_event.is_set():
                    raw = self._serial_conn.read(chunk_bytes)
                    if len(raw) >= bytes_per_sample:
                        data = np.frombuffer(raw, dtype=dtype)
                        if dtype != np.float32:
                            info = np.iinfo(dtype)
                            data = data.astype(np.float64) / float(info.max)
                        else:
                            data = data.astype(np.float64)
                        self.ring_buffer.write(data)
                        try:
                            self._frame_queue.put(data.copy(), block=False)
                        except queue.Full:
                            pass
            
            t = threading.Thread(target=serial_reader, daemon=True)
            t.start()
            
        except ImportError:
            print("Warning: pyserial not installed. Using simulated stream.")
            print("Install with: pip install pyserial")
            self._start_simulated()
        except Exception as e:
            print(f"Serial error: {e}. Falling back to simulated stream.")
            self._start_simulated()
    
    def _start_simulated(self):
        """Start simulated data source for testing."""
        def sim_source():
            t_offset = 0.0
            while not self._stop_event.is_set():
                n = self.config.chunk_size
                fs = self.config.sample_rate
                t = np.arange(n) / fs + t_offset
                
                # Generate test signal: 440Hz + 1kHz + noise
                signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
                          0.3 * np.sin(2 * np.pi * 1000 * t) +
                          0.05 * np.random.randn(n))
                
                t_offset += n / fs
                
                self.ring_buffer.write(signal)
                try:
                    self._frame_queue.put(signal.copy(), block=False)
                except queue.Full:
                    pass
                
                time.sleep(n / fs * 0.9)  # Approximate real-time
        
        t = threading.Thread(target=sim_source, daemon=True)
        t.start()
    
    def _process_loop(self):
        """Main processing loop — dispatches frames to callbacks."""
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                for cb in self._callbacks:
                    try:
                        cb(frame, self.config.sample_rate)
                    except Exception as e:
                        print(f"Callback error: {e}")
            except queue.Empty:
                continue
    
    @staticmethod
    def list_audio_devices() -> List[dict]:
        """List available audio input devices."""
        devices = []
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': info['defaultSampleRate'],
                    })
            pa.terminate()
        except ImportError:
            devices.append({'index': -1, 'name': 'Simulated (PyAudio not installed)',
                           'channels': 1, 'sample_rate': 44100})
        except Exception:
            devices.append({'index': -1, 'name': 'Simulated (no devices)',
                           'channels': 1, 'sample_rate': 44100})
        return devices
    
    @staticmethod
    def list_serial_ports() -> List[str]:
        """List available serial ports."""
        try:
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            return ['COM1', 'COM3', 'COM4']  # Default
        except Exception:
            return []
