"""
Real-time Streaming Panel — 实时流采集面板

Live waveform, spectrum, and spectrogram display
with configurable audio/serial/SDR input.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys, os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.realtime_stream import RealtimeStream, StreamConfig
from core.spectrum_analysis import SpectrumAnalyzer
from gui.visualization import DSPPlotter


class RealtimePanel(ttk.Frame):
    """
    Real-time Streaming Panel — 实时流面板
    
    Live visualization of audio/data input with
    waveform, spectrum, and spectrogram views.
    """
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.stream = None
        self.analyzer = SpectrumAnalyzer()
        self._update_interval = 50  # ms
        self._update_job = None
        self._latest_frame = None
        self._history_buffer = np.zeros(8192)
        self._history_pos = 0
        
        DSPPlotter.setup_style()
        self._create_ui()
    
    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: config
        left = ttk.Frame(paned, width=280)
        paned.add(left, weight=0)
        self._create_controls(left)
        
        # Right: plots
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._create_plots(right)
    
    def _create_controls(self, parent):
        pad = {'padx': 5, 'pady': 2}
        
        # Source selection
        sec1 = ttk.LabelFrame(parent, text="Data Source / 数据源", padding=5)
        sec1.pack(fill=tk.X, **pad)
        
        self.source_type = tk.StringVar(value='simulated')
        for text, val in [('Simulated / 模拟', 'simulated'),
                          ('Soundcard / 声卡', 'soundcard'),
                          ('Serial / 串口', 'serial')]:
            ttk.Radiobutton(sec1, text=text, variable=self.source_type,
                           value=val).pack(anchor=tk.W)
        
        # Parameters
        sec2 = ttk.LabelFrame(parent, text="Parameters / 参数", padding=5)
        sec2.pack(fill=tk.X, **pad)
        
        row = ttk.Frame(sec2)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Fs (Hz):", width=14).pack(side=tk.LEFT)
        self.fs_var = tk.IntVar(value=44100)
        ttk.Combobox(row, textvariable=self.fs_var, width=10,
                     values=[8000, 16000, 22050, 44100, 48000, 96000]
                     ).pack(side=tk.LEFT, padx=5)
        
        row2 = ttk.Frame(sec2)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="Chunk:", width=14).pack(side=tk.LEFT)
        self.chunk_var = tk.IntVar(value=1024)
        ttk.Combobox(row2, textvariable=self.chunk_var, width=10,
                     values=[256, 512, 1024, 2048, 4096]
                     ).pack(side=tk.LEFT, padx=5)
        
        row3 = ttk.Frame(sec2)
        row3.pack(fill=tk.X, pady=1)
        ttk.Label(row3, text="FFT Size:", width=14).pack(side=tk.LEFT)
        self.nfft_var = tk.IntVar(value=2048)
        ttk.Combobox(row3, textvariable=self.nfft_var, width=10,
                     values=[512, 1024, 2048, 4096, 8192]
                     ).pack(side=tk.LEFT, padx=5)
        
        row4 = ttk.Frame(sec2)
        row4.pack(fill=tk.X, pady=1)
        ttk.Label(row4, text="Update (ms):", width=14).pack(side=tk.LEFT)
        self.update_var = tk.IntVar(value=50)
        ttk.Spinbox(row4, from_=20, to=500, textvariable=self.update_var,
                    width=8).pack(side=tk.LEFT, padx=5)
        
        # Serial port (shown when serial selected)
        self.serial_frame = ttk.LabelFrame(parent, text="Serial / 串口", padding=5)
        self.serial_frame.pack(fill=tk.X, **pad)
        
        row_port = ttk.Frame(self.serial_frame)
        row_port.pack(fill=tk.X, pady=1)
        ttk.Label(row_port, text="Port:", width=14).pack(side=tk.LEFT)
        self.serial_port = tk.StringVar(value='COM3')
        ttk.Combobox(row_port, textvariable=self.serial_port, width=10,
                     values=RealtimeStream.list_serial_ports()
                     ).pack(side=tk.LEFT, padx=5)
        
        row_baud = ttk.Frame(self.serial_frame)
        row_baud.pack(fill=tk.X, pady=1)
        ttk.Label(row_baud, text="Baud:", width=14).pack(side=tk.LEFT)
        self.serial_baud = tk.IntVar(value=115200)
        ttk.Combobox(row_baud, textvariable=self.serial_baud, width=10,
                     values=[9600, 19200, 38400, 57600, 115200, 230400, 921600]
                     ).pack(side=tk.LEFT, padx=5)
        
        # Controls
        sec3 = ttk.LabelFrame(parent, text="Control / 控制", padding=5)
        sec3.pack(fill=tk.X, **pad)
        
        self.start_btn = ttk.Button(sec3, text="Start / 开始",
                                     command=self._start)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(sec3, text="Stop / 停止",
                                    command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Status
        self.rt_status = ttk.Label(parent, text="Status: Idle",
                                    font=('Consolas', 8), wraplength=260)
        self.rt_status.pack(fill=tk.X, padx=5, pady=5)
        
        # Live metrics
        sec4 = ttk.LabelFrame(parent, text="Live Metrics / 实时指标", padding=5)
        sec4.pack(fill=tk.BOTH, expand=True, **pad)
        
        self.metrics_text = tk.Text(sec4, height=8, wrap=tk.WORD,
                                     font=('Consolas', 8), state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_plots(self, parent):
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.set_facecolor('#fafafa')
        
        self.ax_wave = self.fig.add_subplot(3, 1, 1)
        self.ax_fft = self.fig.add_subplot(3, 1, 2)
        self.ax_waterfall = self.fig.add_subplot(3, 1, 3)
        
        self.fig.tight_layout(pad=2.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Init waterfall data
        self._waterfall_data = np.full((50, 512), -100.0)
        
        # Empty initial state
        for ax in [self.ax_wave, self.ax_fft, self.ax_waterfall]:
            ax.text(0.5, 0.5, 'Start streaming to see live data',
                    ha='center', va='center', fontsize=11, color='#bdc3c7',
                    transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
        self.canvas.draw()
    
    def _start(self):
        """Start real-time streaming."""
        config = StreamConfig(
            source_type=self.source_type.get(),
            sample_rate=float(self.fs_var.get()),
            chunk_size=self.chunk_var.get(),
            serial_port=self.serial_port.get(),
            serial_baud=self.serial_baud.get(),
        )
        
        self.stream = RealtimeStream(config)
        self.stream.add_callback(self._on_frame)
        self.stream.start()
        
        self._update_interval = self.update_var.get()
        self._schedule_update()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.rt_status.config(text=f"Status: Running ({config.source_type})")
        
        if self.status_callback:
            self.status_callback(f"Real-time: streaming from {config.source_type}")
    
    def _stop(self):
        """Stop streaming."""
        if self.stream:
            self.stream.stop()
        
        if self._update_job:
            self.after_cancel(self._update_job)
            self._update_job = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.rt_status.config(text="Status: Stopped")
        
        if self.status_callback:
            self.status_callback("Real-time: stopped")
    
    def _on_frame(self, frame, fs):
        """Callback for each new frame of data."""
        self._latest_frame = frame.copy()
        
        # Update history
        n = len(frame)
        buf_len = len(self._history_buffer)
        if self._history_pos + n > buf_len:
            # Shift buffer
            shift = buf_len // 2
            self._history_buffer[:buf_len - shift] = self._history_buffer[shift:]
            self._history_pos = buf_len - shift
        
        self._history_buffer[self._history_pos:self._history_pos + n] = frame[:n]
        self._history_pos += n
    
    def _schedule_update(self):
        """Schedule next plot update."""
        self._update_plots()
        if self.stream and self.stream.is_running:
            self._update_job = self.after(self._update_interval,
                                          self._schedule_update)
    
    def _update_plots(self):
        """Update live plots."""
        if self._latest_frame is None or len(self._latest_frame) == 0:
            return
        
        fs = float(self.fs_var.get())
        nfft = self.nfft_var.get()
        
        # Get recent data
        data_len = min(self._history_pos, len(self._history_buffer))
        data = self._history_buffer[:data_len]
        
        if len(data) < 64:
            return
        
        # Use last few chunks for display
        display_len = min(len(data), nfft * 2)
        display_data = data[-display_len:]
        
        t = np.arange(len(display_data)) / fs
        
        # Waveform
        self.ax_wave.clear()
        self.ax_wave.plot(t, display_data, color='#2980b9', linewidth=0.5)
        self.ax_wave.set_title('Waveform / 波形', fontweight='bold', fontsize=9)
        self.ax_wave.set_xlabel('Time (s)', fontsize=8)
        self.ax_wave.set_ylabel('Amplitude', fontsize=8)
        self.ax_wave.grid(True, alpha=0.3)
        
        # FFT
        try:
            from scipy.fft import rfft, rfftfreq
            from scipy.signal import get_window
            
            fft_data = display_data[-nfft:] if len(display_data) >= nfft else display_data
            n = len(fft_data)
            win = get_window('hann', n)
            X = np.abs(rfft(fft_data * win, n=nfft))
            X_db = 20 * np.log10(np.maximum(X, 1e-12))
            freq = rfftfreq(nfft, 1/fs)
            
            self.ax_fft.clear()
            self.ax_fft.plot(freq, X_db, color='#e74c3c', linewidth=0.7)
            self.ax_fft.set_title('Spectrum / 频谱', fontweight='bold', fontsize=9)
            self.ax_fft.set_xlabel('Frequency (Hz)', fontsize=8)
            self.ax_fft.set_ylabel('Magnitude (dB)', fontsize=8)
            self.ax_fft.set_xlim(0, fs/2)
            self.ax_fft.grid(True, alpha=0.3)
            
            # Update waterfall
            n_bins = min(len(X_db), self._waterfall_data.shape[1])
            self._waterfall_data = np.roll(self._waterfall_data, -1, axis=0)
            self._waterfall_data[-1, :n_bins] = X_db[:n_bins]
            
            self.ax_waterfall.clear()
            self.ax_waterfall.imshow(
                self._waterfall_data, aspect='auto', origin='lower',
                cmap='inferno',
                extent=[0, fs/2, 0, self._waterfall_data.shape[0]],
                vmin=-80, vmax=0)
            self.ax_waterfall.set_title('Waterfall / 瀑布图', fontweight='bold', fontsize=9)
            self.ax_waterfall.set_xlabel('Frequency (Hz)', fontsize=8)
            self.ax_waterfall.set_ylabel('Time (frames)', fontsize=8)
            
        except Exception:
            pass
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()
        
        # Update metrics
        rms = np.sqrt(np.mean(display_data ** 2))
        peak = np.max(np.abs(display_data))
        rms_db = 20 * np.log10(max(rms, 1e-12))
        peak_db = 20 * np.log10(max(peak, 1e-12))
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert(tk.END,
            f"RMS: {rms:.6f} ({rms_db:.1f} dB)\n"
            f"Peak: {peak:.6f} ({peak_db:.1f} dB)\n"
            f"Crest Factor: {peak/(rms+1e-12):.2f}\n"
            f"Buffer: {data_len} samples\n"
            f"Display: {display_len} samples\n")
        self.metrics_text.config(state=tk.DISABLED)
    
    def destroy(self):
        """Clean up on destroy."""
        self._stop()
        super().destroy()
