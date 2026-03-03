"""
Spectrum Analysis Panel — 频谱分析面板

Signal loading, FFT/STFT/PSD analysis, measurements, and visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spectrum_analysis import SpectrumAnalyzer, SpectrumResult
from core.measurements import SignalMeasurements
from gui.visualization import DSPPlotter
from io_manager.file_import import FileImporter, SignalData
from utils.helpers import generate_test_signal, format_freq, format_db


class SpectrumPanel(ttk.Frame):
    """
    Spectrum Analysis Panel — 频谱分析面板
    
    Load signals, analyze spectrum, view results.
    """
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.analyzer = SpectrumAnalyzer()
        self.measurer = SignalMeasurements()
        self.importer = FileImporter()
        
        self.signal_data = None     # Current loaded signal
        self.spectrum_result = None
        self.psd_result = None
        self.stft_result = None
        self.measurement_result = None
        
        DSPPlotter.setup_style()
        self._create_ui()
    
    def _create_ui(self):
        """Build the panel UI."""
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        left_frame = ttk.Frame(paned, width=310)
        paned.add(left_frame, weight=0)
        
        canvas = tk.Canvas(left_frame, width=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL,
                                   command=canvas.yview)
        self.param_frame = ttk.Frame(canvas)
        self.param_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.param_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._create_controls(self.param_frame)
        
        # Right panel
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        self._create_plot_area(right_frame)
    
    def _create_controls(self, parent):
        pad = {'padx': 5, 'pady': 2}
        
        # === Data Source ===
        sec1 = ttk.LabelFrame(parent, text="数据源 / Data Source", padding=5)
        sec1.pack(fill=tk.X, **pad)
        
        ttk.Button(sec1, text="📂 加载文件 / Load File",
                   command=self._load_file).pack(fill=tk.X, pady=2)
        ttk.Button(sec1, text="🎵 生成测试信号 / Test Signal",
                   command=self._generate_test).pack(fill=tk.X, pady=2)
        
        self.signal_info = ttk.Label(sec1, text="未加载信号", wraplength=280,
                                      foreground='gray')
        self.signal_info.pack(fill=tk.X, pady=2)
        
        # Signal parameters
        sig_frame = ttk.Frame(sec1)
        sig_frame.pack(fill=tk.X)
        
        row = ttk.Frame(sig_frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="采样率 Fs (Hz):", width=18).pack(side=tk.LEFT)
        self.sig_fs = tk.DoubleVar(value=8000.0)
        ttk.Entry(row, textvariable=self.sig_fs, width=12).pack(side=tk.LEFT, padx=5)
        
        row2 = ttk.Frame(sig_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="通道 Channel:", width=18).pack(side=tk.LEFT)
        self.sig_channel = tk.IntVar(value=0)
        ttk.Spinbox(row2, from_=0, to=16, textvariable=self.sig_channel,
                    width=8).pack(side=tk.LEFT, padx=5)
        
        # === Test Signal Options ===
        sec_test = ttk.LabelFrame(parent, text="测试信号 / Test Signal", padding=5)
        sec_test.pack(fill=tk.X, **pad)
        
        self.test_type = tk.StringVar(value='sine')
        ttk.Label(sec_test, text="类型:").pack(anchor=tk.W)
        ttk.Combobox(sec_test, textvariable=self.test_type, width=15,
                     values=['sine', 'square', 'sawtooth', 'chirp',
                             'impulse', 'noise', 'multi_tone'],
                     state='readonly').pack(fill=tk.X)
        
        row3 = ttk.Frame(sec_test)
        row3.pack(fill=tk.X, pady=1)
        ttk.Label(row3, text="频率 (Hz):", width=15).pack(side=tk.LEFT)
        self.test_freq = tk.DoubleVar(value=440.0)
        ttk.Entry(row3, textvariable=self.test_freq, width=10).pack(side=tk.LEFT, padx=5)
        
        row4 = ttk.Frame(sec_test)
        row4.pack(fill=tk.X, pady=1)
        ttk.Label(row4, text="时长 (s):", width=15).pack(side=tk.LEFT)
        self.test_duration = tk.DoubleVar(value=0.1)
        ttk.Entry(row4, textvariable=self.test_duration, width=10).pack(side=tk.LEFT, padx=5)
        
        row5 = ttk.Frame(sec_test)
        row5.pack(fill=tk.X, pady=1)
        ttk.Label(row5, text="噪声级别:", width=15).pack(side=tk.LEFT)
        self.test_noise = tk.DoubleVar(value=0.0)
        ttk.Entry(row5, textvariable=self.test_noise, width=10).pack(side=tk.LEFT, padx=5)
        
        # === FFT Parameters ===
        sec2 = ttk.LabelFrame(parent, text="FFT 参数 / FFT Settings", padding=5)
        sec2.pack(fill=tk.X, **pad)
        
        row_nfft = ttk.Frame(sec2)
        row_nfft.pack(fill=tk.X, pady=1)
        ttk.Label(row_nfft, text="FFT 长度:", width=15).pack(side=tk.LEFT)
        self.nfft = tk.IntVar(value=2048)
        ttk.Combobox(row_nfft, textvariable=self.nfft, width=10,
                     values=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
                     ).pack(side=tk.LEFT, padx=5)
        
        row_win = ttk.Frame(sec2)
        row_win.pack(fill=tk.X, pady=1)
        ttk.Label(row_win, text="窗函数:", width=15).pack(side=tk.LEFT)
        self.window = tk.StringVar(value='Hann')
        ttk.Combobox(row_win, textvariable=self.window, width=15,
                     values=list(SpectrumAnalyzer.WINDOWS.keys()),
                     state='readonly').pack(side=tk.LEFT, padx=5)
        
        self.detrend = tk.BooleanVar(value=True)
        ttk.Checkbutton(sec2, text="去除直流 / Remove DC",
                        variable=self.detrend).pack(anchor=tk.W)
        
        # === STFT Parameters ===
        sec3 = ttk.LabelFrame(parent, text="STFT 参数", padding=5)
        sec3.pack(fill=tk.X, **pad)
        
        row_seg = ttk.Frame(sec3)
        row_seg.pack(fill=tk.X, pady=1)
        ttk.Label(row_seg, text="分段长度:", width=15).pack(side=tk.LEFT)
        self.nperseg = tk.IntVar(value=256)
        ttk.Combobox(row_seg, textvariable=self.nperseg, width=10,
                     values=[64, 128, 256, 512, 1024]).pack(side=tk.LEFT, padx=5)
        
        row_olap = ttk.Frame(sec3)
        row_olap.pack(fill=tk.X, pady=1)
        ttk.Label(row_olap, text="重叠率 (%):", width=15).pack(side=tk.LEFT)
        self.overlap_pct = tk.IntVar(value=75)
        ttk.Spinbox(row_olap, from_=0, to=95, textvariable=self.overlap_pct,
                    width=8).pack(side=tk.LEFT, padx=5)
        
        # === Actions ===
        sec4 = ttk.LabelFrame(parent, text="分析 / Analysis", padding=5)
        sec4.pack(fill=tk.X, **pad)
        
        ttk.Button(sec4, text="▶ FFT 频谱分析",
                   command=self._run_fft).pack(fill=tk.X, pady=2)
        ttk.Button(sec4, text="▶ PSD 功率谱密度",
                   command=self._run_psd).pack(fill=tk.X, pady=2)
        ttk.Button(sec4, text="▶ STFT 时频分析",
                   command=self._run_stft).pack(fill=tk.X, pady=2)
        ttk.Button(sec4, text="📏 信号测量 / Measurements",
                   command=self._run_measurements).pack(fill=tk.X, pady=2)
        ttk.Button(sec4, text="🔄 全部分析 / Analyze All",
                   command=self._run_all).pack(fill=tk.X, pady=2)
        
        # === Measurements Display ===
        sec5 = ttk.LabelFrame(parent, text="测量结果 / Measurements", padding=5)
        sec5.pack(fill=tk.X, **pad)
        
        self.meas_text = tk.Text(sec5, height=10, wrap=tk.WORD,
                                  font=('Consolas', 8), state=tk.DISABLED)
        self.meas_text.pack(fill=tk.X)
    
    def _create_plot_area(self, parent):
        """Create 4-subplot plot area."""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.set_facecolor('#fafafa')
        
        self.ax_time = self.fig.add_subplot(2, 2, 1)
        self.ax_fft = self.fig.add_subplot(2, 2, 2)
        self.ax_stft = self.fig.add_subplot(2, 2, 3)
        self.ax_psd = self.fig.add_subplot(2, 2, 4)
        
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self._draw_empty()
    
    def _draw_empty(self):
        """Draw placeholder plots."""
        for ax in [self.ax_time, self.ax_fft, self.ax_stft, self.ax_psd]:
            ax.clear()
            ax.text(0.5, 0.5, '加载信号后开始分析\nLoad a signal to start',
                    ha='center', va='center', fontsize=11, color='#bdc3c7',
                    transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
        self.canvas.draw()
    
    def _load_file(self):
        """Load signal file."""
        filepath = filedialog.askopenfilename(
            title="选择信号文件 / Select Signal File",
            filetypes=[
                ("All Supported", "*.wav *.csv *.txt *.mat *.bin *.raw *.dat"),
                ("WAV Files", "*.wav"),
                ("CSV/Text", "*.csv *.txt *.tsv"),
                ("MATLAB", "*.mat"),
                ("Binary", "*.bin *.raw *.dat"),
                ("All Files", "*.*"),
            ])
        
        if not filepath:
            return
        
        try:
            fs_override = None
            self.signal_data = self.importer.load(filepath, fs=fs_override)
            self.sig_fs.set(self.signal_data.fs)
            
            info = (f"✅ {self.signal_data.filename}\n"
                    f"采样率: {self.signal_data.fs:.0f} Hz, "
                    f"样本: {self.signal_data.n_samples}, "
                    f"通道: {self.signal_data.n_channels}, "
                    f"时长: {self.signal_data.duration:.3f}s")
            self.signal_info.config(text=info, foreground='black')
            
            self._plot_time_domain()
            self._update_status(f"已加载: {self.signal_data.filename}")
            
        except Exception as e:
            messagebox.showerror("加载错误", f"无法加载文件:\n{e}")
    
    def _generate_test(self):
        """Generate test signal."""
        try:
            fs = self.sig_fs.get()
            duration = self.test_duration.get()
            freq = self.test_freq.get()
            noise = self.test_noise.get()
            sig_type = self.test_type.get()
            
            # Multi-tone harmonics
            harmonics = None
            if sig_type == 'multi_tone':
                harmonics = [(1, 1.0), (2, 0.5), (3, 0.3), (5, 0.1)]
            elif sig_type == 'sine':
                harmonics = [(2, 0.05), (3, 0.02)]  # Slight harmonics
            
            t, sig = generate_test_signal(sig_type, duration, fs,
                                          freq=freq, noise_level=noise,
                                          harmonics=harmonics if sig_type in ('sine', 'multi_tone') else None)
            
            self.signal_data = SignalData()
            self.signal_data.data = sig
            self.signal_data.fs = fs
            self.signal_data.n_samples = len(sig)
            self.signal_data.n_channels = 1
            self.signal_data.duration = duration
            self.signal_data.filename = f"test_{sig_type}_{freq:.0f}Hz"
            self.signal_data.file_format = 'Generated'
            
            info = (f"🎵 {self.signal_data.filename}\n"
                    f"Fs={fs:.0f}Hz, N={len(sig)}, "
                    f"Duration={duration:.3f}s")
            self.signal_info.config(text=info, foreground='#2980b9')
            
            self._plot_time_domain()
            self._update_status(f"测试信号已生成: {sig_type} {freq}Hz")
            
        except Exception as e:
            messagebox.showerror("错误", f"信号生成失败:\n{e}")
    
    def _get_signal(self):
        """Get current signal data as 1D array."""
        if self.signal_data is None:
            messagebox.showinfo("提示", "请先加载信号 / Load a signal first")
            return None
        return self.signal_data.channel_data
    
    def _plot_time_domain(self):
        """Plot time-domain waveform."""
        sig = self._get_signal()
        if sig is None:
            return
        fs = self.sig_fs.get()
        t = np.arange(len(sig)) / fs
        DSPPlotter.plot_time_domain(self.ax_time, t, sig)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()
    
    def _run_fft(self):
        """Run FFT analysis."""
        sig = self._get_signal()
        if sig is None:
            return
        
        try:
            fs = self.sig_fs.get()
            nfft = self.nfft.get()
            window = self.window.get()
            
            self.spectrum_result = self.analyzer.compute_fft(
                sig, fs, nfft=nfft, window=window,
                detrend=self.detrend.get())
            
            self.ax_fft.clear()
            self.ax_fft.plot(self.spectrum_result.freq_hz,
                            self.spectrum_result.magnitude_db,
                            color='#2980b9', linewidth=0.8)
            self.ax_fft.set_xlabel('频率 / Frequency (Hz)')
            self.ax_fft.set_ylabel('幅度 / Magnitude (dB)')
            self.ax_fft.set_title('频谱 / FFT Spectrum', fontweight='bold')
            self.ax_fft.grid(True, alpha=0.3)
            
            self.fig.tight_layout(pad=2.5)
            self.canvas.draw()
            self._update_status("FFT 分析完成")
            
        except Exception as e:
            messagebox.showerror("FFT 错误", str(e))
    
    def _run_psd(self):
        """Run PSD analysis."""
        sig = self._get_signal()
        if sig is None:
            return
        
        try:
            fs = self.sig_fs.get()
            nfft = self.nfft.get()
            window = self.window.get()
            
            self.psd_result = self.analyzer.compute_psd(
                sig, fs, nfft=nfft, window=window)
            
            DSPPlotter.plot_psd(self.ax_psd,
                               self.psd_result.psd_freq,
                               self.psd_result.psd_db)
            
            self.fig.tight_layout(pad=2.5)
            self.canvas.draw()
            self._update_status("PSD 分析完成")
            
        except Exception as e:
            messagebox.showerror("PSD 错误", str(e))
    
    def _run_stft(self):
        """Run STFT analysis."""
        sig = self._get_signal()
        if sig is None:
            return
        
        try:
            fs = self.sig_fs.get()
            nperseg = self.nperseg.get()
            overlap_pct = self.overlap_pct.get()
            noverlap = int(nperseg * overlap_pct / 100)
            window = self.window.get()
            
            self.stft_result = self.analyzer.compute_stft(
                sig, fs, nperseg=nperseg, noverlap=noverlap,
                window=window)
            
            DSPPlotter.plot_spectrogram(
                self.ax_stft,
                self.stft_result.stft_times,
                self.stft_result.stft_freqs,
                self.stft_result.stft_magnitude_db)
            
            self.fig.tight_layout(pad=2.5)
            self.canvas.draw()
            self._update_status("STFT 分析完成")
            
        except Exception as e:
            messagebox.showerror("STFT 错误", str(e))
    
    def _run_measurements(self):
        """Run signal measurements."""
        sig = self._get_signal()
        if sig is None:
            return
        
        try:
            fs = self.sig_fs.get()
            nfft = self.nfft.get()
            
            win_name = self.analyzer._get_scipy_window_name(self.window.get())
            if isinstance(win_name, tuple):
                win_name = win_name[0]
            
            self.measurement_result = self.measurer.analyze(
                sig, fs, nfft=nfft, window=win_name)
            
            m = self.measurement_result
            
            # Update measurement display
            self.meas_text.config(state=tk.NORMAL)
            self.meas_text.delete('1.0', tk.END)
            
            lines = []
            lines.append("═══ 信号测量结果 ═══")
            lines.append(f"基频: {format_freq(m.fundamental_freq)}")
            lines.append(f"基频幅度: {m.fundamental_mag_db:.1f} dB")
            lines.append(f"")
            lines.append(f"THD:  {m.thd_percent:.4f}% ({m.thd_db:.1f} dB)")
            lines.append(f"SNR:  {m.snr_db:.1f} dB")
            lines.append(f"SFDR: {m.sfdr_db:.1f} dB")
            lines.append(f"SINAD: {m.sinad_db:.1f} dB")
            lines.append(f"ENOB: {m.enob:.2f} bits")
            lines.append(f"")
            lines.append(f"信号功率: {m.signal_power_db:.1f} dB")
            lines.append(f"噪声功率: {m.noise_power_db:.1f} dB")
            lines.append(f"噪声底: {m.noise_floor_db:.1f} dB")
            lines.append(f"")
            
            if m.harmonic_freqs:
                lines.append("谐波分析:")
                for i, (hf, hm) in enumerate(zip(m.harmonic_freqs,
                                                    m.harmonic_mags_db)):
                    label = "基波" if i == 0 else f"H{i+1}"
                    lines.append(f"  {label}: {format_freq(hf)} = {hm:.1f} dB")
            
            self.meas_text.insert(tk.END, '\n'.join(lines))
            self.meas_text.config(state=tk.DISABLED)
            
            # Mark peaks on FFT plot
            if self.spectrum_result is not None:
                for hf, hm in zip(m.harmonic_freqs[:5], m.harmonic_mags_db[:5]):
                    idx = np.argmin(np.abs(self.spectrum_result.freq_hz - hf))
                    self.ax_fft.plot(hf, self.spectrum_result.magnitude_db[idx],
                                    'rv', markersize=6)
                    self.ax_fft.annotate(f"{format_freq(hf)}",
                                        (hf, self.spectrum_result.magnitude_db[idx]),
                                        textcoords="offset points",
                                        xytext=(5, 8), fontsize=6, color='red')
                self.canvas.draw()
            
            self._update_status(f"测量完成: THD={m.thd_percent:.3f}%, SNR={m.snr_db:.1f}dB")
            
        except Exception as e:
            messagebox.showerror("测量错误", str(e))
    
    def _run_all(self):
        """Run all analyses."""
        if self._get_signal() is None:
            return
        self._plot_time_domain()
        self._run_fft()
        self._run_psd()
        self._run_stft()
        self._run_measurements()
    
    def _update_status(self, msg):
        if self.status_callback:
            self.status_callback(msg)
    
    def get_spectrum_result(self):
        """Get current spectrum results for export."""
        return self.spectrum_result
    
    def get_signal_data(self):
        """Get current signal data."""
        return self.signal_data
