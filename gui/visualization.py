"""
Visualization Helpers — 可视化辅助模块

Bode plots, pole-zero plots, spectrograms, waterfall, heatmaps.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from typing import Optional, Tuple


class DSPPlotter:
    """
    DSP Visualization Helper — DSP 可视化辅助工具
    
    Provides methods for specialized DSP plots.
    """
    
    # Color scheme
    COLORS = {
        'primary': '#2980b9',
        'secondary': '#e74c3c',
        'accent': '#27ae60',
        'warning': '#f39c12',
        'grid': '#ecf0f1',
        'bg': '#fafafa',
        'text': '#2c3e50',
        'passband': '#d5efb7',
        'stopband': '#f9cccc',
    }
    
    @staticmethod
    def setup_style():
        """Apply consistent plot style with CJK font support."""
        # Try to find a CJK-compatible font
        import matplotlib.font_manager as fm
        cjk_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun',
                      'FangSong', 'KaiTi', 'Arial Unicode MS', 'Noto Sans CJK SC']
        font_family = 'sans-serif'
        for f in cjk_fonts:
            try:
                if any(f.lower() in fp.name.lower() for fp in fm.fontManager.ttflist):
                    font_family = f
                    break
            except Exception:
                continue
        
        plt.rcParams.update({
            'figure.facecolor': '#fafafa',
            'axes.facecolor': '#ffffff',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#bdc3c7',
            'axes.axisbelow': True,
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'font.family': 'sans-serif',
            'font.sans-serif': [font_family, 'DejaVu Sans', 'Arial'],
            'axes.unicode_minus': False,
        })
    
    @staticmethod
    def plot_magnitude_response(ax, freq_hz, magnitude_db,
                                 title="幅度响应 / Magnitude Response",
                                 passband_limits=None,
                                 stopband_limits=None):
        """
        Plot magnitude response.
        
        Args:
            ax: Matplotlib axes
            freq_hz: Frequency vector (Hz)
            magnitude_db: Magnitude in dB
            title: Plot title
            passband_limits: (f_low, f_high, ripple_db) for shading
            stopband_limits: (f_low, f_high, atten_db) for shading
        """
        ax.clear()
        ax.plot(freq_hz, magnitude_db, color='#2980b9', linewidth=1.5,
                label='|H(f)|')
        
        # Passband/stopband shading
        if passband_limits:
            fl, fh, rip = passband_limits
            ax.axhspan(-rip, rip, alpha=0.15, color='#27ae60',
                       label=f'Passband (±{rip}dB)')
            ax.axvline(fl, color='#27ae60', linestyle='--', alpha=0.5)
            if fh < freq_hz[-1]:
                ax.axvline(fh, color='#27ae60', linestyle='--', alpha=0.5)
        
        if stopband_limits:
            fl, fh, att = stopband_limits
            ax.axhline(-att, color='#e74c3c', linestyle=':', alpha=0.5,
                       label=f'Stopband (-{att}dB)')
        
        ax.set_xlabel('频率 / Frequency (Hz)')
        ax.set_ylabel('幅度 / Magnitude (dB)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)
    
    @staticmethod
    def plot_phase_response(ax, freq_hz, phase_deg,
                            title="相位响应 / Phase Response"):
        """Plot phase response."""
        ax.clear()
        ax.plot(freq_hz, phase_deg, color='#e74c3c', linewidth=1.5)
        ax.set_xlabel('频率 / Frequency (Hz)')
        ax.set_ylabel('相位 / Phase (°)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_group_delay(ax, freq_hz, group_delay,
                         title="群时延 / Group Delay"):
        """Plot group delay."""
        ax.clear()
        # Clip extreme values for display
        gd = np.clip(group_delay, -1000, 1000)
        ax.plot(freq_hz, gd, color='#8e44ad', linewidth=1.5)
        ax.set_xlabel('频率 / Frequency (Hz)')
        ax.set_ylabel('群时延 / Group Delay (samples)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_pole_zero(ax, zeros, poles,
                       title="极零图 / Pole-Zero Plot"):
        """
        Plot pole-zero diagram with unit circle.
        
        Args:
            ax: Matplotlib axes
            zeros: Array of zeros (complex)
            poles: Array of poles (complex)
            title: Plot title
        """
        ax.clear()
        
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.4)
        
        # Zeros (o markers)
        if zeros is not None and len(zeros) > 0:
            ax.plot(np.real(zeros), np.imag(zeros), 'o',
                    markersize=8, markerfacecolor='none',
                    markeredgecolor='#2980b9', markeredgewidth=2,
                    label=f'Zeros ({len(zeros)})')
        
        # Poles (x markers)
        if poles is not None and len(poles) > 0:
            ax.plot(np.real(poles), np.imag(poles), 'x',
                    markersize=8, markeredgecolor='#e74c3c',
                    markeredgewidth=2,
                    label=f'Poles ({len(poles)})')
        
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('实部 / Real')
        ax.set_ylabel('虚部 / Imaginary')
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_impulse_response(ax, t, h,
                              title="脉冲响应 / Impulse Response"):
        """Plot impulse response."""
        ax.clear()
        ax.stem(t, h, linefmt='-', markerfmt='o', basefmt='k-')
        # Style the stem plot
        for line in ax.get_lines():
            line.set_color('#2980b9')
            line.set_markersize(3)
        ax.set_xlabel('时间 / Time (s)')
        ax.set_ylabel('幅度 / Amplitude')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_step_response(ax, t, s,
                           title="阶跃响应 / Step Response"):
        """Plot step response."""
        ax.clear()
        ax.plot(t, s, color='#27ae60', linewidth=1.5)
        ax.axhline(1.0, color='#e74c3c', linestyle='--', alpha=0.5,
                   label='Unity')
        ax.set_xlabel('时间 / Time (s)')
        ax.set_ylabel('幅度 / Amplitude')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_spectrogram(ax, times, freqs, magnitude_db,
                         title="谱图 / Spectrogram",
                         cmap='inferno', vmin=None, vmax=None):
        """
        Plot spectrogram (STFT).
        
        Args:
            ax: Matplotlib axes
            times: Time vector
            freqs: Frequency vector
            magnitude_db: 2D magnitude array in dB
            title: Plot title
            cmap: Colormap
            vmin, vmax: Color limits
        """
        ax.clear()
        
        if vmin is None:
            vmin = np.max(magnitude_db) - 80
        if vmax is None:
            vmax = np.max(magnitude_db)
        
        im = ax.pcolormesh(times, freqs, magnitude_db,
                           shading='gouraud', cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('时间 / Time (s)')
        ax.set_ylabel('频率 / Frequency (Hz)')
        ax.set_title(title, fontweight='bold')
        
        return im
    
    @staticmethod
    def plot_time_domain(ax, t, signal, title="时域波形 / Time Domain",
                         label=None, color='#2980b9'):
        """Plot time-domain waveform."""
        ax.clear()
        ax.plot(t, signal, color=color, linewidth=0.8, label=label)
        ax.set_xlabel('时间 / Time (s)')
        ax.set_ylabel('幅度 / Amplitude')
        ax.set_title(title, fontweight='bold')
        if label:
            ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_psd(ax, freq_hz, psd_db,
                 title="功率谱密度 / Power Spectral Density"):
        """Plot PSD."""
        ax.clear()
        ax.plot(freq_hz, psd_db, color='#8e44ad', linewidth=1.2)
        ax.set_xlabel('频率 / Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_comparison(ax, freq_hz, curves, labels, colors=None,
                        title="对比 / Comparison"):
        """Plot multiple frequency responses for comparison."""
        ax.clear()
        
        default_colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12',
                          '#8e44ad', '#1abc9c', '#e67e22', '#34495e']
        if colors is None:
            colors = default_colors
        
        for i, (curve, label) in enumerate(zip(curves, labels)):
            c = colors[i % len(colors)]
            ax.plot(freq_hz, curve, color=c, linewidth=1.2, label=label)
        
        ax.set_xlabel('频率 / Frequency (Hz)')
        ax.set_ylabel('幅度 / Magnitude (dB)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
