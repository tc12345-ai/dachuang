"""UX & Ecosystem — UI Panel (3D Spectrogram + Plugin Market)."""
import tkinter as tk
from tkinter import ttk
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginUIBase
from core.event_bus import Events
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D


class Panel(PluginUIBase):
    plugin_id = 'ux_ecosystem'
    panel_title = '3D & Ecosystem / 生态扩展'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, bus=bus, ctx=ctx, **kw)
        self._create_ui()

    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: controls
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        # 3D controls
        lf3d = ttk.LabelFrame(left, text="3D Spectrogram / 3D谱图", padding=5)
        lf3d.pack(fill=tk.X, padx=5, pady=5)

        r1 = ttk.Frame(lf3d)
        r1.pack(fill=tk.X, pady=1)
        ttk.Label(r1, text="Mode:").pack(side=tk.LEFT)
        self.mode_3d = tk.StringVar(value='surface')
        ttk.Combobox(r1, textvariable=self.mode_3d, width=12,
                     values=['surface', 'wireframe', 'waterfall'],
                     state='readonly').pack(side=tk.LEFT, padx=5)

        r2 = ttk.Frame(lf3d)
        r2.pack(fill=tk.X, pady=1)
        ttk.Label(r2, text="Colormap:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value='inferno')
        ttk.Combobox(r2, textvariable=self.cmap_var, width=12,
                     values=['inferno', 'viridis', 'plasma', 'magma', 'jet'],
                     state='readonly').pack(side=tk.LEFT, padx=5)

        ttk.Button(lf3d, text="Generate 3D Demo / 生成演示",
                   command=self._demo_3d).pack(fill=tk.X, pady=3)

        # Plugin market
        mkt_lf = ttk.LabelFrame(left, text="Plugin Market / 插件市场", padding=5)
        mkt_lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(mkt_lf, text="Refresh Registry / 刷新",
                   command=self._load_registry).pack(fill=tk.X, pady=2)

        self.registry_list = tk.Listbox(mkt_lf, height=6,
                                         font=('Consolas', 8))
        self.registry_list.pack(fill=tk.X, pady=2)

        # Cloud pool
        cloud_lf = ttk.LabelFrame(left, text="Cloud Pool / 云端算力", padding=5)
        cloud_lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(cloud_lf, text="Submit Test Job / 提交测试任务",
                   command=self._submit_job).pack(fill=tk.X, pady=2)
        self.cloud_status = ttk.Label(cloud_lf, text="Idle")
        self.cloud_status.pack(fill=tk.X)

        # Right: 3D plot
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.fig.set_facecolor('#fafafa')
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.ax3d.set_xlabel('Time (s)', fontsize=8)
        self.ax3d.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax3d.set_zlabel('Magnitude (dB)', fontsize=8)
        self.ax3d.set_title('3D Spectrogram', fontweight='bold')

        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _demo_3d(self):
        """Generate and plot demo 3D spectrogram."""
        from scipy import signal as sig

        # Generate test chirp signal
        fs = 8000
        t = np.linspace(0, 1.0, fs)
        chirp = sig.chirp(t, 200, 1.0, 2000, method='linear')
        chirp += 0.3 * np.sin(2 * np.pi * 440 * t)
        chirp += 0.1 * np.random.randn(len(t))

        # Compute STFT
        f, t_stft, Zxx = sig.stft(chirp, fs=fs, nperseg=256, noverlap=192)
        mag_db = 20 * np.log10(np.maximum(np.abs(Zxx), 1e-12))

        # Limit for 3D rendering performance
        max_f_idx = np.searchsorted(f, 3000)
        f = f[:max_f_idx]
        mag_db = mag_db[:max_f_idx, :]

        T, F = np.meshgrid(t_stft, f)

        self.ax3d.clear()
        mode = self.mode_3d.get()
        cmap = self.cmap_var.get()

        if mode == 'surface':
            self.ax3d.plot_surface(T, F, mag_db, cmap=cmap,
                                   alpha=0.85, rstride=2, cstride=2)
        elif mode == 'wireframe':
            self.ax3d.plot_wireframe(T, F, mag_db, color='#2980b9',
                                     rstride=3, cstride=3, linewidth=0.5)
        else:  # waterfall
            for i in range(0, len(f), 4):
                self.ax3d.plot(t_stft, [f[i]]*len(t_stft), mag_db[i, :],
                              color=matplotlib.cm.get_cmap(cmap)(i/len(f)),
                              linewidth=0.6, alpha=0.7)

        self.ax3d.set_xlabel('Time (s)', fontsize=8)
        self.ax3d.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax3d.set_zlabel('dB', fontsize=8)
        self.ax3d.set_title('3D Spectrogram (Demo Chirp)', fontsize=10)
        self.ax3d.view_init(elev=30, azim=225)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def _load_registry(self):
        self.registry_list.delete(0, tk.END)
        try:
            from service import Service
            from core.event_bus import EventBus
            svc = Service()
            svc.activate(EventBus.instance(), {})
            registry = svc.get_registry()
            for p in registry:
                status = '[OK]' if p.get('installed') else '[--]'
                self.registry_list.insert(
                    tk.END, f"{status} {p['name']} v{p['version']}")
        except Exception as e:
            self.registry_list.insert(tk.END, f"Error: {e}")

    def _submit_job(self):
        self.cloud_status.config(text="Job submitted (simulated local)")
