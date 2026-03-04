"""Domain Toolkits — UI Panel."""
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginUIBase
from core.event_bus import Events, make_event
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Panel(PluginUIBase):
    plugin_id = 'domain_toolkits'
    panel_title = 'Domain / 行业套件'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, bus=bus, ctx=ctx, **kw)
        self._create_ui()

    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        # Domain selector
        dom_lf = ttk.LabelFrame(left, text="Domain / 领域", padding=5)
        dom_lf.pack(fill=tk.X, padx=5, pady=5)
        self.domain_var = tk.StringVar(value='vibration')
        for text, val in [('Vibration / 振动', 'vibration'),
                          ('BioMed / 生物医学', 'biomed'),
                          ('Acoustic / 声学', 'acoustic'),
                          ('Comms / 通信', 'comms')]:
            ttk.Radiobutton(dom_lf, text=text, variable=self.domain_var,
                           value=val).pack(anchor=tk.W)

        # Analysis selector per domain
        anlys_lf = ttk.LabelFrame(left, text="Analysis / 分析", padding=5)
        anlys_lf.pack(fill=tk.X, padx=5, pady=5)
        self.analysis_var = tk.StringVar(value='envelope')
        analyses = {
            'envelope': 'Envelope Demod / 包络解调',
            'third_octave': '1/3 Octave / 倍频程',
            'ecg_baseline': 'ECG Baseline / 基线去除',
            'eeg_rhythm': 'EEG Rhythm / 节律提取',
            'thd_n': 'THD+N Test',
            'constellation': 'Constellation / 星座图',
            'eye_diagram': 'Eye Diagram / 眼图',
        }
        for val, text in analyses.items():
            ttk.Radiobutton(anlys_lf, text=text, variable=self.analysis_var,
                           value=val).pack(anchor=tk.W)

        ttk.Button(left, text="Run Analysis / 运行分析",
                   command=self._run).pack(fill=tk.X, padx=5, pady=5)

        self.result_text = scrolledtext.ScrolledText(
            left, height=8, font=('Consolas', 8), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: Plot
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.set_facecolor('#fafafa')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.text(0.5, 0.5, 'Select analysis and run',
                    ha='center', va='center', fontsize=12, color='#bdc3c7',
                    transform=self.ax.transAxes)
        self.canvas.draw()

    def _run(self):
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END,
            f"Running {self.analysis_var.get()} for {self.domain_var.get()}...\n"
            "Load signal in Spectrum tab first.\n")
