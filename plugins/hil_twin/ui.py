"""HIL & Digital Twin — UI Panel (Virtual Instrument + Resource View)."""
import tkinter as tk
from tkinter import ttk
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginUIBase
from core.event_bus import Events, make_event
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Panel(PluginUIBase):
    plugin_id = 'hil_twin'
    panel_title = 'HIL & Twin / 硬件在环'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, bus=bus, ctx=ctx, **kw)
        self._create_ui()
        if bus:
            bus.subscribe(Events.RESOURCE_ESTIMATED, self._on_resource,
                          subscriber_id=self.plugin_id)
            bus.subscribe(Events.HIL_RESPONSE_RECV, self._on_response,
                          subscriber_id=self.plugin_id)

    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: Controls + Resource display
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        # Device
        dev_lf = ttk.LabelFrame(left, text="Device / 设备", padding=5)
        dev_lf.pack(fill=tk.X, padx=5, pady=5)
        self.device_var = tk.StringVar(value='MockDevice')
        ttk.Combobox(dev_lf, textvariable=self.device_var,
                     values=['MockDevice', 'Serial (TODO)', 'TCP (TODO)'],
                     state='readonly').pack(fill=tk.X, pady=2)
        ttk.Button(dev_lf, text="Connect / 连接",
                   command=self._connect).pack(fill=tk.X, pady=2)
        ttk.Button(dev_lf, text="Push & Test / 推送测试",
                   command=self._push_test).pack(fill=tk.X, pady=2)

        # Resource estimation
        res_lf = ttk.LabelFrame(left, text="Resource Estimate / 资源估算", padding=5)
        res_lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(res_lf, text="Estimate / 估算",
                   command=self._estimate).pack(fill=tk.X, pady=2)

        self.res_text = tk.Text(left, height=15, wrap=tk.WORD,
                                 font=('Consolas', 8), state=tk.DISABLED)
        self.res_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: Virtual oscilloscope
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        ttk.Label(right, text="Virtual Oscilloscope / 虚拟示波器",
                  font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.set_facecolor('#0a0a0a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0a0a0a')
        self.ax.grid(True, color='#1a3a1a', linewidth=0.5)
        self.ax.set_xlabel('Time', color='#00ff00', fontsize=8)
        self.ax.set_ylabel('Amplitude', color='#00ff00', fontsize=8)
        self.ax.tick_params(colors='#00ff00', labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color('#1a3a1a')
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    def _connect(self):
        self._log("MockDevice connected.\n")

    def _push_test(self):
        self._log("Push & Test: use Filter Design tab first.\n")

    def _estimate(self):
        self._log("Estimating resources...\n")

    def _log(self, text):
        self.res_text.config(state=tk.NORMAL)
        self.res_text.insert(tk.END, text)
        self.res_text.see(tk.END)
        self.res_text.config(state=tk.DISABLED)

    def _on_resource(self, event):
        p = event.payload
        lines = ["=== Resource Estimation ===\n"]
        for target in ['cortex_m4', 'zynq']:
            r = p.get(target, {})
            if r:
                lines.append(f"  [{target}]")
                lines.append(f"    MACs/sample: {r.get('macs_per_sample',0)}")
                lines.append(f"    Memory: {r.get('memory_bytes',0)} bytes")
                if r.get('dsp_slices'):
                    lines.append(f"    DSP48: {r.get('dsp_slices')}")
                    lines.append(f"    LUT: {r.get('lut_count')}")
                    lines.append(f"    BRAM: {r.get('bram_blocks')}")
                lines.append(f"    Notes: {r.get('notes','')}\n")
        self._log('\n'.join(lines))

    def _on_response(self, event):
        resp = event.payload.get('response', [])
        if resp:
            data = np.array(resp)
            t = np.arange(len(data))
            self.ax.clear()
            self.ax.set_facecolor('#0a0a0a')
            self.ax.plot(t, data, color='#00ff00', linewidth=0.8)
            self.ax.grid(True, color='#1a3a1a')
            self.ax.set_title('HIL Response', color='#00ff00', fontsize=9)
            self.canvas.draw_idle()
