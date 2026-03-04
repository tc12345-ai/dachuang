"""Stress Testing — UI Panel."""
import tkinter as tk
from tkinter import ttk, scrolledtext
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginUIBase
from core.event_bus import Events

class Panel(PluginUIBase):
    plugin_id = 'stress_testing'
    panel_title = 'Stress Test / 鲁棒性测试'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, bus=bus, ctx=ctx, **kw)
        self._create_ui()
        if bus:
            bus.subscribe(Events.STRESS_TEST_DONE, self._on_result,
                          subscriber_id=self.plugin_id)

    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        # Monte Carlo
        mc_lf = ttk.LabelFrame(left, text="Monte Carlo / 蒙特卡洛", padding=5)
        mc_lf.pack(fill=tk.X, padx=5, pady=5)
        r1 = ttk.Frame(mc_lf)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="Trials:").pack(side=tk.LEFT)
        self.mc_trials = tk.IntVar(value=500)
        ttk.Spinbox(r1, from_=50, to=5000, textvariable=self.mc_trials,
                    width=8).pack(side=tk.LEFT, padx=5)
        r2 = ttk.Frame(mc_lf)
        r2.pack(fill=tk.X)
        ttk.Label(r2, text="Perturb std:").pack(side=tk.LEFT)
        self.perturb_std = tk.DoubleVar(value=0.0001)
        ttk.Entry(r2, textvariable=self.perturb_std, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(mc_lf, text="Run Monte Carlo / 运行",
                   command=self._run_mc).pack(fill=tk.X, pady=3)

        # Fixed-point
        fp_lf = ttk.LabelFrame(left, text="Fixed-Point Sweep / 定点寻优", padding=5)
        fp_lf.pack(fill=tk.X, padx=5, pady=5)
        r3 = ttk.Frame(fp_lf)
        r3.pack(fill=tk.X)
        ttk.Label(r3, text="Max error (dB):").pack(side=tk.LEFT)
        self.max_err = tk.DoubleVar(value=0.5)
        ttk.Entry(r3, textvariable=self.max_err, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(fp_lf, text="Sweep / 扫描",
                   command=self._run_fp).pack(fill=tk.X, pady=3)

        # Degradation
        deg_lf = ttk.LabelFrame(left, text="Degradation / 退化仿真", padding=5)
        deg_lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(deg_lf, text="Full Stress Test / 完整测试",
                   command=self._run_full).pack(fill=tk.X, pady=3)

        # Results
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        ttk.Label(right, text="Test Results / 测试结果",
                  font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=5)
        self.output = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=('Consolas', 9),
            bg='#1a1a2e', fg='#e0e0e0')
        self.output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output.tag_configure('pass', foreground='#2ecc71')
        self.output.tag_configure('fail', foreground='#e74c3c')
        self.output.tag_configure('info', foreground='#3498db')

    def _log(self, text, tag=None):
        self.output.insert(tk.END, text + '\n', tag)
        self.output.see(tk.END)

    def _run_mc(self):
        self._log("Running Monte Carlo... (design filter first)", 'info')

    def _run_fp(self):
        self._log("Running fixed-point sweep... (design filter first)", 'info')

    def _run_full(self):
        self._log("Running full stress test... (design filter first)", 'info')

    def _on_result(self, event):
        r = event.payload.get('report', {})
        self._log(f"\n=== Stress Test: {r.get('method','')} ===", 'info')
        self._log(f"  Trials: {r.get('n_trials',0)}")
        pr = r.get('pass_rate', 0)
        tag = 'pass' if pr >= 0.95 else 'fail'
        self._log(f"  Pass rate: {pr:.1%}", tag)
        self._log(f"  Max deviation: {r.get('max_deviation_db',0):.3f} dB")
        if r.get('optimal_q'):
            self._log(f"  Optimal Q: {r['optimal_q']}", 'pass')
