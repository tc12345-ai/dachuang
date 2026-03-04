"""AI Assistant — UI Panel."""
import tkinter as tk
from tkinter import ttk, scrolledtext
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from core.protocols import PluginUIBase
from core.event_bus import Events, make_event


class Panel(PluginUIBase):
    """AI Assistant Panel — 智能助手面板."""
    plugin_id = 'ai_assistant'
    panel_title = 'AI Assistant / 智能助手'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, bus=bus, ctx=ctx, **kw)
        self._create_ui()
        if bus:
            bus.subscribe(Events.ANOMALY_DETECTED, self._on_anomalies,
                          subscriber_id=self.plugin_id)
            bus.subscribe(Events.FILTER_RECOMMENDED, self._on_recs,
                          subscriber_id=self.plugin_id)
            bus.subscribe(Events.CHAT_RESPONSE, self._on_chat_resp,
                          subscriber_id=self.plugin_id)

    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: Chat & Commands
        left = ttk.Frame(paned, width=350)
        paned.add(left, weight=0)

        # Chat section
        chat_lf = ttk.LabelFrame(left, text="Chat-to-Filter / 自然语言设计",
                                  padding=5)
        chat_lf.pack(fill=tk.X, padx=5, pady=5)

        self.chat_entry = ttk.Entry(chat_lf, font=('Segoe UI', 10))
        self.chat_entry.pack(fill=tk.X, pady=2)
        self.chat_entry.insert(0, "notch 50Hz + lowpass 4kHz")
        self.chat_entry.bind('<Return>', lambda e: self._send_chat())

        ttk.Button(chat_lf, text="Send / 发送",
                   command=self._send_chat).pack(fill=tk.X, pady=2)

        # Recommend section
        rec_lf = ttk.LabelFrame(left, text="Smart Recommend / 智能推荐",
                                 padding=5)
        rec_lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(rec_lf, text="Load signal in Spectrum tab, then:",
                  wraplength=300).pack(anchor=tk.W)
        ttk.Button(rec_lf, text="Analyze & Recommend / 分析推荐",
                   command=self._do_recommend).pack(fill=tk.X, pady=2)

        # Right: Output
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        ttk.Label(right, text="AI Output / 分析输出",
                  font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        self.output = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=('Consolas', 9),
            bg='#1a1a2e', fg='#e0e0e0', insertbackground='white')
        self.output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output.tag_configure('warn', foreground='#f39c12')
        self.output.tag_configure('good', foreground='#2ecc71')
        self.output.tag_configure('info', foreground='#3498db')
        self._log("AI Assistant ready. Type a filter description or click Recommend.\n", 'info')

    def _log(self, text, tag=None):
        self.output.insert(tk.END, text + '\n', tag)
        self.output.see(tk.END)

    def _send_chat(self):
        text = self.chat_entry.get().strip()
        if not text or not self.bus:
            return
        self._log(f"> {text}", 'info')
        self.bus.publish(make_event(Events.CHAT_COMMAND, source='ui',
                                    text=text, fs=8000))

    def _do_recommend(self):
        self._log("Analyzing signal for smart filter recommendation...", 'info')
        # TODO: Get current signal from ctx or bus history
        self._log("  Load a signal in Spectrum tab first.", 'warn')

    def _on_anomalies(self, event):
        anomalies = event.payload.get('anomalies', [])
        if anomalies:
            self._log(f"\n=== Anomaly Detection ({len(anomalies)}) ===", 'warn')
            for a in anomalies:
                self._log(f"  [{a['type']}] {a['desc']} (severity {a['severity']:.0%})", 'warn')

    def _on_recs(self, event):
        recs = event.payload.get('recommendations', [])
        if recs:
            self._log(f"\n=== Filter Recommendations ({len(recs)}) ===", 'good')
            for r in recs:
                self._log(f"  {r['type']}: {r['method']}", 'good')
                self._log(f"    Reason: {r['reason']}")

    def _on_chat_resp(self, event):
        text = event.payload.get('text', '')
        actions = event.payload.get('actions', [])
        self._log(f"\n{text}", 'good')
        for a in actions:
            self._log(f"  -> {a.get('desc', a.get('type', '?'))}")
