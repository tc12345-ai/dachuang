"""
Main Window — 主窗口

Tabbed application window integrating filter design, spectrum analysis,
verification, real-time streaming, pipeline editor, and scripting.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.filter_panel import FilterPanel
from gui.spectrum_panel import SpectrumPanel
from gui.realtime_panel import RealtimePanel
from gui.pipeline_panel import PipelinePanel
from gui.script_panel import ScriptPanel
from gui.dialogs import ExportDialog, AboutDialog
from gui.visualization import DSPPlotter
from io_manager.file_export import FileExporter
from io_manager.project import ProjectManager
from core.filter_design import FilterDesigner
from api.server import DSPApiServer
from core.event_bus import EventBus
from core.plugin_manager import PluginManager


class MainWindow:
    """
    Main Application Window — 应用主窗口
    
    Provides tabbed interface for filter design and spectrum analysis.
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DSP Platform — 数字滤波器设计与信号频谱分析平台")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        self.exporter = FileExporter()
        self.project_mgr = ProjectManager()
        self.api_server = DSPApiServer()
        
        # EventBus & Plugin system
        self.bus = EventBus.instance()
        self.plugin_mgr = PluginManager(self.bus)
        
        # Set icon and style
        self._setup_styles()
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Load plugins (after notebook exists)
        self._load_plugins()
        
        # Cleanup on close
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        
        # Initial status
        n_plugins = len([p for p in self.plugin_mgr.list_plugins() if p['active']])
        self._update_status(f"就绪 / Ready — {n_plugins} plugins loaded")
    
    def _setup_styles(self):
        """Configure ttk styles for premium look."""
        style = ttk.Style()
        
        # Try a modern theme
        available = style.theme_names()
        for theme in ['clam', 'alt', 'vista', 'xpnative']:
            if theme in available:
                style.theme_use(theme)
                break
        
        # Custom styles
        style.configure('TNotebook', background='#ecf0f1')
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'),
                        padding=[15, 5])
        style.map('TNotebook.Tab',
                  background=[('selected', '#2980b9')],
                  foreground=[('selected', 'white')])
        
        style.configure('TLabelframe', font=('Segoe UI', 9, 'bold'))
        style.configure('TLabelframe.Label', foreground='#2c3e50')
        
        style.configure('TButton', font=('Segoe UI', 9), padding=4)
        style.configure('Status.TLabel', font=('Segoe UI', 9),
                        background='#2c3e50', foreground='white')
        
        # Apply DSP plot style
        DSPPlotter.setup_style()
    
    def _create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件 / File", menu=file_menu)
        file_menu.add_command(label="📂 打开工程 / Open Project",
                             command=self._open_project)
        file_menu.add_command(label="💾 保存工程 / Save Project",
                             command=self._save_project)
        file_menu.add_separator()
        file_menu.add_command(label="📤 导出 / Export...",
                             command=self._show_export)
        file_menu.add_separator()
        file_menu.add_command(label="退出 / Exit",
                             command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具 / Tools", menu=tools_menu)
        tools_menu.add_command(label="生成代码 / Generate Code",
                              command=self._gen_code_menu)
        tools_menu.add_command(label="定点分析 / Fixed-Point Analysis",
                              command=self._fixed_point_menu)
        tools_menu.add_separator()
        tools_menu.add_command(label="生成 HTML 报告 / HTML Report",
                              command=self._generate_report)
        tools_menu.add_separator()
        tools_menu.add_command(label="启动 API 服务器 / Start API Server",
                              command=self._start_api)
        tools_menu.add_command(label="停止 API 服务器 / Stop API Server",
                              command=self._stop_api)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图 / View", menu=view_menu)
        view_menu.add_command(label="滤波器设计 / Filter Design",
                             command=lambda: self.notebook.select(0))
        view_menu.add_command(label="频谱分析 / Spectrum Analysis",
                             command=lambda: self.notebook.select(1))
        view_menu.add_command(label="验证对比 / Verification",
                             command=lambda: self.notebook.select(2))
        view_menu.add_separator()
        view_menu.add_command(label="实时采集 / Real-time",
                             command=lambda: self.notebook.select(3))
        view_menu.add_command(label="流程编排 / Pipeline",
                             command=lambda: self.notebook.select(4))
        view_menu.add_command(label="脚本引擎 / Scripting",
                             command=lambda: self.notebook.select(5))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助 / Help", menu=help_menu)
        help_menu.add_command(label="关于 / About",
                             command=self._show_about)
    
    def _create_main_layout(self):
        """Create main tabbed layout."""
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Filter Design
        self.filter_panel = FilterPanel(self.notebook,
                                         status_callback=self._update_status)
        self.notebook.add(self.filter_panel,
                         text="  🔧 滤波器设计 / Filter Design  ")
        
        # Tab 2: Spectrum Analysis
        self.spectrum_panel = SpectrumPanel(self.notebook,
                                            status_callback=self._update_status)
        self.notebook.add(self.spectrum_panel,
                         text="  📊 频谱分析 / Spectrum Analysis  ")
        
        # Tab 3: Verification & Comparison
        self._create_verification_tab()
        
        # Tab 4: Real-time Streaming
        self.realtime_panel = RealtimePanel(self.notebook,
                                            status_callback=self._update_status)
        self.notebook.add(self.realtime_panel,
                         text="  Real-time / 实时采集  ")
        
        # Tab 5: Pipeline Editor
        self.pipeline_panel = PipelinePanel(self.notebook,
                                            status_callback=self._update_status)
        self.notebook.add(self.pipeline_panel,
                         text="  Pipeline / 流程编排  ")
        
        # Tab 6: Scripting
        self.script_panel = ScriptPanel(self.notebook,
                                        status_callback=self._update_status)
        self.notebook.add(self.script_panel,
                         text="  Script / 脚本引擎  ")
    
    def _load_plugins(self):
        """Discover and load all plugins, add UI tabs."""
        plugins_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'plugins')
        
        manifests = self.plugin_mgr.discover(plugins_dir)
        if manifests:
            self.plugin_mgr.load_all(ctx={'status': self._update_status})
            
            # Create UI panels
            panels = self.plugin_mgr.create_ui_panels(
                self.notebook, bus=self.bus,
                ctx={'status': self._update_status})
            
            for manifest, panel in panels:
                tab_title = f"  {manifest.ui_tab_title}  "
                self.notebook.add(panel, text=tab_title)
            
            print(f"[MainWindow] Loaded {len(panels)} plugin panels")
    
    def _create_verification_tab(self):
        """Create verification and comparison tab."""
        verify_frame = ttk.Frame(self.notebook)
        self.notebook.add(verify_frame,
                         text="  ✅ 验证对比 / Verification  ")
        
        # Horizontal split
        paned = ttk.PanedWindow(verify_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Controls
        left = ttk.Frame(paned, width=320)
        paned.add(left, weight=0)
        
        # Apply filter section
        sec1 = ttk.LabelFrame(left, text="滤波验证 / Filter Verification",
                               padding=10)
        sec1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sec1, text="将当前滤波器应用到频谱分析中的信号:",
                  wraplength=280).pack(anchor=tk.W)
        
        self.verify_mode = tk.StringVar(value='lfilter')
        ttk.Radiobutton(sec1, text="因果滤波 / Causal (lfilter)",
                        variable=self.verify_mode,
                        value='lfilter').pack(anchor=tk.W)
        ttk.Radiobutton(sec1, text="零相位滤波 / Zero-Phase (filtfilt)",
                        variable=self.verify_mode,
                        value='filtfilt').pack(anchor=tk.W)
        
        ttk.Button(sec1, text="▶ 应用滤波 / Apply Filter",
                   command=self._apply_filter_verify).pack(fill=tk.X, pady=5)
        
        # Verification checks
        sec2 = ttk.LabelFrame(left, text="自动验收 / Auto Verification",
                               padding=10)
        sec2.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(sec2, text="🔍 检查通带纹波 / Passband Ripple",
                   command=self._check_passband).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="🔍 检查阻带衰减 / Stopband Atten.",
                   command=self._check_stopband).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="🔍 稳定性检查 / Stability Check",
                   command=self._check_stability).pack(fill=tk.X, pady=2)
        ttk.Button(sec2, text="📋 完整验收报告 / Full Report",
                   command=self._full_verification).pack(fill=tk.X, pady=5)
        
        # Verification results
        self.verify_text = tk.Text(left, height=18, wrap=tk.WORD,
                                    font=('Consolas', 8), state=tk.DISABLED)
        self.verify_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right: Plot area
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        
        self.verify_fig = Figure(figsize=(10, 8), dpi=100)
        self.verify_fig.set_facecolor('#fafafa')
        
        self.verify_ax1 = self.verify_fig.add_subplot(2, 2, 1)
        self.verify_ax2 = self.verify_fig.add_subplot(2, 2, 2)
        self.verify_ax3 = self.verify_fig.add_subplot(2, 2, 3)
        self.verify_ax4 = self.verify_fig.add_subplot(2, 2, 4)
        
        self.verify_fig.tight_layout(pad=3.0)
        
        self.verify_canvas = FigureCanvasTkAgg(self.verify_fig, right)
        self.verify_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.verify_canvas, toolbar_frame)
        
        # Empty initial state
        for ax in [self.verify_ax1, self.verify_ax2,
                   self.verify_ax3, self.verify_ax4]:
            ax.text(0.5, 0.5, '应用滤波器后查看结果\nApply filter to see results',
                    ha='center', va='center', fontsize=11, color='#bdc3c7',
                    transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
        self.verify_canvas.draw()
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_frame = tk.Frame(self.root, bg='#2c3e50', height=28)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame, text="就绪 / Ready",
            bg='#2c3e50', fg='white', font=('Segoe UI', 9),
            anchor=tk.W, padx=10)
        self.status_label.pack(fill=tk.BOTH, expand=True)
    
    def _update_status(self, message: str):
        """Update status bar text."""
        self.status_label.config(text=f"📊 {message}")
        self.root.update_idletasks()
    
    # === File Operations ===
    
    def _save_project(self):
        """Save current project state."""
        filepath = filedialog.asksaveasfilename(
            defaultextension='.dspproj',
            filetypes=[("DSP Project", "*.dspproj"), ("All Files", "*.*")],
            title="保存工程 / Save Project")
        if not filepath:
            return
        
        try:
            state = self._collect_state()
            self.project_mgr.save_project(filepath, state)
            self.project_mgr.add_to_recent(filepath)
            self._update_status(f"工程已保存: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("保存错误", str(e))
    
    def _open_project(self):
        """Open project file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("DSP Project", "*.dspproj"), ("All Files", "*.*")],
            title="打开工程 / Open Project")
        if not filepath:
            return
        
        try:
            state = self.project_mgr.load_project(filepath)
            self._restore_state(state)
            self._update_status(f"工程已加载: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("加载错误", str(e))
    
    def _collect_state(self) -> dict:
        """Collect current application state for saving."""
        state = {
            'active_tab': self.notebook.index(self.notebook.select()),
        }
        
        # Filter state
        fp = self.filter_panel
        state['filter'] = {
            'filter_class': fp.filter_class.get(),
            'design_method': fp.design_method.get(),
            'filter_type': fp.filter_type.get(),
            'order': fp.order_var.get(),
            'fs': fp.fs_var.get(),
            'fp1': fp.fp1_var.get(),
            'fp2': fp.fp2_var.get(),
            'fs1': fp.fs1_var.get(),
            'fs2': fp.fs2_var.get(),
            'ripple': fp.ripple_var.get(),
            'atten': fp.atten_var.get(),
            'window_type': fp.window_type.get(),
            'kaiser_beta': fp.kaiser_beta_var.get(),
        }
        
        # Save coefficients if available
        if fp.current_result is not None:
            state['filter']['b'] = fp.current_result.b
            state['filter']['a'] = fp.current_result.a
        
        # Spectrum state
        sp = self.spectrum_panel
        state['spectrum'] = {
            'fs': sp.sig_fs.get(),
            'nfft': sp.nfft.get(),
            'window': sp.window.get(),
            'nperseg': sp.nperseg.get(),
            'overlap': sp.overlap_pct.get(),
            'test_type': sp.test_type.get(),
            'test_freq': sp.test_freq.get(),
            'test_duration': sp.test_duration.get(),
        }
        
        return state
    
    def _restore_state(self, state: dict):
        """Restore application state from project."""
        # Filter state
        fs = state.get('filter', {})
        fp = self.filter_panel
        if fs:
            fp.filter_class.set(fs.get('filter_class', 'FIR'))
            fp._on_class_changed()
            fp.filter_type.set(fs.get('filter_type', 'lowpass'))
            fp._on_type_changed()
            fp.order_var.set(fs.get('order', 32))
            fp.fs_var.set(fs.get('fs', 8000))
            fp.fp1_var.set(fs.get('fp1', 1000))
            fp.fp2_var.set(fs.get('fp2', 2000))
            fp.fs1_var.set(fs.get('fs1', 1500))
            fp.fs2_var.set(fs.get('fs2', 500))
            fp.ripple_var.set(fs.get('ripple', 1.0))
            fp.atten_var.set(fs.get('atten', 60.0))
            fp.window_type.set(fs.get('window_type', 'hamming'))
            fp.kaiser_beta_var.set(fs.get('kaiser_beta', 5.0))
        
        # Spectrum state
        ss = state.get('spectrum', {})
        sp = self.spectrum_panel
        if ss:
            sp.sig_fs.set(ss.get('fs', 8000))
            sp.nfft.set(ss.get('nfft', 2048))
            sp.window.set(ss.get('window', 'Hann'))
            sp.nperseg.set(ss.get('nperseg', 256))
            sp.overlap_pct.set(ss.get('overlap', 75))
            sp.test_type.set(ss.get('test_type', 'sine'))
            sp.test_freq.set(ss.get('test_freq', 440))
            sp.test_duration.set(ss.get('test_duration', 0.1))
        
        # Switch to saved tab
        tab = state.get('active_tab', 0)
        self.notebook.select(tab)
    
    # === Export ===
    
    def _show_export(self):
        """Show export dialog."""
        def do_export(export_type, filepath):
            try:
                self._perform_export(export_type, filepath)
                self._update_status(f"导出完成: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("导出错误", str(e))
        
        ExportDialog(self.root, do_export)
    
    def _perform_export(self, export_type: str, filepath: str):
        """Perform the actual export."""
        fr = self.filter_panel.get_current_result()
        sr = self.spectrum_panel.get_spectrum_result()
        
        if export_type.startswith('coefficients') and fr:
            fmt = export_type.split('_')[1]
            self.exporter.export_coefficients(
                filepath, fr.b, fr.a, fr.spec.fs, fmt=fmt,
                info=fr.info, sos=fr.sos)
            
        elif export_type == 'freq_response' and fr:
            self.exporter.export_frequency_response(
                filepath, fr.freq_hz, fr.magnitude_db,
                fr.phase_deg, fr.group_delay)
            
        elif export_type == 'spectrum_data' and sr:
            psd_db = None
            pr = self.spectrum_panel.psd_result
            if pr and pr.psd_db is not None:
                psd_db = pr.psd_db
            self.exporter.export_spectrum_data(
                filepath, sr.freq_hz, sr.magnitude_db, psd_db)
            
        elif export_type in ('chart_png', 'chart_svg'):
            # Find the active tab's figure
            tab = self.notebook.index(self.notebook.select())
            if tab == 0 and hasattr(self.filter_panel, 'fig'):
                self.exporter.save_figure(self.filter_panel.fig, filepath)
            elif tab == 1 and hasattr(self.spectrum_panel, 'fig'):
                self.exporter.save_figure(self.spectrum_panel.fig, filepath)
            elif tab == 2:
                self.exporter.save_figure(self.verify_fig, filepath)
            
        elif export_type == 'html_report':
            self._generate_report_to(filepath)
        else:
            messagebox.showinfo("提示", "没有可导出的数据。请先设计滤波器或分析信号。")
    
    def _generate_report(self):
        """Generate HTML report."""
        filepath = filedialog.asksaveasfilename(
            defaultextension='.html',
            filetypes=[("HTML", "*.html"), ("All Files", "*.*")],
            title="保存报告 / Save Report")
        if filepath:
            try:
                self._generate_report_to(filepath)
                self._update_status(f"报告已生成: {os.path.basename(filepath)}")
                # Ask to open
                if messagebox.askyesno("提示", "报告已生成。是否打开？"):
                    os.startfile(filepath)
            except Exception as e:
                messagebox.showerror("报告错误", str(e))
    
    def _generate_report_to(self, filepath: str):
        """Generate report to specific path."""
        sections = []
        chart_paths = []
        
        fr = self.filter_panel.get_current_result()
        if fr:
            sections.append({
                'title': '滤波器设计 / Filter Design',
                'content': [
                    {'参数': '类型', '值': fr.info},
                    {'参数': '阶数', '值': str(fr.order)},
                    {'参数': '稳定性', '值': '✅ 稳定' if fr.is_stable else '❌ 不稳定'},
                    {'参数': '系数 b 数量', '值': str(len(fr.b))},
                    {'参数': '系数 a 数量', '值': str(len(fr.a))},
                ]
            })
            
            # Save filter chart
            chart_path = filepath.replace('.html', '_filter.png')
            self.exporter.save_figure(self.filter_panel.fig, chart_path)
            chart_paths.append(chart_path)
        
        mr = self.spectrum_panel.measurement_result if hasattr(
            self.spectrum_panel, 'measurement_result') else None
        if mr and mr is not None:
            sections.append({
                'title': '信号测量 / Signal Measurements',
                'content': [
                    {'指标': '基频', '值': f'{mr.fundamental_freq:.1f} Hz'},
                    {'指标': 'THD', '值': f'{mr.thd_percent:.4f}% ({mr.thd_db:.1f} dB)'},
                    {'指标': 'SNR', '值': f'{mr.snr_db:.1f} dB'},
                    {'指标': 'SFDR', '值': f'{mr.sfdr_db:.1f} dB'},
                    {'指标': 'SINAD', '值': f'{mr.sinad_db:.1f} dB'},
                    {'指标': 'ENOB', '值': f'{mr.enob:.2f} bits'},
                ]
            })
            
            chart_path2 = filepath.replace('.html', '_spectrum.png')
            self.exporter.save_figure(self.spectrum_panel.fig, chart_path2)
            chart_paths.append(chart_path2)
        
        if not sections:
            sections.append({'title': '提示', 'content': '请先设计滤波器或分析信号后再生成报告。'})
        
        self.exporter.generate_html_report(
            filepath,
            title="DSP 分析报告 / DSP Analysis Report",
            sections=sections,
            chart_paths=chart_paths)
    
    # === Menu callbacks ===
    
    def _gen_code_menu(self):
        """Code generation from menu."""
        self.notebook.select(0)
        self.filter_panel._show_code_gen()
    
    def _fixed_point_menu(self):
        """Fixed-point analysis from menu."""
        self.notebook.select(0)
        self.filter_panel._show_fixed_point()
    
    def _show_about(self):
        """Show about dialog."""
        AboutDialog(self.root)
    
    # === API Server ===
    
    def _start_api(self):
        """Start REST API server."""
        if self.api_server.is_running:
            messagebox.showinfo("Info", f"API server already running at {self.api_server.url}")
            return
        try:
            url = self.api_server.start()
            self._update_status(f"API server started at {url}")
            messagebox.showinfo("API Server",
                f"REST API server started!\n\n"
                f"URL: {url}\n\n"
                f"Endpoints:\n"
                f"  GET  /api/status\n"
                f"  POST /api/filter/design\n"
                f"  POST /api/filter/codegen\n"
                f"  POST /api/spectrum/fft\n"
                f"  POST /api/spectrum/psd\n"
                f"  POST /api/spectrum/measure")
        except Exception as e:
            messagebox.showerror("API Error", str(e))
    
    def _stop_api(self):
        """Stop REST API server."""
        self.api_server.stop()
        self._update_status("API server stopped")
    
    def _on_close(self):
        """Cleanup on window close."""
        self.api_server.stop()
        if hasattr(self, 'realtime_panel'):
            try:
                self.realtime_panel._stop()
            except Exception:
                pass
        self.root.destroy()
    
    # === Verification Tab ===
    
    def _apply_filter_verify(self):
        """Apply current filter to loaded signal and show before/after."""
        fr = self.filter_panel.get_current_result()
        sd = self.spectrum_panel.get_signal_data()
        
        if fr is None:
            messagebox.showinfo("提示", "请先设计滤波器 (Tab 1)")
            return
        if sd is None:
            messagebox.showinfo("提示", "请先加载/生成信号 (Tab 2)")
            return
        
        try:
            signal_in = sd.channel_data
            fs = sd.fs
            designer = FilterDesigner()
            
            mode = self.verify_mode.get()
            if mode == 'filtfilt':
                signal_out = designer.apply_filtfilt(signal_in, fr)
            else:
                signal_out = designer.apply_filter(signal_in, fr)
            
            t = np.arange(len(signal_in)) / fs
            
            # Plot 1: Time domain comparison
            self.verify_ax1.clear()
            self.verify_ax1.plot(t, signal_in, alpha=0.6, linewidth=0.7,
                               label='原始 / Original', color='#bdc3c7')
            self.verify_ax1.plot(t, signal_out, linewidth=0.9,
                               label='滤波后 / Filtered', color='#2980b9')
            self.verify_ax1.set_title('时域对比 / Time Domain', fontweight='bold')
            self.verify_ax1.set_xlabel('Time (s)')
            self.verify_ax1.set_ylabel('Amplitude')
            self.verify_ax1.legend(fontsize=7)
            self.verify_ax1.grid(True, alpha=0.3)
            
            # Plot 2: Spectrum comparison
            from scipy.fft import fft
            n = len(signal_in)
            nfft = max(n, 2048)
            n_half = nfft // 2 + 1
            freq = np.linspace(0, fs / 2, n_half)
            
            from scipy.signal import get_window
            win = get_window('hann', n)
            win_sum = np.sum(win)
            
            X_in = np.abs(fft(signal_in * win, n=nfft))[:n_half] * 2 / win_sum
            X_out = np.abs(fft(signal_out * win, n=nfft))[:n_half] * 2 / win_sum
            
            X_in_db = 20 * np.log10(np.maximum(X_in, 1e-12))
            X_out_db = 20 * np.log10(np.maximum(X_out, 1e-12))
            
            self.verify_ax2.clear()
            self.verify_ax2.plot(freq, X_in_db, alpha=0.6, linewidth=0.7,
                               label='原始', color='#bdc3c7')
            self.verify_ax2.plot(freq, X_out_db, linewidth=0.9,
                               label='滤波后', color='#e74c3c')
            self.verify_ax2.set_title('频谱对比 / Spectrum', fontweight='bold')
            self.verify_ax2.set_xlabel('Frequency (Hz)')
            self.verify_ax2.set_ylabel('Magnitude (dB)')
            self.verify_ax2.legend(fontsize=7)
            self.verify_ax2.grid(True, alpha=0.3)
            
            # Plot 3: Filter frequency response
            DSPPlotter.plot_magnitude_response(
                self.verify_ax3, fr.freq_hz, fr.magnitude_db,
                title="滤波器响应 / Filter Response")
            
            # Plot 4: Error / difference
            min_len = min(len(signal_in), len(signal_out))
            error = signal_out[:min_len] - signal_in[:min_len]
            t_err = t[:min_len]
            
            self.verify_ax4.clear()
            self.verify_ax4.plot(t_err, error, color='#e74c3c', linewidth=0.7)
            self.verify_ax4.set_title('差异 / Difference', fontweight='bold')
            self.verify_ax4.set_xlabel('Time (s)')
            self.verify_ax4.set_ylabel('Amplitude')
            self.verify_ax4.grid(True, alpha=0.3)
            
            self.verify_fig.tight_layout(pad=2.5)
            self.verify_canvas.draw()
            
            self._update_status("滤波验证完成 — 查看时域/频域对比")
            
        except Exception as e:
            messagebox.showerror("验证错误", str(e))
    
    def _check_passband(self):
        """Check passband ripple."""
        fr = self.filter_panel.get_current_result()
        if fr is None:
            messagebox.showinfo("提示", "请先设计滤波器")
            return
        
        spec = fr.spec
        nyq = spec.fs / 2
        
        # Find passband region
        if spec.filter_type == 'lowpass':
            mask = fr.freq_hz <= spec.fp1
        elif spec.filter_type == 'highpass':
            mask = fr.freq_hz >= spec.fp1
        elif spec.filter_type == 'bandpass':
            mask = (fr.freq_hz >= spec.fp1) & (fr.freq_hz <= spec.fp2)
        elif spec.filter_type == 'bandstop':
            mask = (fr.freq_hz <= spec.fp1) | (fr.freq_hz >= spec.fp2)
        else:
            mask = np.ones(len(fr.freq_hz), dtype=bool)
        
        if np.any(mask):
            pb_mag = fr.magnitude_db[mask]
            ripple = np.max(pb_mag) - np.min(pb_mag)
            max_db = np.max(pb_mag)
            min_db = np.min(pb_mag)
            
            result = (f"═══ 通带纹波检查 ═══\n"
                     f"通带范围: see filter type\n"
                     f"最大增益: {max_db:.4f} dB\n"
                     f"最小增益: {min_db:.4f} dB\n"
                     f"纹波: {ripple:.4f} dB\n"
                     f"目标纹波: {spec.passband_ripple:.4f} dB\n"
                     f"结果: {'✅ 通过' if ripple <= spec.passband_ripple * 1.1 else '❌ 未通过'}")
        else:
            result = "无法确定通带区域"
        
        self._show_verify_result(result)
    
    def _check_stopband(self):
        """Check stopband attenuation."""
        fr = self.filter_panel.get_current_result()
        if fr is None:
            messagebox.showinfo("提示", "请先设计滤波器")
            return
        
        spec = fr.spec
        
        if spec.filter_type == 'lowpass':
            mask = fr.freq_hz >= spec.fs1
        elif spec.filter_type == 'highpass':
            mask = (fr.freq_hz <= spec.fs1) & (fr.freq_hz > 0)
        elif spec.filter_type == 'bandpass':
            mask = (fr.freq_hz <= spec.fs2) | (fr.freq_hz >= spec.fs1)
        elif spec.filter_type == 'bandstop':
            mask = (fr.freq_hz >= spec.fs2) & (fr.freq_hz <= spec.fs1)
        else:
            mask = np.zeros(len(fr.freq_hz), dtype=bool)
        
        if np.any(mask):
            sb_mag = fr.magnitude_db[mask]
            max_sb = np.max(sb_mag)
            atten = -max_sb
            
            result = (f"═══ 阻带衰减检查 ═══\n"
                     f"阻带最大增益: {max_sb:.2f} dB\n"
                     f"实际衰减: {atten:.2f} dB\n"
                     f"目标衰减: {spec.stopband_atten:.2f} dB\n"
                     f"结果: {'✅ 通过' if atten >= spec.stopband_atten * 0.9 else '❌ 未通过'}")
        else:
            result = "无法确定阻带区域"
        
        self._show_verify_result(result)
    
    def _check_stability(self):
        """Check filter stability."""
        fr = self.filter_panel.get_current_result()
        if fr is None:
            messagebox.showinfo("提示", "请先设计滤波器")
            return
        
        lines = ["═══ 稳定性检查 ═══"]
        lines.append(f"滤波器: {fr.info}")
        
        if fr.poles is not None and len(fr.poles) > 0:
            pole_mags = np.abs(fr.poles)
            max_pole = np.max(pole_mags)
            lines.append(f"极点数: {len(fr.poles)}")
            lines.append(f"最大极点幅度: {max_pole:.8f}")
            lines.append(f"稳定性: {'✅ 稳定 (所有极点在单位圆内)' if max_pole < 1.0 else '❌ 不稳定'}")
            lines.append(f"\n极点位置:")
            for i, p in enumerate(fr.poles):
                lines.append(f"  p[{i}] = {p:.6f}  |p| = {abs(p):.6f}")
        else:
            lines.append("FIR 滤波器 — 恒定稳定 ✅")
        
        self._show_verify_result('\n'.join(lines))
    
    def _full_verification(self):
        """Run full verification report."""
        fr = self.filter_panel.get_current_result()
        if fr is None:
            messagebox.showinfo("提示", "请先设计滤波器")
            return
        
        lines = ["═══════════════════════════════════"]
        lines.append("      完整验收报告 / Full Report")
        lines.append("═══════════════════════════════════\n")
        lines.append(f"滤波器: {fr.info}")
        lines.append(f"阶数: {fr.order}")
        lines.append(f"稳定: {'✅' if fr.is_stable else '❌'}")
        
        if fr.poles is not None and len(fr.poles) > 0:
            lines.append(f"最大极点: {np.max(np.abs(fr.poles)):.8f}")
        
        spec = fr.spec
        
        # Passband check
        if spec.filter_type == 'lowpass':
            mask = fr.freq_hz <= spec.fp1
        elif spec.filter_type == 'highpass':
            mask = fr.freq_hz >= spec.fp1
        elif spec.filter_type == 'bandpass':
            mask = (fr.freq_hz >= spec.fp1) & (fr.freq_hz <= spec.fp2)
        else:
            mask = np.ones(len(fr.freq_hz), dtype=bool)
        
        if np.any(mask):
            pb = fr.magnitude_db[mask]
            ripple = np.max(pb) - np.min(pb)
            lines.append(f"\n通带纹波: {ripple:.4f} dB (目标 ≤ {spec.passband_ripple} dB)")
            lines.append(f"  → {'✅ 通过' if ripple <= spec.passband_ripple * 1.1 else '❌ 未通过'}")
        
        # Stopband check
        if spec.filter_type == 'lowpass':
            sb_mask = fr.freq_hz >= spec.fs1
        elif spec.filter_type == 'highpass':
            sb_mask = (fr.freq_hz <= spec.fs1) & (fr.freq_hz > 0)
        elif spec.filter_type == 'bandpass':
            sb_mask = (fr.freq_hz <= spec.fs2) | (fr.freq_hz >= spec.fs1)
        else:
            sb_mask = np.zeros(len(fr.freq_hz), dtype=bool)
        
        if np.any(sb_mask):
            sb = fr.magnitude_db[sb_mask]
            atten = -np.max(sb)
            lines.append(f"\n阻带衰减: {atten:.2f} dB (目标 ≥ {spec.stopband_atten} dB)")
            lines.append(f"  → {'✅ 通过' if atten >= spec.stopband_atten * 0.9 else '❌ 未通过'}")
        
        # Group delay
        if fr.group_delay is not None:
            gd = fr.group_delay
            gd_valid = gd[np.isfinite(gd)]
            if len(gd_valid) > 0:
                gd_var = np.max(gd_valid) - np.min(gd_valid)
                lines.append(f"\n群时延变化: {gd_var:.2f} samples")
                if spec.filter_class == 'FIR':
                    lines.append(f"  (FIR 线性相位: 群时延应近似恒定)")
        
        # Coefficients summary
        lines.append(f"\n系数 b: {len(fr.b)} 个, 范围 [{np.min(fr.b):.6f}, {np.max(fr.b):.6f}]")
        if len(fr.a) > 1:
            lines.append(f"系数 a: {len(fr.a)} 个, 范围 [{np.min(fr.a):.6f}, {np.max(fr.a):.6f}]")
        
        self._show_verify_result('\n'.join(lines))
    
    def _show_verify_result(self, text):
        """Show verification result text."""
        self.verify_text.config(state=tk.NORMAL)
        self.verify_text.delete('1.0', tk.END)
        self.verify_text.insert(tk.END, text)
        self.verify_text.config(state=tk.DISABLED)


def run_app():
    """Run the application."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    run_app()
