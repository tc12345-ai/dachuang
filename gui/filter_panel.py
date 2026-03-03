"""
Filter Design Panel — 滤波器设计面板

Complete filter design interface with parameter controls and 4-subplot visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filter_design import FilterDesigner, FilterSpec, FilterResult
from core.code_generator import CodeGenerator
from core.fixed_point import FixedPointAnalyzer
from gui.visualization import DSPPlotter
from gui.dialogs import CodeGenDialog, FixedPointDialog


class FilterPanel(ttk.Frame):
    """
    Filter Design Panel — 滤波器设计面板
    
    Left: parameter controls, Right: 4-subplot visualization
    """
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.designer = FilterDesigner()
        self.code_gen = CodeGenerator()
        self.current_result = None
        self.comparison_results = []  # For multi-filter comparison
        
        DSPPlotter.setup_style()
        self._create_ui()
    
    def _create_ui(self):
        """Build the panel UI."""
        # Main horizontal paned window
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # ---- Left Panel: Parameters ----
        left_frame = ttk.Frame(paned, width=320)
        paned.add(left_frame, weight=0)
        
        # Scrollable parameter area
        canvas = tk.Canvas(left_frame, width=310, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL,
                                   command=canvas.yview)
        self.param_frame = ttk.Frame(canvas)
        
        self.param_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.param_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        self._create_param_controls(self.param_frame)
        
        # ---- Right Panel: Plots ----
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        self._create_plot_area(right_frame)
    
    def _create_param_controls(self, parent):
        """Create parameter input controls."""
        pad = {'padx': 5, 'pady': 2}
        
        # === Filter Class ===
        sec = ttk.LabelFrame(parent, text="滤波器类别 / Filter Class", padding=5)
        sec.pack(fill=tk.X, **pad)
        
        self.filter_class = tk.StringVar(value='FIR')
        row = ttk.Frame(sec)
        row.pack(fill=tk.X)
        ttk.Radiobutton(row, text='FIR', variable=self.filter_class,
                        value='FIR', command=self._on_class_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text='IIR', variable=self.filter_class,
                        value='IIR', command=self._on_class_changed).pack(side=tk.LEFT)
        
        # === Design Method ===
        sec2 = ttk.LabelFrame(parent, text="设计方法 / Design Method", padding=5)
        sec2.pack(fill=tk.X, **pad)
        
        self.design_method = tk.StringVar(value='window')
        self.method_combo = ttk.Combobox(sec2, textvariable=self.design_method,
                                          state='readonly', width=25)
        self.method_combo.pack(fill=tk.X)
        self._update_methods()
        
        # === Filter Type ===
        sec3 = ttk.LabelFrame(parent, text="滤波器类型 / Filter Type", padding=5)
        sec3.pack(fill=tk.X, **pad)
        
        self.filter_type = tk.StringVar(value='lowpass')
        types = [('低通 / Lowpass', 'lowpass'), ('高通 / Highpass', 'highpass'),
                 ('带通 / Bandpass', 'bandpass'), ('带阻 / Bandstop', 'bandstop'),
                 ('陷波 / Notch', 'notch')]
        for text, val in types:
            ttk.Radiobutton(sec3, text=text, variable=self.filter_type,
                           value=val, command=self._on_type_changed).pack(anchor=tk.W)
        
        # === Frequency Parameters ===
        sec4 = ttk.LabelFrame(parent, text="频率参数 / Frequency (Hz)", padding=5)
        sec4.pack(fill=tk.X, **pad)
        
        self.fs_var = tk.DoubleVar(value=8000.0)
        self._add_param_row(sec4, "采样率 Fs:", self.fs_var)
        
        self.fp1_var = tk.DoubleVar(value=1000.0)
        self.fp1_label = self._add_param_row(sec4, "通带频率 Fp1:", self.fp1_var)
        
        self.fp2_var = tk.DoubleVar(value=2000.0)
        self.fp2_label = self._add_param_row(sec4, "通带频率 Fp2:", self.fp2_var)
        
        self.fs1_var = tk.DoubleVar(value=1500.0)
        self.fs1_label = self._add_param_row(sec4, "阻带频率 Fs1:", self.fs1_var)
        
        self.fs2_var = tk.DoubleVar(value=500.0)
        self.fs2_label = self._add_param_row(sec4, "阻带频率 Fs2:", self.fs2_var)
        
        # Notch parameters
        self.notch_frame = ttk.Frame(sec4)
        self.notch_freq_var = tk.DoubleVar(value=50.0)
        self._add_param_row(self.notch_frame, "陷波频率:", self.notch_freq_var)
        self.notch_bw_var = tk.DoubleVar(value=10.0)
        self._add_param_row(self.notch_frame, "陷波带宽:", self.notch_bw_var)
        
        # === Performance Specs ===
        sec5 = ttk.LabelFrame(parent, text="性能指标 / Performance", padding=5)
        sec5.pack(fill=tk.X, **pad)
        
        self.order_var = tk.IntVar(value=32)
        self._add_param_row(sec5, "阶数 Order:", self.order_var)
        
        self.ripple_var = tk.DoubleVar(value=1.0)
        self._add_param_row(sec5, "通带纹波 Rp (dB):", self.ripple_var)
        
        self.atten_var = tk.DoubleVar(value=60.0)
        self._add_param_row(sec5, "阻带衰减 Rs (dB):", self.atten_var)
        
        # === Window (FIR) ===
        self.window_frame = ttk.LabelFrame(parent, text="窗函数 / Window", padding=5)
        self.window_frame.pack(fill=tk.X, **pad)
        
        self.window_type = tk.StringVar(value='hamming')
        windows = ['hamming', 'hann', 'blackman', 'bartlett', 'kaiser',
                    'rectangular', 'flattop', 'nuttall', 'blackmanharris']
        self.window_combo = ttk.Combobox(self.window_frame,
                                          textvariable=self.window_type,
                                          values=windows, state='readonly',
                                          width=20)
        self.window_combo.pack(fill=tk.X)
        
        self.kaiser_frame = ttk.Frame(self.window_frame)
        self.kaiser_beta_var = tk.DoubleVar(value=5.0)
        self._add_param_row(self.kaiser_frame, "Kaiser β:", self.kaiser_beta_var)
        self.window_combo.bind('<<ComboboxSelected>>', self._on_window_changed)
        
        # === Action Buttons ===
        btn_frame = ttk.LabelFrame(parent, text="操作 / Actions", padding=5)
        btn_frame.pack(fill=tk.X, **pad)
        
        ttk.Button(btn_frame, text="🔧 估算阶数 / Estimate Order",
                   command=self._estimate_order).pack(fill=tk.X, pady=2)
        
        design_btn = ttk.Button(btn_frame, text="▶ 设计滤波器 / Design Filter",
                                command=self.design_filter)
        design_btn.pack(fill=tk.X, pady=2)
        
        ttk.Separator(btn_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="📝 生成代码 / Generate Code",
                   command=self._show_code_gen).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🔢 定点分析 / Fixed-Point",
                   command=self._show_fixed_point).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="➕ 添加到对比 / Add to Compare",
                   command=self._add_to_comparison).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🗑️ 清除对比 / Clear Compare",
                   command=self._clear_comparison).pack(fill=tk.X, pady=2)
        
        # === Info Display ===
        info_frame = ttk.LabelFrame(parent, text="滤波器信息 / Filter Info", padding=5)
        info_frame.pack(fill=tk.X, **pad)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD,
                                 font=('Consolas', 8), state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)
        
        self._on_type_changed()
    
    def _add_param_row(self, parent, label_text, variable):
        """Add a parameter row with label and entry."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        label = ttk.Label(row, text=label_text, width=20)
        label.pack(side=tk.LEFT)
        
        if isinstance(variable, tk.IntVar):
            entry = ttk.Spinbox(row, from_=1, to=1000,
                                textvariable=variable, width=12)
        else:
            entry = ttk.Entry(row, textvariable=variable, width=14)
        entry.pack(side=tk.LEFT, padx=5)
        return label
    
    def _create_plot_area(self, parent):
        """Create the 4-subplot visualization area."""
        # Figure with 2x2 subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.set_facecolor('#fafafa')
        
        self.ax_mag = self.fig.add_subplot(2, 2, 1)
        self.ax_phase = self.fig.add_subplot(2, 2, 2)
        self.ax_gd = self.fig.add_subplot(2, 2, 3)
        self.ax_pz = self.fig.add_subplot(2, 2, 4)
        
        self.fig.tight_layout(pad=3.0)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # View toggle buttons
        view_frame = ttk.Frame(toolbar_frame)
        view_frame.pack(side=tk.RIGHT)
        
        self.view_mode = tk.StringVar(value='standard')
        ttk.Radiobutton(view_frame, text="标准/Standard",
                        variable=self.view_mode, value='standard',
                        command=self._update_plots).pack(side=tk.LEFT)
        ttk.Radiobutton(view_frame, text="脉冲+阶跃/Impulse+Step",
                        variable=self.view_mode, value='time',
                        command=self._update_plots).pack(side=tk.LEFT)
        ttk.Radiobutton(view_frame, text="对比/Compare",
                        variable=self.view_mode, value='compare',
                        command=self._update_plots).pack(side=tk.LEFT)
        
        # Initial empty plots
        self._draw_empty_plots()
    
    def _draw_empty_plots(self):
        """Draw placeholder plots."""
        for ax in [self.ax_mag, self.ax_phase, self.ax_gd, self.ax_pz]:
            ax.clear()
            ax.set_facecolor('#fafafa')
            ax.text(0.5, 0.5, '请设计滤波器\nDesign a filter first',
                    ha='center', va='center', fontsize=12, color='#bdc3c7',
                    transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
        self.canvas.draw()
    
    def _on_class_changed(self):
        """Handle filter class change (FIR/IIR)."""
        self._update_methods()
        is_fir = self.filter_class.get() == 'FIR'
        if is_fir:
            self.window_frame.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.window_frame.pack_forget()
    
    def _update_methods(self):
        """Update available design methods."""
        if self.filter_class.get() == 'FIR':
            methods = [('窗函数法 / Window', 'window'),
                       ('频率采样 / Freq Sampling', 'freq_sampling'),
                       ('等波纹 / Equiripple (PM)', 'equiripple')]
            self.method_combo['values'] = [m[0] for m in methods]
            self._method_map = {m[0]: m[1] for m in methods}
            self.method_combo.set(methods[0][0])
        else:
            methods = [('Butterworth', 'butter'),
                       ('Chebyshev I', 'cheby1'),
                       ('Chebyshev II', 'cheby2'),
                       ('Elliptic (Cauer)', 'ellip'),
                       ('Bessel', 'bessel')]
            self.method_combo['values'] = [m[0] for m in methods]
            self._method_map = {m[0]: m[1] for m in methods}
            self.method_combo.set(methods[0][0])
    
    def _on_type_changed(self):
        """Handle filter type change."""
        ftype = self.filter_type.get()
        
        if ftype in ('lowpass', 'highpass'):
            self.fp2_label.master.pack_forget() if hasattr(self, 'fp2_label') else None
            self.fs2_label.master.pack_forget() if hasattr(self, 'fs2_label') else None
            self.notch_frame.pack_forget()
            self.fp1_label.master.pack(fill=tk.X, pady=1)
            self.fs1_label.master.pack(fill=tk.X, pady=1)
        elif ftype in ('bandpass', 'bandstop'):
            self.fp1_label.master.pack(fill=tk.X, pady=1)
            self.fp2_label.master.pack(fill=tk.X, pady=1)
            self.fs1_label.master.pack(fill=tk.X, pady=1)
            self.fs2_label.master.pack(fill=tk.X, pady=1)
            self.notch_frame.pack_forget()
        elif ftype == 'notch':
            self.fp1_label.master.pack_forget()
            self.fp2_label.master.pack_forget()
            self.fs1_label.master.pack_forget()
            self.fs2_label.master.pack_forget()
            self.notch_frame.pack(fill=tk.X, pady=1)
    
    def _on_window_changed(self, event=None):
        """Handle window type change."""
        if self.window_type.get() == 'kaiser':
            self.kaiser_frame.pack(fill=tk.X, pady=2)
        else:
            self.kaiser_frame.pack_forget()
    
    def _build_spec(self) -> FilterSpec:
        """Build FilterSpec from UI values."""
        method_display = self.design_method.get()
        method = self._method_map.get(method_display, 'window')
        
        spec = FilterSpec(
            filter_class=self.filter_class.get(),
            design_method=method,
            filter_type=self.filter_type.get(),
            order=self.order_var.get(),
            fs=self.fs_var.get(),
            fp1=self.fp1_var.get(),
            fp2=self.fp2_var.get(),
            fs1=self.fs1_var.get(),
            fs2=self.fs2_var.get(),
            passband_ripple=self.ripple_var.get(),
            stopband_atten=self.atten_var.get(),
            window_type=self.window_type.get(),
            kaiser_beta=self.kaiser_beta_var.get(),
            notch_freq=self.notch_freq_var.get(),
            notch_bw=self.notch_bw_var.get(),
        )
        return spec
    
    def _estimate_order(self):
        """Estimate required filter order."""
        try:
            spec = self._build_spec()
            order = FilterDesigner.estimate_order(spec)
            self.order_var.set(order)
            self._update_status(f"估算阶数: {order}")
        except Exception as e:
            messagebox.showerror("错误", f"阶数估算失败: {e}")
    
    def design_filter(self):
        """Design filter with current parameters."""
        try:
            spec = self._build_spec()
            result = self.designer.design(spec)
            self.current_result = result
            
            self._update_plots()
            self._update_info(result)
            self._update_status(f"设计完成: {result.info}")
            
        except Exception as e:
            messagebox.showerror("设计错误 / Design Error", str(e))
    
    def _update_plots(self):
        """Update all plots based on current result and view mode."""
        if self.current_result is None:
            return
        
        r = self.current_result
        mode = self.view_mode.get()
        
        if mode == 'standard':
            # Standard view: Magnitude, Phase, Group Delay, Pole-Zero
            DSPPlotter.plot_magnitude_response(self.ax_mag, r.freq_hz, r.magnitude_db)
            DSPPlotter.plot_phase_response(self.ax_phase, r.freq_hz, r.phase_deg)
            DSPPlotter.plot_group_delay(self.ax_gd, r.gd_freq_hz, r.group_delay)
            DSPPlotter.plot_pole_zero(self.ax_pz, r.zeros, r.poles)
            
        elif mode == 'time':
            # Time view: Magnitude, Impulse, Step, Pole-Zero
            DSPPlotter.plot_magnitude_response(self.ax_mag, r.freq_hz, r.magnitude_db)
            DSPPlotter.plot_impulse_response(self.ax_phase, r.impulse_t, r.impulse_response)
            DSPPlotter.plot_step_response(self.ax_gd, r.step_t, r.step_response)
            DSPPlotter.plot_pole_zero(self.ax_pz, r.zeros, r.poles)
            
        elif mode == 'compare':
            # Comparison view
            self.ax_mag.clear()
            curves = [r.magnitude_db]
            labels = ['Current']
            for i, cr in enumerate(self.comparison_results):
                curves.append(cr.magnitude_db)
                labels.append(f"#{i+1}: {cr.info[:30]}")
            
            DSPPlotter.plot_comparison(self.ax_mag, r.freq_hz, curves, labels,
                                       title="幅度对比 / Magnitude Comparison")
            
            # Phase comparison
            self.ax_phase.clear()
            self.ax_phase.plot(r.freq_hz, r.phase_deg, linewidth=1.2, label='Current')
            for i, cr in enumerate(self.comparison_results):
                self.ax_phase.plot(cr.freq_hz, cr.phase_deg, linewidth=1.0,
                                   label=f"#{i+1}")
            self.ax_phase.set_title("相位对比 / Phase Comparison", fontweight='bold')
            self.ax_phase.set_xlabel('频率 (Hz)')
            self.ax_phase.set_ylabel('相位 (°)')
            self.ax_phase.legend(fontsize=7)
            self.ax_phase.grid(True, alpha=0.3)
            
            DSPPlotter.plot_group_delay(self.ax_gd, r.gd_freq_hz, r.group_delay)
            DSPPlotter.plot_pole_zero(self.ax_pz, r.zeros, r.poles)
        
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()
    
    def _update_info(self, result: FilterResult):
        """Update info text."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        lines = []
        lines.append(f"📊 {result.info}")
        lines.append(f"阶数 Order: {result.order}")
        lines.append(f"稳定性 Stable: {'✅ Yes' if result.is_stable else '❌ No'}")
        
        if result.b is not None:
            lines.append(f"系数 b: {len(result.b)} taps")
        if result.a is not None and len(result.a) > 1:
            lines.append(f"系数 a: {len(result.a)} taps")
        if result.sos is not None:
            lines.append(f"SOS: {result.sos.shape[0]} sections")
        
        # Passband/stopband performance
        if result.freq_hz is not None and result.magnitude_db is not None:
            max_mag = np.max(result.magnitude_db)
            min_mag = np.min(result.magnitude_db)
            lines.append(f"最大增益 Max: {max_mag:.2f} dB")
            lines.append(f"最小增益 Min: {min_mag:.2f} dB")
        
        if result.poles is not None and len(result.poles) > 0:
            max_pole = np.max(np.abs(result.poles))
            lines.append(f"最大极点幅度: {max_pole:.6f}")
        
        self.info_text.insert(tk.END, '\n'.join(lines))
        self.info_text.config(state=tk.DISABLED)
    
    def _show_code_gen(self):
        """Show code generation dialog."""
        if self.current_result is None:
            messagebox.showinfo("提示", "请先设计滤波器 / Design a filter first")
            return
        
        # Ask for language
        dialog = tk.Toplevel(self)
        dialog.title("选择语言 / Select Language")
        dialog.geometry("300x200")
        dialog.transient(self)
        
        ttk.Label(dialog, text="选择目标语言:", font=('', 11, 'bold')).pack(pady=10)
        
        lang_var = tk.StringVar(value='C')
        for lang in ['C', 'Python', 'MATLAB']:
            ttk.Radiobutton(dialog, text=lang, variable=lang_var,
                           value=lang).pack(anchor=tk.W, padx=30)
        
        def do_gen():
            lang = lang_var.get()
            r = self.current_result
            code = self.code_gen.generate(
                r.b, r.a, r.spec.fs, lang,
                filter_name='designed_filter',
                filter_info=r.info,
                sos=r.sos)
            dialog.destroy()
            CodeGenDialog(self, code, lang)
        
        ttk.Button(dialog, text="生成 / Generate", command=do_gen).pack(pady=15)
        dialog.grab_set()
    
    def _show_fixed_point(self):
        """Show fixed-point analysis dialog."""
        if self.current_result is None:
            messagebox.showinfo("提示", "请先设计滤波器 / Design a filter first")
            return
        
        r = self.current_result
        
        def analyze_cb(b, a, fs, word_length, method):
            fpa = FixedPointAnalyzer()
            result = fpa.analyze(b, a, fs, word_length, method=method)
            
            lines = []
            lines.append(f"═══ 定点量化分析 / Fixed-Point Analysis ═══\n")
            lines.append(f"Q 格式: {result.q_format}")
            lines.append(f"字长: {result.word_length} bits")
            lines.append(f"整数位: {result.int_bits}, 小数位: {result.frac_bits}")
            lines.append(f"")
            lines.append(f"系数量化误差:")
            lines.append(f"  最大误差: {result.max_coeff_error:.2e}")
            lines.append(f"  平均误差: {result.mean_coeff_error:.2e}")
            lines.append(f"  量化SNR: {result.quant_snr_db:.1f} dB")
            lines.append(f"")
            lines.append(f"量化后 b 系数:")
            for i, (orig, quant) in enumerate(zip(result.b_float, result.b_quant)):
                err = abs(orig - quant)
                lines.append(f"  b[{i}] = {quant:.10f}  (err={err:.2e})")
            
            if len(result.a_float) > 1:
                lines.append(f"\n量化后 a 系数:")
                for i, (orig, quant) in enumerate(zip(result.a_float, result.a_quant)):
                    err = abs(orig - quant)
                    lines.append(f"  a[{i}] = {quant:.10f}  (err={err:.2e})")
            
            # SOS analysis
            sos_result = fpa.decompose_sos(b, a, word_length)
            if sos_result.sos_float is not None:
                lines.append(f"\n═══ SOS 分解 ═══")
                lines.append(f"{sos_result.info}")
                for i in range(sos_result.sos_float.shape[0]):
                    lines.append(f"\nSection {i+1}:")
                    lines.append(f"  b = {sos_result.sos_float[i,:3]}")
                    lines.append(f"  a = {sos_result.sos_float[i,3:]}")
            
            # Hex representation
            hex_b = fpa.format_coefficients_hex(result.b_quant, word_length, result.frac_bits)
            lines.append(f"\n═══ 十六进制 / Hex ═══")
            for i, h in enumerate(hex_b):
                lines.append(f"  b[{i}] = {h}")
            
            return '\n'.join(lines)
        
        FixedPointDialog(self, r.b, r.a, r.spec.fs, analyze_cb)
    
    def _add_to_comparison(self):
        """Add current result to comparison list."""
        if self.current_result is None:
            return
        self.comparison_results.append(self.current_result)
        self._update_status(
            f"已添加到对比 (共 {len(self.comparison_results)} 个)")
    
    def _clear_comparison(self):
        """Clear comparison list."""
        self.comparison_results.clear()
        self._update_status("对比已清除")
    
    def _update_status(self, msg):
        """Update status bar."""
        if self.status_callback:
            self.status_callback(msg)
    
    def get_current_result(self):
        """Get current filter result for export."""
        return self.current_result
