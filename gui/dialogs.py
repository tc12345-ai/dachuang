"""
Dialogs — 对话框模块

Code generation, fixed-point analysis, export, and about dialogs.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os


class CodeGenDialog(tk.Toplevel):
    """Code generation dialog / 代码生成对话框."""
    
    def __init__(self, parent, code_text: str, language: str = 'C'):
        super().__init__(parent)
        self.title(f"生成代码 / Generated Code ({language})")
        self.geometry("750x600")
        self.transient(parent)
        
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="📋 复制 / Copy",
                   command=self._copy).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 保存 / Save",
                   command=lambda: self._save(language)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="关闭 / Close",
                   command=self.destroy).pack(side=tk.RIGHT, padx=2)
        
        ttk.Label(toolbar, text=f"Language: {language}",
                  font=('Consolas', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        
        # Code display
        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.NONE, font=('Consolas', 10),
            bg='#1e1e1e', fg='#d4d4d4', insertbackground='white')
        self.text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text.insert('1.0', code_text)
        self.text.config(state=tk.NORMAL)
        
        self.code_text = code_text
        self.grab_set()
    
    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.code_text)
        messagebox.showinfo("提示", "代码已复制到剪贴板 / Code copied!")
    
    def _save(self, language):
        ext_map = {'C': '.c', 'Python': '.py', 'MATLAB': '.m'}
        ext = ext_map.get(language, '.txt')
        ft = [(f"{language} Files", f"*{ext}"), ("All Files", "*.*")]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=ft,
            title="保存代码 / Save Code")
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.code_text)
            messagebox.showinfo("提示", f"代码已保存: {filepath}")


class ExportDialog(tk.Toplevel):
    """Export options dialog / 导出选项对话框."""
    
    def __init__(self, parent, callback):
        """
        Args:
            parent: Parent window
            callback: Function(export_type, filepath, options) called on export
        """
        super().__init__(parent)
        self.title("导出 / Export")
        self.geometry("450x400")
        self.transient(parent)
        self.callback = callback
        
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Export type
        ttk.Label(main, text="导出类型 / Export Type:",
                  font=('', 10, 'bold')).pack(anchor=tk.W)
        
        self.export_type = tk.StringVar(value='coefficients_csv')
        types = [
            ('滤波器系数 CSV / Coefficients CSV', 'coefficients_csv'),
            ('滤波器系数 JSON / Coefficients JSON', 'coefficients_json'),
            ('滤波器系数 MAT / Coefficients MAT', 'coefficients_mat'),
            ('频率响应数据 / Frequency Response', 'freq_response'),
            ('频谱数据 / Spectrum Data', 'spectrum_data'),
            ('图表 PNG / Chart PNG', 'chart_png'),
            ('图表 SVG / Chart SVG', 'chart_svg'),
            ('HTML 报告 / HTML Report', 'html_report'),
        ]
        for text, val in types:
            ttk.Radiobutton(main, text=text, variable=self.export_type,
                           value=val).pack(anchor=tk.W, padx=20)
        
        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="导出 / Export",
                   command=self._do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消 / Cancel",
                   command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
        self.grab_set()
    
    def _do_export(self):
        etype = self.export_type.get()
        
        ext_map = {
            'coefficients_csv': ('.csv', [("CSV", "*.csv")]),
            'coefficients_json': ('.json', [("JSON", "*.json")]),
            'coefficients_mat': ('.mat', [("MAT", "*.mat")]),
            'freq_response': ('.csv', [("CSV", "*.csv")]),
            'spectrum_data': ('.csv', [("CSV", "*.csv")]),
            'chart_png': ('.png', [("PNG", "*.png")]),
            'chart_svg': ('.svg', [("SVG", "*.svg")]),
            'html_report': ('.html', [("HTML", "*.html")]),
        }
        
        ext, ft = ext_map.get(etype, ('.txt', [("Text", "*.txt")]))
        ft.append(("All Files", "*.*"))
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=ft,
            title="选择保存位置 / Choose Save Location")
        
        if filepath:
            self.callback(etype, filepath)
            self.destroy()


class FixedPointDialog(tk.Toplevel):
    """Fixed-point analysis dialog / 定点分析对话框."""
    
    def __init__(self, parent, b, a, fs, analyze_callback):
        super().__init__(parent)
        self.title("定点量化分析 / Fixed-Point Analysis")
        self.geometry("700x550")
        self.transient(parent)
        
        self.b = b
        self.a = a
        self.fs = fs
        self.analyze_callback = analyze_callback
        
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Parameters
        param_frame = ttk.LabelFrame(main, text="量化参数 / Quantization Parameters",
                                      padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        row = ttk.Frame(param_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="字长 / Word Length (bits):").pack(side=tk.LEFT)
        self.word_length = tk.IntVar(value=16)
        ttk.Spinbox(row, from_=8, to=64, textvariable=self.word_length,
                    width=8).pack(side=tk.LEFT, padx=10)
        
        row2 = ttk.Frame(param_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="量化方法 / Method:").pack(side=tk.LEFT)
        self.method = tk.StringVar(value='round')
        ttk.Combobox(row2, textvariable=self.method, width=12,
                     values=['round', 'truncate', 'ceil'],
                     state='readonly').pack(side=tk.LEFT, padx=10)
        
        ttk.Button(param_frame, text="▶ 分析 / Analyze",
                   command=self._analyze).pack(pady=5)
        
        # Results
        self.result_text = scrolledtext.ScrolledText(
            main, wrap=tk.WORD, font=('Consolas', 9),
            height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.grab_set()
        self._analyze()
    
    def _analyze(self):
        result = self.analyze_callback(
            self.b, self.a, self.fs,
            self.word_length.get(), self.method.get())
        
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, result)


class AboutDialog(tk.Toplevel):
    """About dialog / 关于对话框."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("关于 / About")
        self.geometry("450x350")
        self.resizable(False, False)
        self.transient(parent)
        
        main = ttk.Frame(self, padding=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main, text="📊 DSP Platform",
                  font=('', 18, 'bold')).pack(pady=5)
        ttk.Label(main, text="数字滤波器设计与信号频谱分析平台",
                  font=('', 12)).pack()
        ttk.Label(main, text="Digital Filter Design & Signal Spectrum Analysis",
                  font=('', 10)).pack()
        
        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        info = (
            "Version: 1.0\n\n"
            "功能 / Features:\n"
            "  • FIR/IIR 滤波器设计 (窗函数/等波纹/双线性变换)\n"
            "  • FFT/STFT/PSD 频谱分析\n"
            "  • THD/SNR/SFDR 测量\n"
            "  • 定点量化分析\n"
            "  • C/Python/MATLAB 代码生成\n"
            "  • 多格式数据导入导出\n\n"
            "依赖 / Dependencies:\n"
            "  NumPy, SciPy, Matplotlib"
        )
        
        ttk.Label(main, text=info, justify=tk.LEFT,
                  font=('', 9)).pack(anchor=tk.W)
        
        ttk.Button(main, text="确定 / OK",
                   command=self.destroy).pack(pady=10)
        
        self.grab_set()
