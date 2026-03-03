"""
Scripting Panel — 脚本编辑与执行面板

Python/Script editor with syntax highlighting, example scripts,
execution output, and variable inspector.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.script_engine import ScriptEngine


class ScriptPanel(ttk.Frame):
    """
    Script Editor Panel — 脚本编辑面板
    
    Provides code editing, execution, and output display
    for DSP automation scripts.
    """
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.engine = ScriptEngine()
        self._create_ui()
    
    def _create_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=5, pady=3)
        
        ttk.Button(toolbar, text="Run / 运行",
                   command=self._run_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Stop / 停止",
                   command=self._stop_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear / 清空",
                   command=self._clear_output).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(toolbar, text="Examples:").pack(side=tk.LEFT, padx=5)
        self.example_var = tk.StringVar(value='-- Select --')
        examples = list(ScriptEngine.get_example_scripts().keys())
        ttk.Combobox(toolbar, textvariable=self.example_var,
                     values=['-- Select --'] + examples,
                     state='readonly', width=22).pack(side=tk.LEFT, padx=2)
        self.example_var.trace_add('write', self._load_example)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Button(toolbar, text="Open / 打开",
                   command=self._open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save / 保存",
                   command=self._save_file).pack(side=tk.LEFT, padx=2)
        
        # Main split: editor above, output below
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Editor
        editor_frame = ttk.LabelFrame(paned, text="Script Editor / 脚本编辑器",
                                       padding=3)
        paned.add(editor_frame, weight=3)
        
        # Line numbers + code
        edit_container = ttk.Frame(editor_frame)
        edit_container.pack(fill=tk.BOTH, expand=True)
        
        self.line_nums = tk.Text(edit_container, width=4, padx=3,
                                  bg='#2c3e50', fg='#95a5a6',
                                  font=('Consolas', 10), state=tk.DISABLED,
                                  takefocus=0)
        self.line_nums.pack(side=tk.LEFT, fill=tk.Y)
        
        self.editor = scrolledtext.ScrolledText(
            edit_container, wrap=tk.NONE,
            font=('Consolas', 10),
            bg='#1e1e1e', fg='#d4d4d4',
            insertbackground='white',
            selectbackground='#264f78',
            undo=True)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.editor.bind('<KeyRelease>', self._update_line_numbers)
        self.editor.bind('<Return>', lambda e: self.after(10, self._update_line_numbers))
        
        # Default script
        self.editor.insert('1.0', '''# DSP Platform Script
# Available: designer, analyzer, measurer, codegen, np
# FilterSpec, generate_test_signal, format_freq, format_db

# Example: Quick filter design
spec = FilterSpec(
    filter_class='FIR', design_method='window',
    filter_type='lowpass', order=32,
    fs=8000, fp1=1000, window_type='hamming')

result = designer.design(spec)
print(f"Filter: {result.info}")
print(f"Order: {result.order}")
print(f"Stable: {result.is_stable}")
print(f"Coefficients b: {len(result.b)} taps")
''')
        self._update_line_numbers()
        
        # Output area
        output_frame = ttk.LabelFrame(paned, text="Output / 输出", padding=3)
        paned.add(output_frame, weight=2)
        
        self.output = scrolledtext.ScrolledText(
            output_frame, wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#0d1117', fg='#c9d1d9',
            state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True)
        
        # Configure output tags
        self.output.tag_configure('error', foreground='#f85149')
        self.output.tag_configure('success', foreground='#3fb950')
        self.output.tag_configure('info', foreground='#58a6ff')
    
    def _update_line_numbers(self, event=None):
        """Update line number display."""
        self.line_nums.config(state=tk.NORMAL)
        self.line_nums.delete('1.0', tk.END)
        
        content = self.editor.get('1.0', tk.END)
        n_lines = content.count('\n')
        line_str = '\n'.join(str(i) for i in range(1, n_lines + 1))
        self.line_nums.insert('1.0', line_str)
        self.line_nums.config(state=tk.DISABLED)
    
    def _run_script(self):
        """Execute the script."""
        code = self.editor.get('1.0', tk.END).strip()
        if not code:
            return
        
        self._append_output(">>> Running script...\n", 'info')
        
        def on_complete(result):
            if result['output']:
                self._append_output(result['output'])
            
            if result['error']:
                self._append_output(f"\nError:\n{result['error']}\n", 'error')
            else:
                self._append_output(
                    f"\n--- Completed in {result['elapsed_ms']:.1f}ms ---\n",
                    'success')
            
            if result['variables']:
                self._append_output("Variables:\n", 'info')
                for k, v in result['variables'].items():
                    self._append_output(f"  {k} = {v}\n")
            
            if self.status_callback:
                status = "Script error" if result['error'] else "Script completed"
                self.status_callback(f"{status} ({result['elapsed_ms']:.0f}ms)")
        
        self.engine.execute_async(code, callback=lambda r: self.after(0, lambda: on_complete(r)))
    
    def _stop_script(self):
        """Stop running script."""
        self.engine.stop()
        self._append_output("\n--- Script stopped ---\n", 'error')
    
    def _append_output(self, text, tag=None):
        """Append text to output."""
        self.output.config(state=tk.NORMAL)
        if tag:
            self.output.insert(tk.END, text, tag)
        else:
            self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)
    
    def _clear_output(self):
        """Clear output area."""
        self.output.config(state=tk.NORMAL)
        self.output.delete('1.0', tk.END)
        self.output.config(state=tk.DISABLED)
    
    def _load_example(self, *args):
        """Load example script."""
        name = self.example_var.get()
        if name == '-- Select --':
            return
        
        examples = ScriptEngine.get_example_scripts()
        if name in examples:
            self.editor.delete('1.0', tk.END)
            self.editor.insert('1.0', examples[name])
            self._update_line_numbers()
    
    def _open_file(self):
        """Open script file."""
        fp = filedialog.askopenfilename(
            filetypes=[("Python", "*.py"), ("All", "*.*")],
            title="Open Script")
        if fp:
            with open(fp, 'r', encoding='utf-8') as f:
                self.editor.delete('1.0', tk.END)
                self.editor.insert('1.0', f.read())
            self._update_line_numbers()
    
    def _save_file(self):
        """Save script file."""
        fp = filedialog.asksaveasfilename(
            defaultextension='.py',
            filetypes=[("Python", "*.py"), ("All", "*.*")],
            title="Save Script")
        if fp:
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(self.editor.get('1.0', tk.END))
