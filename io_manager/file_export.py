"""
File Export — 数据导出模块

Export filter coefficients, frequency response data, charts, and HTML reports.
"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, Dict, List


class FileExporter:
    """
    File Exporter — 文件导出器
    
    Exports coefficients, data, charts, and reports.
    """
    
    def __init__(self):
        pass
    
    def export_coefficients(self, filepath: str,
                            b: np.ndarray, a: np.ndarray,
                            fs: float = 1.0,
                            fmt: str = 'csv',
                            info: str = '',
                            sos: np.ndarray = None):
        """
        Export filter coefficients.
        
        Args:
            filepath: Output file path
            b: Numerator coefficients
            a: Denominator coefficients
            fs: Sampling rate
            fmt: 'csv', 'json', 'mat', 'txt'
            info: Filter description
            sos: Optional SOS coefficients
        """
        if fmt == 'csv':
            self._export_coeff_csv(filepath, b, a, fs, info, sos)
        elif fmt == 'json':
            self._export_coeff_json(filepath, b, a, fs, info, sos)
        elif fmt == 'mat':
            self._export_coeff_mat(filepath, b, a, fs, info, sos)
        elif fmt == 'txt':
            self._export_coeff_txt(filepath, b, a, fs, info, sos)
    
    def _export_coeff_csv(self, filepath, b, a, fs, info, sos):
        """Export coefficients as CSV."""
        with open(filepath, 'w') as f:
            f.write(f"# Filter Coefficients\n")
            f.write(f"# {info}\n")
            f.write(f"# Sampling Rate: {fs} Hz\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"\n")
            f.write(f"# Numerator (b)\n")
            f.write(f"index,coefficient\n")
            for i, val in enumerate(b):
                f.write(f"{i},{val:.15e}\n")
            f.write(f"\n")
            if not (len(a) == 1 and a[0] == 1.0):
                f.write(f"# Denominator (a)\n")
                f.write(f"index,coefficient\n")
                for i, val in enumerate(a):
                    f.write(f"{i},{val:.15e}\n")
    
    def _export_coeff_json(self, filepath, b, a, fs, info, sos):
        """Export coefficients as JSON."""
        data = {
            'info': info,
            'fs': fs,
            'generated': datetime.now().isoformat(),
            'b': b.tolist(),
            'a': a.tolist(),
            'order': max(len(b), len(a)) - 1,
        }
        if sos is not None:
            data['sos'] = sos.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_coeff_mat(self, filepath, b, a, fs, info, sos):
        """Export coefficients as MATLAB .mat file."""
        from scipy.io import savemat
        mdict = {'b': b, 'a': a, 'fs': fs}
        if sos is not None:
            mdict['sos'] = sos
        savemat(filepath, mdict)
    
    def _export_coeff_txt(self, filepath, b, a, fs, info, sos):
        """Export coefficients as plain text."""
        with open(filepath, 'w') as f:
            f.write(f"Filter: {info}\n")
            f.write(f"Sampling Rate: {fs} Hz\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Numerator coefficients (b):\n")
            for i, val in enumerate(b):
                f.write(f"  b[{i}] = {val:.15e}\n")
            f.write(f"\nDenominator coefficients (a):\n")
            for i, val in enumerate(a):
                f.write(f"  a[{i}] = {val:.15e}\n")
    
    def export_frequency_response(self, filepath: str,
                                   freq_hz: np.ndarray,
                                   magnitude_db: np.ndarray,
                                   phase_deg: np.ndarray = None,
                                   group_delay: np.ndarray = None):
        """Export frequency response data as CSV."""
        with open(filepath, 'w') as f:
            f.write("# Frequency Response Data\n")
            header = "Frequency_Hz,Magnitude_dB"
            if phase_deg is not None:
                header += ",Phase_deg"
            if group_delay is not None:
                header += ",GroupDelay_samples"
            f.write(header + "\n")
            
            n = len(freq_hz)
            for i in range(n):
                line = f"{freq_hz[i]:.6f},{magnitude_db[i]:.6f}"
                if phase_deg is not None and i < len(phase_deg):
                    line += f",{phase_deg[i]:.6f}"
                if group_delay is not None and i < len(group_delay):
                    line += f",{group_delay[i]:.6f}"
                f.write(line + "\n")
    
    def export_spectrum_data(self, filepath: str,
                             freq_hz: np.ndarray,
                             magnitude_db: np.ndarray,
                             psd_db: np.ndarray = None):
        """Export spectrum analysis data as CSV."""
        with open(filepath, 'w') as f:
            f.write("# Spectrum Data\n")
            header = "Frequency_Hz,Magnitude_dB"
            if psd_db is not None:
                header += ",PSD_dB_Hz"
            f.write(header + "\n")
            
            for i in range(len(freq_hz)):
                line = f"{freq_hz[i]:.6f},{magnitude_db[i]:.6f}"
                if psd_db is not None and i < len(psd_db):
                    line += f",{psd_db[i]:.6f}"
                f.write(line + "\n")
    
    def generate_html_report(self, filepath: str,
                              title: str = "DSP Analysis Report",
                              sections: List[Dict] = None,
                              chart_paths: List[str] = None):
        """
        Generate HTML report.
        
        Args:
            filepath: Output HTML path
            title: Report title
            sections: List of dicts with 'title', 'content' (text or table)
            chart_paths: List of chart image paths to embed
        """
        html = []
        html.append('<!DOCTYPE html>')
        html.append('<html lang="zh-CN">')
        html.append('<head>')
        html.append(f'<meta charset="UTF-8">')
        html.append(f'<title>{title}</title>')
        html.append('<style>')
        html.append('body { font-family: "Segoe UI", Arial, sans-serif; '
                     'max-width: 1000px; margin: 0 auto; padding: 20px; '
                     'background: #f5f5f5; color: #333; }')
        html.append('h1 { color: #1a5276; border-bottom: 3px solid #2980b9; '
                     'padding-bottom: 10px; }')
        html.append('h2 { color: #2980b9; margin-top: 30px; }')
        html.append('.section { background: white; padding: 20px; '
                     'border-radius: 8px; margin: 15px 0; '
                     'box-shadow: 0 2px 6px rgba(0,0,0,0.1); }')
        html.append('table { border-collapse: collapse; width: 100%; '
                     'margin: 10px 0; }')
        html.append('th, td { border: 1px solid #ddd; padding: 8px 12px; '
                     'text-align: left; }')
        html.append('th { background-color: #2980b9; color: white; }')
        html.append('tr:nth-child(even) { background-color: #f8f9fa; }')
        html.append('img { max-width: 100%; height: auto; margin: 10px 0; '
                     'border-radius: 4px; }')
        html.append('.meta { color: #888; font-size: 0.9em; }')
        html.append('.highlight { background: #eaf2f8; padding: 5px 10px; '
                     'border-radius: 4px; font-family: monospace; }')
        html.append('</style>')
        html.append('</head>')
        html.append('<body>')
        html.append(f'<h1>📊 {title}</h1>')
        html.append(f'<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
                     f'| Tool: DSP Platform 数字滤波器设计与信号频谱分析平台</p>')
        
        if sections:
            for sec in sections:
                html.append('<div class="section">')
                html.append(f'<h2>{sec.get("title", "")}</h2>')
                content = sec.get('content', '')
                if isinstance(content, str):
                    html.append(f'<p>{content}</p>')
                elif isinstance(content, list):
                    # Table data: list of dicts
                    if content and isinstance(content[0], dict):
                        keys = list(content[0].keys())
                        html.append('<table>')
                        html.append('<tr>' + ''.join(
                            f'<th>{k}</th>' for k in keys) + '</tr>')
                        for row in content:
                            html.append('<tr>' + ''.join(
                                f'<td>{row.get(k, "")}</td>'
                                for k in keys) + '</tr>')
                        html.append('</table>')
                html.append('</div>')
        
        if chart_paths:
            html.append('<div class="section">')
            html.append('<h2>图表 / Charts</h2>')
            for cp in chart_paths:
                if os.path.exists(cp):
                    import base64
                    with open(cp, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    ext = os.path.splitext(cp)[1].lower()
                    mime = 'image/png' if ext == '.png' else 'image/svg+xml'
                    html.append(f'<img src="data:{mime};base64,{img_data}" '
                                f'alt="{os.path.basename(cp)}">')
                else:
                    html.append(f'<p>Chart: {cp} (file not found)</p>')
            html.append('</div>')
        
        html.append('</body>')
        html.append('</html>')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
    
    def save_figure(self, fig, filepath: str, dpi: int = 150):
        """
        Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            filepath: Output path (.png or .svg)
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
