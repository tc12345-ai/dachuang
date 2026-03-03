#!/usr/bin/env python3
"""
DSP Platform — 数字滤波器设计与信号频谱分析平台
Digital Filter Design & Signal Spectrum Analysis Platform

Main entry point for the application.

Features:
- FIR/IIR filter design (Window, Frequency Sampling, Equiripple,
  Butterworth, Chebyshev I/II, Elliptic, Bessel)
- FFT/STFT/PSD spectrum analysis
- THD/SNR/SFDR/SINAD/ENOB measurements
- Multi-view synchronized visualization
- C/Python/MATLAB code generation
- Fixed-point quantization analysis
- Data import/export (WAV, CSV, MAT, BIN)
- HTML report generation
- Project save/load

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import run_app


def main():
    """Main entry point."""
    print("=" * 60)
    print("  DSP Platform")
    print("  数字滤波器设计与信号频谱分析平台")
    print("  Digital Filter Design & Spectrum Analysis")
    print("=" * 60)
    print("\n启动图形界面...\n")
    
    run_app()


if __name__ == '__main__':
    main()
