"""
Code Generator — 代码生成器

Generate filter implementation code in C/C++, Python, MATLAB.
"""

import numpy as np
from typing import Optional
from datetime import datetime


class CodeGenerator:
    """
    Filter Code Generator — 滤波器代码生成器
    
    Generates ready-to-use filter implementation code.
    """
    
    LANGUAGES = ['C', 'Python', 'MATLAB', 'Verilog', 'VHDL']
    
    def __init__(self):
        pass
    
    def generate(self, b: np.ndarray, a: np.ndarray,
                 fs: float = 1.0,
                 language: str = 'C',
                 filter_name: str = 'my_filter',
                 filter_info: str = '',
                 precision: str = 'double',
                 sos: np.ndarray = None,
                 word_length: int = 16,
                 frac_bits: int = 14) -> str:
        """
        Generate filter code.
        
        Args:
            b: Numerator coefficients
            a: Denominator coefficients
            fs: Sampling rate
            language: Target language ('C', 'Python', 'MATLAB', 'Verilog', 'VHDL')
            filter_name: Name for the filter
            filter_info: Description string
            precision: 'double', 'float', or 'fixed'
            sos: Optional SOS coefficients
            word_length: Bit width for HDL (Verilog/VHDL)
            frac_bits: Fractional bits for fixed-point HDL
        Returns:
            Generated code as string
        """
        if language == 'C':
            return self._generate_c(b, a, fs, filter_name, filter_info, 
                                    precision, sos)
        elif language == 'Python':
            return self._generate_python(b, a, fs, filter_name, filter_info, sos)
        elif language == 'MATLAB':
            return self._generate_matlab(b, a, fs, filter_name, filter_info, sos)
        elif language == 'Verilog':
            return self._generate_verilog(b, a, fs, filter_name, filter_info,
                                           word_length, frac_bits)
        elif language == 'VHDL':
            return self._generate_vhdl(b, a, fs, filter_name, filter_info,
                                        word_length, frac_bits)
        else:
            return f"// Unsupported language: {language}"

    
    def _generate_c(self, b, a, fs, name, info, precision, sos):
        """Generate C/C++ implementation."""
        is_fir = (len(a) == 1 and a[0] == 1.0) or np.allclose(a, [1.0])
        
        dtype = 'double' if precision == 'double' else 'float'
        suffix = '' if precision == 'double' else 'f'
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        code = []
        code.append(f"/*")
        code.append(f" * Digital Filter: {name}")
        code.append(f" * {info}")
        code.append(f" * Sampling Rate: {fs} Hz")
        code.append(f" * Generated: {timestamp}")
        code.append(f" * Tool: DSP Platform - 数字滤波器设计平台")
        code.append(f" */")
        code.append(f"")
        code.append(f"#include <string.h>")
        code.append(f"")
        
        # Coefficients
        nb = len(b)
        code.append(f"#define {name.upper()}_NUM_TAPS {nb}")
        
        b_str = ', '.join([f"{v:.15e}{suffix}" for v in b])
        code.append(f"static const {dtype} {name}_b[{nb}] = {{{b_str}}};")
        
        if not is_fir:
            na = len(a)
            code.append(f"#define {name.upper()}_DEN_LEN {na}")
            a_str = ', '.join([f"{v:.15e}{suffix}" for v in a])
            code.append(f"static const {dtype} {name}_a[{na}] = {{{a_str}}};")
        
        code.append(f"")
        
        # State structure
        code.append(f"typedef struct {{")
        if is_fir:
            code.append(f"    {dtype} delay[{name.upper()}_NUM_TAPS];")
            code.append(f"    int index;")
        else:
            code.append(f"    {dtype} x_delay[{name.upper()}_NUM_TAPS];")
            code.append(f"    {dtype} y_delay[{name.upper()}_DEN_LEN];")
            code.append(f"    int index;")
        code.append(f"}} {name}_state_t;")
        code.append(f"")
        
        # Init function
        code.append(f"void {name}_init({name}_state_t *state) {{")
        if is_fir:
            code.append(f"    memset(state->delay, 0, sizeof(state->delay));")
        else:
            code.append(f"    memset(state->x_delay, 0, sizeof(state->x_delay));")
            code.append(f"    memset(state->y_delay, 0, sizeof(state->y_delay));")
        code.append(f"    state->index = 0;")
        code.append(f"}}")
        code.append(f"")
        
        # Filter function
        if is_fir:
            code.append(f"{dtype} {name}_process({name}_state_t *state, {dtype} input) {{")
            code.append(f"    {dtype} output = 0.0{suffix};")
            code.append(f"    int i, j;")
            code.append(f"")
            code.append(f"    state->delay[state->index] = input;")
            code.append(f"    j = state->index;")
            code.append(f"")
            code.append(f"    for (i = 0; i < {name.upper()}_NUM_TAPS; i++) {{")
            code.append(f"        output += {name}_b[i] * state->delay[j];")
            code.append(f"        if (--j < 0) j = {name.upper()}_NUM_TAPS - 1;")
            code.append(f"    }}")
            code.append(f"")
            code.append(f"    if (++state->index >= {name.upper()}_NUM_TAPS)")
            code.append(f"        state->index = 0;")
            code.append(f"")
            code.append(f"    return output;")
            code.append(f"}}")
        else:
            code.append(f"{dtype} {name}_process({name}_state_t *state, {dtype} input) {{")
            code.append(f"    {dtype} output = 0.0{suffix};")
            code.append(f"    int i;")
            code.append(f"")
            code.append(f"    /* Shift delay lines */")
            code.append(f"    for (i = {name.upper()}_NUM_TAPS - 1; i > 0; i--)")
            code.append(f"        state->x_delay[i] = state->x_delay[i - 1];")
            code.append(f"    state->x_delay[0] = input;")
            code.append(f"")
            code.append(f"    /* Compute output */")
            code.append(f"    for (i = 0; i < {name.upper()}_NUM_TAPS; i++)")
            code.append(f"        output += {name}_b[i] * state->x_delay[i];")
            code.append(f"")
            code.append(f"    for (i = 1; i < {name.upper()}_DEN_LEN; i++)")
            code.append(f"        output -= {name}_a[i] * state->y_delay[i - 1];")
            code.append(f"")
            code.append(f"    /* Update y delay */")
            code.append(f"    for (i = {name.upper()}_DEN_LEN - 2; i > 0; i--)")
            code.append(f"        state->y_delay[i] = state->y_delay[i - 1];")
            code.append(f"    state->y_delay[0] = output;")
            code.append(f"")
            code.append(f"    return output;")
            code.append(f"}}")
        
        code.append(f"")
        
        # Block processing function
        code.append(f"void {name}_process_block({name}_state_t *state,")
        code.append(f"                         const {dtype} *input,")
        code.append(f"                         {dtype} *output, int length) {{")
        code.append(f"    int n;")
        code.append(f"    for (n = 0; n < length; n++)")
        code.append(f"        output[n] = {name}_process(state, input[n]);")
        code.append(f"}}")
        
        return '\n'.join(code)
    
    def _generate_python(self, b, a, fs, name, info, sos):
        """Generate Python implementation."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_fir = (len(a) == 1 and a[0] == 1.0) or np.allclose(a, [1.0])
        
        code = []
        code.append(f'"""')
        code.append(f'Digital Filter: {name}')
        code.append(f'{info}')
        code.append(f'Sampling Rate: {fs} Hz')
        code.append(f'Generated: {timestamp}')
        code.append(f'Tool: DSP Platform - 数字滤波器设计平台')
        code.append(f'"""')
        code.append(f'')
        code.append(f'import numpy as np')
        code.append(f'from scipy import signal')
        code.append(f'')
        code.append(f'# Sampling rate')
        code.append(f'FS = {fs}')
        code.append(f'')
        
        # Coefficients
        code.append(f'# Filter coefficients (numerator)')
        b_str = ', '.join([f'{v:.15e}' for v in b])
        code.append(f'{name}_b = np.array([{b_str}])')
        code.append(f'')
        
        if not is_fir:
            a_str = ', '.join([f'{v:.15e}' for v in a])
            code.append(f'# Filter coefficients (denominator)')
            code.append(f'{name}_a = np.array([{a_str}])')
            code.append(f'')
        
        # SOS if available
        if sos is not None:
            code.append(f'# Second-order sections (recommended for IIR)')
            code.append(f'{name}_sos = np.array([')
            for i in range(sos.shape[0]):
                row = ', '.join([f'{v:.15e}' for v in sos[i]])
                code.append(f'    [{row}],')
            code.append(f'])')
            code.append(f'')
        
        # Filter function
        code.append(f'')
        code.append(f'def apply_{name}(data):')
        code.append(f'    """')
        code.append(f'    Apply {name} filter to input data.')
        code.append(f'    ')
        code.append(f'    Args:')
        code.append(f'        data: Input signal (numpy array)')
        code.append(f'    Returns:')
        code.append(f'        Filtered signal')
        code.append(f'    """')
        
        if sos is not None:
            code.append(f'    return signal.sosfilt({name}_sos, data)')
        elif is_fir:
            code.append(f'    return signal.lfilter({name}_b, [1.0], data)')
        else:
            code.append(f'    return signal.lfilter({name}_b, {name}_a, data)')
        
        code.append(f'')
        code.append(f'')
        code.append(f'def apply_{name}_filtfilt(data):')
        code.append(f'    """Apply zero-phase (forward-backward) filtering."""')
        if sos is not None:
            code.append(f'    return signal.sosfiltfilt({name}_sos, data)')
        elif is_fir:
            code.append(f'    return signal.filtfilt({name}_b, [1.0], data)')
        else:
            code.append(f'    return signal.filtfilt({name}_b, {name}_a, data)')
        
        code.append(f'')
        code.append(f'')
        code.append(f'# === Example usage ===')
        code.append(f'if __name__ == "__main__":')
        code.append(f'    import matplotlib.pyplot as plt')
        code.append(f'    ')
        code.append(f'    # Generate test signal')
        code.append(f'    t = np.arange(0, 0.1, 1.0/FS)')
        code.append(f'    x = np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*1000*t)')
        code.append(f'    ')
        code.append(f'    # Apply filter')
        code.append(f'    y = apply_{name}(x)')
        code.append(f'    ')
        code.append(f'    # Plot')
        code.append(f'    fig, axes = plt.subplots(2, 1, figsize=(10, 6))')
        code.append(f'    axes[0].plot(t, x, label="Input")')
        code.append(f'    axes[0].set_title("Input Signal")')
        code.append(f'    axes[1].plot(t, y, label="Filtered", color="orange")')
        code.append(f'    axes[1].set_title("Filtered Signal")')
        code.append(f'    plt.tight_layout()')
        code.append(f'    plt.show()')
        
        return '\n'.join(code)
    
    def _generate_matlab(self, b, a, fs, name, info, sos):
        """Generate MATLAB implementation."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_fir = (len(a) == 1 and a[0] == 1.0) or np.allclose(a, [1.0])
        
        code = []
        code.append(f'%% Digital Filter: {name}')
        code.append(f'%% {info}')
        code.append(f'%% Sampling Rate: {fs} Hz')
        code.append(f'%% Generated: {timestamp}')
        code.append(f'%% Tool: DSP Platform - 数字滤波器设计平台')
        code.append(f'')
        code.append(f'function [y] = {name}(x)')
        code.append(f'    %% {name.upper()} Apply digital filter')
        code.append(f'    %%   y = {name}(x) filters the input signal x')
        code.append(f'    ')
        code.append(f'    Fs = {fs};  % Sampling rate (Hz)')
        code.append(f'    ')
        
        # Coefficients
        b_str = ' '.join([f'{v:.15e}' for v in b])
        code.append(f'    %% Numerator coefficients')
        code.append(f'    b = [{b_str}];')
        code.append(f'    ')
        
        if not is_fir:
            a_str = ' '.join([f'{v:.15e}' for v in a])
            code.append(f'    %% Denominator coefficients')
            code.append(f'    a = [{a_str}];')
            code.append(f'    ')
        
        if sos is not None:
            code.append(f'    %% Second-order sections')
            code.append(f'    sos = [')
            for i in range(sos.shape[0]):
                row = ' '.join([f'{v:.15e}' for v in sos[i]])
                code.append(f'        {row};')
            code.append(f'    ];')
            code.append(f'    ')
            code.append(f'    %% Filter using SOS (recommended for stability)')
            code.append(f'    y = sosfilt(sos, x);')
        elif is_fir:
            code.append(f'    %% Apply FIR filter')
            code.append(f'    y = filter(b, 1, x);')
        else:
            code.append(f'    %% Apply IIR filter')
            code.append(f'    y = filter(b, a, x);')
        
        code.append(f'end')
        code.append(f'')
        code.append(f'')
        code.append(f'%% === Frequency Response ===')
        code.append(f'%% Uncomment below to plot frequency response')
        code.append(f'%% b = [...]; a = [...];')
        code.append(f'%% [H, w] = freqz(b, a, 2048, {fs});')
        code.append(f'%% figure;')
        code.append(f'%% subplot(2,1,1); plot(w, 20*log10(abs(H)));')
        code.append(f'%% ylabel("Magnitude (dB)"); xlabel("Frequency (Hz)");')
        code.append(f'%% subplot(2,1,2); plot(w, angle(H)*180/pi);')
        code.append(f'%% ylabel("Phase (deg)"); xlabel("Frequency (Hz)");')
        
        return '\n'.join(code)
    
    def _generate_verilog(self, b, a, fs, name, info, word_length, frac_bits):
        """Generate Verilog implementation."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_fir = (len(a) == 1 and a[0] == 1.0) or np.allclose(a, [1.0])
        nb = len(b)
        wl = word_length
        acc_wl = wl * 2 + 4  # Accumulator width
        
        # Quantize coefficients
        scale = 2 ** frac_bits
        b_int = [int(round(v * scale)) for v in b]
        
        code = []
        code.append(f"// Digital Filter: {name}")
        code.append(f"// {info}")
        code.append(f"// Fs = {fs} Hz, Word Length = {wl} bits, Frac = {frac_bits}")
        code.append(f"// Generated: {timestamp}")
        code.append(f"// Tool: DSP Platform")
        code.append(f"")
        code.append(f"`timescale 1ns / 1ps")
        code.append(f"")
        code.append(f"module {name} (")
        code.append(f"    input  wire                    clk,")
        code.append(f"    input  wire                    rst_n,")
        code.append(f"    input  wire                    valid_in,")
        code.append(f"    input  wire signed [{wl-1}:0]  data_in,")
        code.append(f"    output reg  signed [{wl-1}:0]  data_out,")
        code.append(f"    output reg                     valid_out")
        code.append(f");")
        code.append(f"")
        code.append(f"    // Parameters")
        code.append(f"    localparam NUM_TAPS = {nb};")
        code.append(f"    localparam FRAC_BITS = {frac_bits};")
        code.append(f"    localparam ACC_WIDTH = {acc_wl};")
        code.append(f"")
        
        # Coefficient ROM
        code.append(f"    // Coefficients (Q{wl - frac_bits - 1}.{frac_bits} format)")
        for i, c in enumerate(b_int):
            code.append(f"    localparam signed [{wl-1}:0] COEFF_B{i} = {wl}'sd{c};")
        
        if not is_fir:
            a_int = [int(round(v * scale)) for v in a]
            for i, c in enumerate(a_int):
                code.append(f"    localparam signed [{wl-1}:0] COEFF_A{i} = {wl}'sd{c};")
        
        code.append(f"")
        
        # Delay line
        code.append(f"    // Delay line registers")
        code.append(f"    reg signed [{wl-1}:0] x_delay [0:NUM_TAPS-1];")
        if not is_fir:
            na = len(a)
            code.append(f"    reg signed [{wl-1}:0] y_delay [0:{na-2}];")
        code.append(f"    reg signed [{acc_wl-1}:0] acc;")
        code.append(f"    integer i;")
        code.append(f"")
        
        # Main process
        code.append(f"    always @(posedge clk or negedge rst_n) begin")
        code.append(f"        if (!rst_n) begin")
        code.append(f"            for (i = 0; i < NUM_TAPS; i = i + 1)")
        code.append(f"                x_delay[i] <= {wl}'sd0;")
        if not is_fir:
            code.append(f"            for (i = 0; i < {na-1}; i = i + 1)")
            code.append(f"                y_delay[i] <= {wl}'sd0;")
        code.append(f"            data_out <= {wl}'sd0;")
        code.append(f"            valid_out <= 1'b0;")
        code.append(f"        end else if (valid_in) begin")
        code.append(f"            // Shift delay line")
        code.append(f"            for (i = NUM_TAPS-1; i > 0; i = i - 1)")
        code.append(f"                x_delay[i] <= x_delay[i-1];")
        code.append(f"            x_delay[0] <= data_in;")
        code.append(f"")
        code.append(f"            // MAC (Multiply-Accumulate)")
        code.append(f"            acc = {acc_wl}'sd0;")
        
        # Unrolled MAC for b coefficients
        for i in range(nb):
            code.append(f"            acc = acc + COEFF_B{i} * x_delay[{i}];")
        
        if not is_fir:
            for i in range(1, len(a)):
                code.append(f"            acc = acc - COEFF_A{i} * y_delay[{i-1}];")
            code.append(f"")
            code.append(f"            // Update y delay")
            for i in range(len(a)-2, 0, -1):
                code.append(f"            y_delay[{i}] <= y_delay[{i-1}];")
            code.append(f"            y_delay[0] <= acc[FRAC_BITS +: {wl}];")
        
        code.append(f"")
        code.append(f"            // Output with rounding")
        code.append(f"            data_out <= acc[FRAC_BITS +: {wl}];")
        code.append(f"            valid_out <= 1'b1;")
        code.append(f"        end else begin")
        code.append(f"            valid_out <= 1'b0;")
        code.append(f"        end")
        code.append(f"    end")
        code.append(f"")
        code.append(f"endmodule")
        
        return '\n'.join(code)
    
    def _generate_vhdl(self, b, a, fs, name, info, word_length, frac_bits):
        """Generate VHDL implementation."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        is_fir = (len(a) == 1 and a[0] == 1.0) or np.allclose(a, [1.0])
        nb = len(b)
        wl = word_length
        acc_wl = wl * 2 + 4
        
        scale = 2 ** frac_bits
        b_int = [int(round(v * scale)) for v in b]
        
        code = []
        code.append(f"-- Digital Filter: {name}")
        code.append(f"-- {info}")
        code.append(f"-- Fs = {fs} Hz, Word Length = {wl} bits, Frac = {frac_bits}")
        code.append(f"-- Generated: {timestamp}")
        code.append(f"-- Tool: DSP Platform")
        code.append(f"")
        code.append(f"library IEEE;")
        code.append(f"use IEEE.STD_LOGIC_1164.ALL;")
        code.append(f"use IEEE.NUMERIC_STD.ALL;")
        code.append(f"")
        code.append(f"entity {name} is")
        code.append(f"    generic (")
        code.append(f"        DATA_WIDTH : integer := {wl};")
        code.append(f"        FRAC_BITS  : integer := {frac_bits};")
        code.append(f"        ACC_WIDTH  : integer := {acc_wl}")
        code.append(f"    );")
        code.append(f"    port (")
        code.append(f"        clk       : in  std_logic;")
        code.append(f"        rst_n     : in  std_logic;")
        code.append(f"        valid_in  : in  std_logic;")
        code.append(f"        data_in   : in  signed(DATA_WIDTH-1 downto 0);")
        code.append(f"        data_out  : out signed(DATA_WIDTH-1 downto 0);")
        code.append(f"        valid_out : out std_logic")
        code.append(f"    );")
        code.append(f"end entity {name};")
        code.append(f"")
        code.append(f"architecture rtl of {name} is")
        code.append(f"")
        code.append(f"    constant NUM_TAPS : integer := {nb};")
        code.append(f"")
        
        # Coefficient constants
        code.append(f"    -- Coefficients (Q{wl - frac_bits - 1}.{frac_bits} fixed-point)")
        code.append(f"    type coeff_array_t is array (0 to NUM_TAPS-1) of signed(DATA_WIDTH-1 downto 0);")
        b_vals = ', '.join([f'to_signed({c}, DATA_WIDTH)' for c in b_int])
        code.append(f"    constant COEFF_B : coeff_array_t := ({b_vals});")
        code.append(f"")
        
        if not is_fir:
            na = len(a)
            a_int = [int(round(v * scale)) for v in a]
            code.append(f"    type coeff_a_array_t is array (0 to {na-1}) of signed(DATA_WIDTH-1 downto 0);")
            a_vals = ', '.join([f'to_signed({c}, DATA_WIDTH)' for c in a_int])
            code.append(f"    constant COEFF_A : coeff_a_array_t := ({a_vals});")
            code.append(f"")
        
        # Delay line type
        code.append(f"    type delay_line_t is array (0 to NUM_TAPS-1) of signed(DATA_WIDTH-1 downto 0);")
        code.append(f"    signal x_delay : delay_line_t := (others => (others => '0'));")
        
        if not is_fir:
            code.append(f"    type y_delay_line_t is array (0 to {na-2}) of signed(DATA_WIDTH-1 downto 0);")
            code.append(f"    signal y_delay : y_delay_line_t := (others => (others => '0'));")
        
        code.append(f"    signal acc : signed(ACC_WIDTH-1 downto 0);")
        code.append(f"")
        code.append(f"begin")
        code.append(f"")
        code.append(f"    process(clk, rst_n)")
        code.append(f"        variable v_acc : signed(ACC_WIDTH-1 downto 0);")
        code.append(f"    begin")
        code.append(f"        if rst_n = '0' then")
        code.append(f"            x_delay <= (others => (others => '0'));")
        if not is_fir:
            code.append(f"            y_delay <= (others => (others => '0'));")
        code.append(f"            data_out <= (others => '0');")
        code.append(f"            valid_out <= '0';")
        code.append(f"        elsif rising_edge(clk) then")
        code.append(f"            if valid_in = '1' then")
        code.append(f"                -- Shift delay line")
        code.append(f"                for i in NUM_TAPS-1 downto 1 loop")
        code.append(f"                    x_delay(i) <= x_delay(i-1);")
        code.append(f"                end loop;")
        code.append(f"                x_delay(0) <= data_in;")
        code.append(f"")
        code.append(f"                -- MAC")
        code.append(f"                v_acc := (others => '0');")
        code.append(f"                for i in 0 to NUM_TAPS-1 loop")
        code.append(f"                    v_acc := v_acc + resize(COEFF_B(i) * x_delay(i), ACC_WIDTH);")
        code.append(f"                end loop;")
        
        if not is_fir:
            code.append(f"                -- Feedback")
            code.append(f"                for i in 1 to {na-1} loop")
            code.append(f"                    v_acc := v_acc - resize(COEFF_A(i) * y_delay(i-1), ACC_WIDTH);")
            code.append(f"                end loop;")
            code.append(f"                -- Update y delay")
            for i in range(na-2, 0, -1):
                code.append(f"                y_delay({i}) <= y_delay({i-1});")
            code.append(f"                y_delay(0) <= v_acc(FRAC_BITS + DATA_WIDTH - 1 downto FRAC_BITS);")
        
        code.append(f"")
        code.append(f"                -- Output")
        code.append(f"                data_out <= v_acc(FRAC_BITS + DATA_WIDTH - 1 downto FRAC_BITS);")
        code.append(f"                valid_out <= '1';")
        code.append(f"            else")
        code.append(f"                valid_out <= '0';")
        code.append(f"            end if;")
        code.append(f"        end if;")
        code.append(f"    end process;")
        code.append(f"")
        code.append(f"end architecture rtl;")
        
        return '\n'.join(code)
