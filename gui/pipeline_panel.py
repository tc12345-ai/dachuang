"""
Pipeline Editor Panel — 拖拽式流程编排面板

Visual node-based signal processing pipeline designer using Canvas.
Nodes: Source → Process → Analyze → Output
Drag connections, real-time preview.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import uuid
import json


# ─── Node Definitions ───

NODE_CATALOG = {
    'source': {
        'Signal Gen': {'type': 'source', 'params': {'freq': 440, 'fs': 8000, 'duration': 0.1, 'noise': 0.0}},
        'File Input': {'type': 'source', 'params': {'filepath': '', 'fs': 8000}},
        'Microphone': {'type': 'source', 'params': {'fs': 44100, 'chunk_size': 1024}},
    },
    'filter': {
        'FIR Filter': {'type': 'filter', 'params': {'method': 'window', 'order': 32, 'fc': 1000, 'filter_type': 'lowpass'}},
        'IIR Filter': {'type': 'filter', 'params': {'method': 'butter', 'order': 4, 'fc': 1000, 'filter_type': 'lowpass'}},
        'Notch Filter': {'type': 'filter', 'params': {'freq': 50, 'bw': 10}},
        'Custom': {'type': 'filter', 'params': {'b': '1.0', 'a': '1.0'}},
    },
    'analysis': {
        'FFT': {'type': 'analysis', 'params': {'nfft': 2048, 'window': 'Hann'}},
        'PSD': {'type': 'analysis', 'params': {'nfft': 1024, 'method': 'welch'}},
        'STFT': {'type': 'analysis', 'params': {'nperseg': 256, 'overlap': 75}},
        'Measurements': {'type': 'analysis', 'params': {'metrics': 'THD,SNR,SFDR'}},
        'ML Detector': {'type': 'analysis', 'params': {'method': 'energy', 'threshold': 'auto'}},
    },
    'output': {
        'Plot': {'type': 'output', 'params': {'chart_type': 'auto'}},
        'Export CSV': {'type': 'output', 'params': {'filepath': 'output.csv'}},
        'Export WAV': {'type': 'output', 'params': {'filepath': 'output.wav'}},
        'Console': {'type': 'output', 'params': {}},
    }
}

NODE_COLORS = {
    'source': '#3498db',
    'filter': '#e74c3c',
    'analysis': '#27ae60',
    'output': '#f39c12',
}


class PipelineNode:
    """Visual node in the pipeline."""
    
    def __init__(self, canvas, name, category, node_type, params, x=100, y=100):
        self.id = str(uuid.uuid4())[:8]
        self.canvas = canvas
        self.name = name
        self.category = category
        self.node_type = node_type
        self.params = dict(params)
        self.x = x
        self.y = y
        self.w = 150
        self.h = 60
        self.connections_out = []
        self.connections_in = []
        self.canvas_items = []
        self.selected = False
        self._draw()
    
    def _draw(self):
        """Draw node on canvas."""
        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items.clear()
        
        color = NODE_COLORS.get(self.node_type, '#95a5a6')
        outline = '#2c3e50' if not self.selected else '#e74c3c'
        lw = 2 if not self.selected else 3
        
        # Shadow
        shadow = self.canvas.create_rectangle(
            self.x + 3, self.y + 3,
            self.x + self.w + 3, self.y + self.h + 3,
            fill='#bdc3c7', outline='', width=0)
        self.canvas_items.append(shadow)
        
        # Body
        body = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.w, self.y + self.h,
            fill=color, outline=outline, width=lw, tags='node')
        self.canvas_items.append(body)
        
        # Title bar
        bar = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.w, self.y + 22,
            fill=self._darken(color), outline='', tags='node')
        self.canvas_items.append(bar)
        
        # Name text
        name_text = self.canvas.create_text(
            self.x + self.w // 2, self.y + 11,
            text=self.name, fill='white', font=('Segoe UI', 8, 'bold'),
            tags='node')
        self.canvas_items.append(name_text)
        
        # Category text
        cat_text = self.canvas.create_text(
            self.x + self.w // 2, self.y + 40,
            text=self.category, fill='white', font=('Segoe UI', 7),
            tags='node')
        self.canvas_items.append(cat_text)
        
        # Input port (left circle)
        if self.node_type != 'source':
            inp = self.canvas.create_oval(
                self.x - 6, self.y + self.h // 2 - 6,
                self.x + 6, self.y + self.h // 2 + 6,
                fill='white', outline='#2c3e50', width=2, tags='port_in')
            self.canvas_items.append(inp)
        
        # Output port (right circle)
        if self.node_type != 'output':
            outp = self.canvas.create_oval(
                self.x + self.w - 6, self.y + self.h // 2 - 6,
                self.x + self.w + 6, self.y + self.h // 2 + 6,
                fill='white', outline='#2c3e50', width=2, tags='port_out')
            self.canvas_items.append(outp)
        
        # Bind events
        for item in self.canvas_items:
            self.canvas.tag_bind(item, '<ButtonPress-1>', self._on_press)
            self.canvas.tag_bind(item, '<B1-Motion>', self._on_drag)
            self.canvas.tag_bind(item, '<ButtonRelease-1>', self._on_release)
    
    def _darken(self, color, factor=0.7):
        """Darken a hex color."""
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'
    
    def _on_press(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self.selected = True
        self._draw()
        # Notify panel
        if hasattr(self.canvas, 'on_node_selected'):
            self.canvas.on_node_selected(self)
    
    def _on_drag(self, event):
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.x += dx
        self.y += dy
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._draw()
        # Redraw connections
        if hasattr(self.canvas, 'redraw_connections'):
            self.canvas.redraw_connections()
    
    def _on_release(self, event):
        pass
    
    def get_output_port(self):
        return (self.x + self.w, self.y + self.h // 2)
    
    def get_input_port(self):
        return (self.x, self.y + self.h // 2)
    
    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'category': self.category,
            'node_type': self.node_type, 'params': self.params,
            'x': self.x, 'y': self.y,
            'connections_out': [c.id for c in self.connections_out],
        }


class PipelinePanel(ttk.Frame):
    """
    Pipeline Editor Panel — 流程编排面板
    
    Visual drag-and-drop pipeline editor for building
    signal processing chains.
    """
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        self.nodes = {}
        self.connections = []  # (from_node_id, to_node_id, line_item)
        self.selected_node = None
        self._create_ui()
    
    def _create_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left: Node catalog + properties
        left = ttk.Frame(paned, width=280)
        paned.add(left, weight=0)
        
        # Node catalog
        cat_frame = ttk.LabelFrame(left, text="Node Catalog / 节点目录", padding=5)
        cat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for category, nodes in NODE_CATALOG.items():
            color = NODE_COLORS.get(category, '#95a5a6')
            cat_label = ttk.Label(cat_frame, text=f"  {category.upper()}",
                                   font=('Segoe UI', 9, 'bold'))
            cat_label.pack(anchor=tk.W, pady=(5, 0))
            
            for node_name in nodes:
                btn = tk.Button(cat_frame, text=f"  + {node_name}",
                               bg=color, fg='white', relief=tk.FLAT,
                               anchor=tk.W, font=('Segoe UI', 8),
                               command=lambda n=node_name, c=category: self._add_node(c, n))
                btn.pack(fill=tk.X, padx=10, pady=1)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Actions
        act_frame = ttk.LabelFrame(left, text="Actions / 操作", padding=5)
        act_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(act_frame, text="Connect Selected / 连接选中",
                   command=self._connect_mode).pack(fill=tk.X, pady=2)
        ttk.Button(act_frame, text="Delete Selected / 删除选中",
                   command=self._delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(act_frame, text="Run Pipeline / 运行流程",
                   command=self._run_pipeline).pack(fill=tk.X, pady=2)
        ttk.Button(act_frame, text="Clear All / 清空",
                   command=self._clear_all).pack(fill=tk.X, pady=2)
        
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Properties
        self.props_frame = ttk.LabelFrame(left, text="Properties / 属性", padding=5)
        self.props_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.props_text = tk.Text(self.props_frame, height=10, wrap=tk.WORD,
                                   font=('Consolas', 8))
        self.props_text.pack(fill=tk.BOTH, expand=True)
        
        # Right: Canvas
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(right)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(toolbar, text="Pipeline Canvas",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.node_count_label = ttk.Label(toolbar, text="Nodes: 0")
        self.node_count_label.pack(side=tk.RIGHT, padx=10)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#ecf0f1',
                                scrollregion=(0, 0, 2000, 1500),
                                highlightthickness=0)
        
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL,
                                  command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                  command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set,
                              yscrollcommand=v_scroll.set)
        
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Grid
        self._draw_grid()
        
        # Canvas callbacks
        self.canvas.on_node_selected = self._on_node_selected
        self.canvas.redraw_connections = self._redraw_connections
        
        self._connecting = False
        self._connect_from = None
    
    def _draw_grid(self):
        """Draw background grid."""
        for x in range(0, 2000, 50):
            self.canvas.create_line(x, 0, x, 1500, fill='#d5dbdb', width=1)
        for y in range(0, 1500, 50):
            self.canvas.create_line(0, y, 2000, y, fill='#d5dbdb', width=1)
    
    def _add_node(self, category, node_name):
        """Add a node to the canvas."""
        node_info = NODE_CATALOG[category][node_name]
        
        # Position: stagger based on count
        n = len(self.nodes)
        x = 100 + (n % 4) * 200
        y = 100 + (n // 4) * 100
        
        node = PipelineNode(
            self.canvas, node_name, category,
            node_info['type'], node_info['params'], x, y)
        self.nodes[node.id] = node
        
        self.node_count_label.config(text=f"Nodes: {len(self.nodes)}")
        if self.status_callback:
            self.status_callback(f"Added node: {node_name}")
    
    def _on_node_selected(self, node):
        """Handle node selection."""
        # Deselect previous
        if self.selected_node and self.selected_node.id != node.id:
            self.selected_node.selected = False
            self.selected_node._draw()
        
        self.selected_node = node
        
        # If connecting, make connection
        if self._connecting and self._connect_from:
            if self._connect_from.id != node.id:
                self._make_connection(self._connect_from, node)
            self._connecting = False
            self._connect_from = None
        
        # Update properties
        self._show_properties(node)
    
    def _show_properties(self, node):
        """Show node properties."""
        self.props_text.delete('1.0', tk.END)
        lines = [f"Node: {node.name}", f"Type: {node.node_type}",
                 f"ID: {node.id}", ""]
        lines.append("Parameters:")
        for k, v in node.params.items():
            lines.append(f"  {k} = {v}")
        self.props_text.insert(tk.END, '\n'.join(lines))
    
    def _connect_mode(self):
        """Enter connection mode."""
        if self.selected_node:
            self._connecting = True
            self._connect_from = self.selected_node
            if self.status_callback:
                self.status_callback(f"Click target node to connect from '{self.selected_node.name}'")
    
    def _make_connection(self, from_node, to_node):
        """Create connection between nodes."""
        # Check if already connected
        for fn, tn, _ in self.connections:
            if fn == from_node.id and tn == to_node.id:
                return
        
        from_node.connections_out.append(to_node)
        to_node.connections_in.append(from_node)
        
        line = self._draw_connection_line(from_node, to_node)
        self.connections.append((from_node.id, to_node.id, line))
        
        if self.status_callback:
            self.status_callback(f"Connected: {from_node.name} -> {to_node.name}")
    
    def _draw_connection_line(self, from_node, to_node):
        """Draw bezier-like connection line."""
        x1, y1 = from_node.get_output_port()
        x2, y2 = to_node.get_input_port()
        
        # Control points for smooth curve
        cx1 = x1 + abs(x2 - x1) * 0.4
        cy1 = y1
        cx2 = x2 - abs(x2 - x1) * 0.4
        cy2 = y2
        
        line = self.canvas.create_line(
            x1, y1, cx1, cy1, cx2, cy2, x2, y2,
            smooth=True, width=3, fill='#2c3e50',
            arrow=tk.LAST, arrowshape=(10, 12, 5))
        return line
    
    def _redraw_connections(self):
        """Redraw all connections."""
        new_connections = []
        for fn_id, tn_id, old_line in self.connections:
            self.canvas.delete(old_line)
            fn = self.nodes.get(fn_id)
            tn = self.nodes.get(tn_id)
            if fn and tn:
                line = self._draw_connection_line(fn, tn)
                new_connections.append((fn_id, tn_id, line))
        self.connections = new_connections
    
    def _delete_selected(self):
        """Delete selected node."""
        if not self.selected_node:
            return
        
        node = self.selected_node
        for item in node.canvas_items:
            self.canvas.delete(item)
        
        # Remove connections
        self.connections = [(f, t, l) for f, t, l in self.connections
                           if f != node.id and t != node.id]
        
        del self.nodes[node.id]
        self.selected_node = None
        self.node_count_label.config(text=f"Nodes: {len(self.nodes)}")
        self._redraw_connections()
    
    def _run_pipeline(self):
        """Execute the pipeline."""
        if not self.nodes:
            messagebox.showinfo("Info", "Pipeline is empty")
            return
        
        # Find source nodes
        sources = [n for n in self.nodes.values() if n.node_type == 'source']
        if not sources:
            messagebox.showinfo("Info", "No source node. Add a Source first.")
            return
        
        # Topological sort
        try:
            order = self._topo_sort()
            result_lines = [f"Pipeline execution order ({len(order)} nodes):"]
            for i, nid in enumerate(order):
                node = self.nodes[nid]
                result_lines.append(f"  {i+1}. [{node.node_type}] {node.name}")
            
            result_lines.append("\nPipeline ready for execution.")
            result_lines.append("(Full execution engine in development)")
            
            self.props_text.delete('1.0', tk.END)
            self.props_text.insert(tk.END, '\n'.join(result_lines))
            
            if self.status_callback:
                self.status_callback(f"Pipeline validated: {len(order)} nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline error: {e}")
    
    def _topo_sort(self):
        """Topological sort of nodes."""
        in_degree = {nid: 0 for nid in self.nodes}
        adj = {nid: [] for nid in self.nodes}
        
        for fn_id, tn_id, _ in self.connections:
            adj[fn_id].append(tn_id)
            in_degree[tn_id] += 1
        
        queue = [nid for nid, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Cycle detected in pipeline")
        
        return result
    
    def _clear_all(self):
        """Clear all nodes and connections."""
        self.canvas.delete('all')
        self.nodes.clear()
        self.connections.clear()
        self.selected_node = None
        self._draw_grid()
        self.node_count_label.config(text="Nodes: 0")
    
    def get_pipeline_data(self):
        """Get pipeline data for saving."""
        return {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'connections': [(f, t) for f, t, _ in self.connections]
        }
