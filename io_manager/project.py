"""
Project Manager — 工程管理模块

Save/load workspace configurations and filter designs.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy__': True, 'data': obj.tolist(),
                    'dtype': str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def numpy_decoder(dct):
    """JSON decoder hook for numpy arrays."""
    if '__numpy__' in dct:
        return np.array(dct['data'], dtype=dct.get('dtype', 'float64'))
    return dct


class ProjectManager:
    """
    Project Manager — 工程管理器
    
    Handles saving and loading workspace state.
    """
    
    PROJECT_EXT = '.dspproj'
    PROJECT_VERSION = '1.0'
    
    def __init__(self):
        self.current_path = None
    
    def save_project(self, filepath: str, state: Dict[str, Any]):
        """
        Save project state to file.
        
        Args:
            filepath: Output file path (.dspproj)
            state: Project state dictionary containing:
                - filter_spec: Filter specification parameters
                - filter_result: Coefficients and info
                - spectrum_settings: Spectrum analysis settings
                - signal_path: Path to loaded signal file
                - ui_state: Window layout, tab selection, etc.
        """
        project = {
            'version': self.PROJECT_VERSION,
            'created': datetime.now().isoformat(),
            'tool': 'DSP Platform v1.0',
            'state': state,
        }
        
        if not filepath.endswith(self.PROJECT_EXT):
            filepath += self.PROJECT_EXT
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(project, f, cls=NumpyEncoder, indent=2,
                      ensure_ascii=False)
        
        self.current_path = filepath
    
    def load_project(self, filepath: str) -> Dict[str, Any]:
        """
        Load project state from file.
        
        Args:
            filepath: Project file path
        Returns:
            Project state dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            project = json.load(f, object_hook=numpy_decoder)
        
        version = project.get('version', '0.0')
        state = project.get('state', {})
        
        self.current_path = filepath
        return state
    
    def get_recent_projects(self, config_dir: str = None,
                            max_count: int = 10) -> list:
        """Get list of recent project paths."""
        if config_dir is None:
            config_dir = os.path.expanduser('~/.dsp-platform')
        
        recent_file = os.path.join(config_dir, 'recent.json')
        if not os.path.exists(recent_file):
            return []
        
        try:
            with open(recent_file, 'r') as f:
                recent = json.load(f)
            return [p for p in recent[:max_count] if os.path.exists(p)]
        except Exception:
            return []
    
    def add_to_recent(self, filepath: str, config_dir: str = None):
        """Add project to recent list."""
        if config_dir is None:
            config_dir = os.path.expanduser('~/.dsp-platform')
        os.makedirs(config_dir, exist_ok=True)
        
        recent_file = os.path.join(config_dir, 'recent.json')
        recent = []
        if os.path.exists(recent_file):
            try:
                with open(recent_file, 'r') as f:
                    recent = json.load(f)
            except Exception:
                recent = []
        
        filepath = os.path.abspath(filepath)
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        recent = recent[:20]
        
        with open(recent_file, 'w') as f:
            json.dump(recent, f, indent=2)
