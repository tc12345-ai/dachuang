"""
Plugin Protocols — 插件协议接口

Defines the contracts that every plugin must satisfy.
Uses typing.Protocol (PEP 544 style, compatible with Python 3.7 via comments).
"""

from typing import Any, Callable, Dict, List, Optional
import tkinter as tk
from tkinter import ttk


class PluginServiceBase:
    """
    Base class for plugin service layer — 插件服务层基类.

    Every plugin's service.py must subclass this.
    """
    plugin_id: str = ''

    def activate(self, bus, ctx: Dict[str, Any]):
        """Called when plugin is loaded. Store bus & context."""
        raise NotImplementedError

    def deactivate(self):
        """Called when plugin is unloaded."""
        pass


class PluginUIBase(ttk.Frame):
    """
    Base class for plugin UI panel — 插件 UI 面板基类.

    Must be a ttk.Frame so it can be embedded into Notebook tabs.
    """
    plugin_id: str = ''
    panel_title: str = 'Plugin'

    def __init__(self, parent, bus=None, ctx=None, **kw):
        super().__init__(parent, **kw)
        self.bus = bus
        self.ctx = ctx or {}

    def on_show(self):
        """Called when tab is selected."""
        pass

    def on_hide(self):
        """Called when tab is deselected."""
        pass

    def refresh(self):
        """Force refresh of UI contents."""
        pass


class PluginAPIBase:
    """
    Base class for plugin API endpoints — 插件 API 端点基类.

    Subclass must implement get_routes() returning a list of tuples:
      [('GET', '/api/my_plugin/health', handler), ...]
    """
    plugin_id: str = ''

    def get_routes(self) -> List[tuple]:
        """
        Return HTTP routes this plugin provides.

        Each tuple: (method: str, path: str, handler: Callable)
        handler signature: handler(body: dict, query: dict) -> (status: int, response: dict)
        """
        return []


class PluginManifest:
    """
    Plugin manifest descriptor — 插件清单描述符.

    Loaded from manifest.json inside each plugin directory.
    """
    def __init__(self, data: dict):
        self.id: str = data.get('id', '')
        self.name: str = data.get('name', '')
        self.version: str = data.get('version', '0.1.0')
        self.description: str = data.get('description', '')
        self.author: str = data.get('author', '')
        self.category: str = data.get('category', 'general')
        self.enabled: bool = data.get('enabled', True)
        self.dependencies: List[str] = data.get('dependencies', [])
        self.service_class: str = data.get('service_class', 'service.Service')
        self.ui_class: str = data.get('ui_class', 'ui.Panel')
        self.api_class: str = data.get('api_class', 'api.Api')
        self.events_published: List[str] = data.get('events_published', [])
        self.events_subscribed: List[str] = data.get('events_subscribed', [])
        self.ui_tab_title: str = data.get('ui_tab_title', self.name)
        self.ui_tab_icon: str = data.get('ui_tab_icon', '')
        self.min_python: str = data.get('min_python', '3.7')

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'name': self.name, 'version': self.version,
            'description': self.description, 'author': self.author,
            'category': self.category, 'enabled': self.enabled,
            'service_class': self.service_class, 'ui_class': self.ui_class,
            'api_class': self.api_class, 'ui_tab_title': self.ui_tab_title,
        }


# ─── Manifest JSON Schema (for validation reference) ───
MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "version"],
    "properties": {
        "id":                {"type": "string", "pattern": "^[a-z_]+$"},
        "name":              {"type": "string"},
        "version":           {"type": "string"},
        "description":       {"type": "string"},
        "author":            {"type": "string"},
        "category":          {"type": "string", "enum": [
            "ai", "hil", "domain", "stress", "ux", "general"]},
        "enabled":           {"type": "boolean"},
        "dependencies":      {"type": "array", "items": {"type": "string"}},
        "service_class":     {"type": "string"},
        "ui_class":          {"type": "string"},
        "api_class":         {"type": "string"},
        "events_published":  {"type": "array", "items": {"type": "string"}},
        "events_subscribed": {"type": "array", "items": {"type": "string"}},
        "ui_tab_title":      {"type": "string"},
        "min_python":        {"type": "string"},
    }
}
