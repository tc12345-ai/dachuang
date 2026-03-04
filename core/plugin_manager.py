"""
PluginManager — 插件管理器

Discovers, loads, and manages lifecycle of plugins from plugins/ directory.
Supports manifest-driven loading, dependency resolution, and hot-reload.
"""

import importlib
import importlib.util
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus, Events, make_event
from core.protocols import (
    PluginManifest, PluginServiceBase, PluginUIBase, PluginAPIBase)


class _LoadedPlugin:
    """Internal record of a loaded plugin."""
    def __init__(self, manifest: PluginManifest, path: str):
        self.manifest = manifest
        self.path = path
        self.service: Optional[PluginServiceBase] = None
        self.ui_class = None          # Class reference (not instance)
        self.api: Optional[PluginAPIBase] = None
        self.active = False


class PluginManager:
    """
    Plugin Manager — 插件管理器

    Lifecycle:
        1. discover(plugins_dir)   — scan for manifest.json
        2. load_all()              — import modules, instantiate services
        3. create_ui(parent)       — instantiate UI panels (deferred)
        4. register_api(router)    — register HTTP routes
        5. unload(plugin_id)       — teardown one plugin
        6. unload_all()            — teardown everything
    """

    def __init__(self, bus: EventBus = None):
        self.bus = bus or EventBus.instance()
        self._plugins: Dict[str, _LoadedPlugin] = {}
        self._plugins_dir: str = ''

    # ─── Discovery ───

    def discover(self, plugins_dir: str) -> List[PluginManifest]:
        """
        Scan plugins directory for manifest.json files.

        Returns list of discovered manifests (not yet loaded).
        """
        self._plugins_dir = plugins_dir
        manifests = []

        if not os.path.isdir(plugins_dir):
            return manifests

        for entry in sorted(os.listdir(plugins_dir)):
            plugin_path = os.path.join(plugins_dir, entry)
            manifest_path = os.path.join(plugin_path, 'manifest.json')

            if not os.path.isfile(manifest_path):
                continue

            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                m = PluginManifest(data)
                if m.id:
                    self._plugins[m.id] = _LoadedPlugin(m, plugin_path)
                    manifests.append(m)
            except Exception as e:
                print(f"[PluginManager] Bad manifest in {entry}: {e}")

        return manifests

    # ─── Loading ───

    def load_all(self, ctx: Dict[str, Any] = None):
        """Load and activate all enabled plugins."""
        ctx = ctx or {}
        # Dependency-order: simple topological sort
        order = self._resolve_order()
        for pid in order:
            self.load_plugin(pid, ctx)

    def load_plugin(self, plugin_id: str, ctx: Dict[str, Any] = None):
        """Load a single plugin by id."""
        ctx = ctx or {}
        lp = self._plugins.get(plugin_id)
        if lp is None or lp.active:
            return
        if not lp.manifest.enabled:
            return

        # Add plugin path to sys.path temporarily
        if lp.path not in sys.path:
            sys.path.insert(0, lp.path)

        try:
            # Import and instantiate service
            svc_mod, svc_cls = lp.manifest.service_class.rsplit('.', 1)
            mod = importlib.import_module(svc_mod)
            ServiceClass = getattr(mod, svc_cls)
            lp.service = ServiceClass()
            lp.service.plugin_id = lp.manifest.id
            lp.service.activate(self.bus, ctx)

            # Import UI class (don't instantiate yet — needs parent widget)
            try:
                ui_mod, ui_cls = lp.manifest.ui_class.rsplit('.', 1)
                umod = importlib.import_module(ui_mod)
                lp.ui_class = getattr(umod, ui_cls)
            except Exception:
                lp.ui_class = None

            # Import and instantiate API
            try:
                api_mod, api_cls = lp.manifest.api_class.rsplit('.', 1)
                amod = importlib.import_module(api_mod)
                ApiClass = getattr(amod, api_cls)
                lp.api = ApiClass()
                lp.api.plugin_id = lp.manifest.id
            except Exception:
                lp.api = None

            lp.active = True
            self.bus.publish(make_event(
                Events.PLUGIN_LOADED, source='plugin_manager',
                plugin_id=plugin_id, name=lp.manifest.name))
            print(f"[PluginManager] Loaded: {lp.manifest.name} v{lp.manifest.version}")

        except Exception as e:
            traceback.print_exc()
            print(f"[PluginManager] Failed to load {plugin_id}: {e}")

    # ─── UI creation ───

    def create_ui_panels(self, parent, bus=None, ctx=None):
        """
        Instantiate UI panels for all loaded plugins.

        Args:
            parent: Tkinter parent widget (Notebook)
        Returns:
            List of (manifest, panel_instance) tuples
        """
        panels = []
        for pid, lp in self._plugins.items():
            if not lp.active or lp.ui_class is None:
                continue
            try:
                panel = lp.ui_class(parent, bus=bus or self.bus, ctx=ctx or {})
                panel.plugin_id = lp.manifest.id
                panel.panel_title = lp.manifest.ui_tab_title
                panels.append((lp.manifest, panel))
            except Exception as e:
                print(f"[PluginManager] UI creation failed for {pid}: {e}")
        return panels

    # ─── API registration ───

    def get_all_routes(self) -> List[tuple]:
        """Collect HTTP routes from all loaded plugins."""
        routes = []
        for pid, lp in self._plugins.items():
            if lp.active and lp.api:
                try:
                    routes.extend(lp.api.get_routes())
                except Exception as e:
                    print(f"[PluginManager] API routes error for {pid}: {e}")
        return routes

    # ─── Unloading ───

    def unload_plugin(self, plugin_id: str):
        """Unload a single plugin."""
        lp = self._plugins.get(plugin_id)
        if lp is None or not lp.active:
            return

        # Teardown service
        if lp.service:
            try:
                lp.service.deactivate()
            except Exception:
                pass

        # Remove event subscriptions
        self.bus.unsubscribe_all(plugin_id)

        lp.active = False
        lp.service = None
        lp.api = None

        self.bus.publish(make_event(
            Events.PLUGIN_UNLOADED, source='plugin_manager',
            plugin_id=plugin_id))
        print(f"[PluginManager] Unloaded: {plugin_id}")

    def unload_all(self):
        """Unload all plugins."""
        for pid in list(self._plugins.keys()):
            self.unload_plugin(pid)

    # ─── Queries ───

    def list_plugins(self) -> List[dict]:
        """List all discovered plugins with status."""
        return [
            {**lp.manifest.to_dict(), 'active': lp.active, 'path': lp.path}
            for lp in self._plugins.values()
        ]

    def get_service(self, plugin_id: str) -> Optional[PluginServiceBase]:
        lp = self._plugins.get(plugin_id)
        return lp.service if lp else None

    def is_loaded(self, plugin_id: str) -> bool:
        lp = self._plugins.get(plugin_id)
        return lp.active if lp else False

    # ─── Internal ───

    def _resolve_order(self) -> List[str]:
        """Simple topological sort by dependencies."""
        visited = set()
        order = []

        def visit(pid):
            if pid in visited:
                return
            visited.add(pid)
            lp = self._plugins.get(pid)
            if lp:
                for dep in lp.manifest.dependencies:
                    visit(dep)
            order.append(pid)

        for pid in self._plugins:
            visit(pid)
        return order
