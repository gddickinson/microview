#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:52:07 2024

@author: george
"""

# microview/plugin_manager.py

import os
import importlib
import inspect
from typing import Dict, List
from microview.plugins.base_plugin import BasePlugin

class PluginManager:
    def __init__(self, microview):
        self.microview = microview
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_dir = os.path.join(os.path.dirname(__file__), 'plugins')

    def discover_plugins(self) -> None:
        """Discover all plugins in the plugin directory."""
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"microview.plugins.{module_name}")
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                            plugin = obj(self.microview)
                            self.plugins[plugin.name] = plugin
                except Exception as e:
                    print(f"Error loading plugin {module_name}: {str(e)}")

    def get_plugin_list(self) -> List[str]:
        """Return a list of all available plugin names."""
        return list(self.plugins.keys())

    def run_plugin(self, plugin_name: str) -> None:
        """Run a specific plugin by name."""
        if plugin_name in self.plugins:
            try:
                self.plugins[plugin_name].run()
            except Exception as e:
                print(f"Error running plugin {plugin_name}: {str(e)}")
        else:
            print(f"Plugin {plugin_name} not found.")

    def cleanup_plugins(self) -> None:
        """Clean up all plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up plugin {plugin.name}: {str(e)}")
