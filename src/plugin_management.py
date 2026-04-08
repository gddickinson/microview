import os
import importlib.util
import traceback
import logging

logger = logging.getLogger(__name__)

class PluginManagement:
    def __init__(self, parent):
        self.parent = parent
        self.plugins = {}

    def load_plugins(self):
        logger.info("Starting to load plugins")
        plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        logger.info(f"Plugins directory: {plugins_dir}")
        startup_file = os.path.join(plugins_dir, 'startup.json')
        logger.info(f"Startup file path: {startup_file}")

        enabled_plugins = self.get_enabled_plugins(startup_file)

        for item in os.listdir(plugins_dir):
            self.load_plugin(item, plugins_dir, enabled_plugins)

        logger.info(f"Finished loading plugins. Total plugins loaded: {len(self.plugins)}")
        return self.plugins

    def get_enabled_plugins(self, startup_file):
        if os.path.exists(startup_file):
            with open(startup_file, 'r') as f:
                import json
                startup_config = json.load(f)
            enabled_plugins = startup_config.get('enabled_plugins', [])
            logger.info(f"Enabled plugins from startup file: {enabled_plugins}")
        else:
            enabled_plugins = []
            logger.warning("Startup file not found. All plugins will be loaded.")
        return enabled_plugins

    def load_plugin(self, item, plugins_dir, enabled_plugins):
        plugin_dir = os.path.join(plugins_dir, item)
        logger.info(f"Checking directory: {plugin_dir}")
        if os.path.isdir(plugin_dir):
            plugin_file = os.path.join(plugin_dir, f"{item}.py")
            logger.info(f"Looking for plugin file: {plugin_file}")
            if os.path.exists(plugin_file):
                plugin_name = item
                logger.info(f"Found plugin: {plugin_name}")
                if plugin_name in enabled_plugins or not enabled_plugins:
                    try:
                        logger.info(f"Attempting to load plugin: {plugin_name}")
                        spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        plugin_class = getattr(module, 'Plugin')
                        plugin = plugin_class(self.parent)
                        self.plugins[plugin.name] = plugin
                        logger.info(f"Successfully loaded plugin: {plugin.name}")
                    except Exception as e:
                        logger.error(f"Error loading plugin {plugin_name}: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.info(f"Plugin {plugin_name} is not enabled in startup.json")
            else:
                logger.warning(f"Plugin file not found: {plugin_file}")
        else:
            logger.info(f"Not a directory: {plugin_dir}")

    def update_plugin_list(self, plugin_list):
        plugin_list.clear()
        for plugin_name in self.plugins.keys():
            plugin_list.addItem(plugin_name)

    def run_selected_plugin(self, plugin_list):
        selected_items = plugin_list.selectedItems()
        if selected_items:
            plugin_name = selected_items[0].text()
            if plugin_name in self.plugins:
                self.plugins[plugin_name].run()
            else:
                print(f"Plugin {plugin_name} not found.")
        else:
            print("No plugin selected.")

    def close_all_plugins(self):
        for plugin_name, plugin in self.plugins.items():
            try:
                if hasattr(plugin, 'close') and callable(plugin.close):
                    plugin.close()
                logger.info(f"Closed plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error closing plugin {plugin_name}: {str(e)}")
