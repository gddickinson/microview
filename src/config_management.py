import os
import json

class ConfigManagement:
    def __init__(self, parent):
        self.parent = parent
        self.config_file = os.path.expanduser('~/.microview_config.json')

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.parent.recent_files = config.get('recent_files', [])
                self.parent.auto_load_plugins = config.get('auto_load_plugins', [])
        else:
            self.parent.recent_files = []
            self.parent.auto_load_plugins = []

    def save_config(self):
        config = {
            'recent_files': self.parent.recent_files,
            'auto_load_plugins': self.parent.auto_load_plugins
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
