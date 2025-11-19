import os
import sys

from .config import Config

class ClassifierConfig(Config):
    def get(self, field=None):
        if field not in self.config:
            print(f"Field '{field}' not found in config file: {self.config_path}")
            return None
        return self.config[field]
    
    def get_all_config(self):
        return self.config
    
    def get_template_file(self):
        if self.config is not None:
            return self.get('Classifier')['marker_template']
        return None
    
    def get_threshold(self):
        if self.config is not None:
            return self.get('Classifier')['threshold']
        return None
    
    def get_base_path(self):
        if self.config is not None:
            return self.get('Classifier')['base_path']
        return None
