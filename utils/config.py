import yaml
import os

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None
        if config_path is not None:
            self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, field=None):
        if field not in self.config:
            raise KeyError(f"Field '{field}' not found in config file: {self.config_path}")
        return self.config[field]
    
    def get_all_config(self):
        return self.config
    
    def get_pixel_cm(self):
        if (self.config is not None):
            return self.get('Predictor')['pixel_cm']
        else:
            return 1

