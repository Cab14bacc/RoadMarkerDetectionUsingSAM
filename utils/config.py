import yaml
import os

class Config:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = None
        if config_path is not None:
            self.config = self.load_config()

    def set_config_with_dict(self, config_dict):
        self.config = config_dict

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, field=None):
        if field not in self.config:
            print(f"Field '{field}' not found in config file: {self.config_path}")
            return None
        return self.config[field]
    
    def get_all_config(self):
        return self.config
    
    def get_pixel_cm(self):
        if (self.config is not None):
            return self.get('Common')['pixel_cm']
        else:
            return 1
    
    def get_threshold_from_usage(self, usage):
        index = self.get_index_from_usage(usage)
        if (self.config is not None):
            predict_config = self.get('Predictor')
            return predict_config['min_area_threshold'][index], predict_config['max_area_threshold'][index]
        return 30000, 100

    def get_index_from_usage(self, usage):
        if (self.config is not None):
            usage_list = self.get(field='Common')['usage_list']            
            if (usage in usage_list):
                return usage_list.index(usage)
        return 0
    
    def get_sample_points_interval(self, usage):
        index = self.get_index_from_usage(usage)
        if (self.config is not None):
            predict_config = self.get('Predictor')
            return predict_config['sample_points_interval'][index]
        return 32
