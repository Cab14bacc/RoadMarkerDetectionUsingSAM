from .config import Config


class PredictorConfig(Config):

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
