import json

class MapJsonData:
    def __init__(self, components, start_coordinate, original_shape):
        self.start_coordinate = start_coordinate
        self.original_shape = original_shape
        self.components = components
        self.labels = [
            {f"component_{i}": {
                  'label': [[int(x), int(y)] for (x, y) in component]
                }}
            for i, component in enumerate(components)
        ]

    def to_json_dict(self):
        merged = { 
            'info':
            {
                'start_coordinate': self.start_coordinate, 
                'shape':self.original_shape
            }
        }
        for label in self.labels:
            merged.update(label)
        return merged
    
    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_json_dict(), f)

    @classmethod
    def from_file(cls, filepath):
        
        with open(filepath, "r") as f:
            json_data = json.load(f)

        info = json_data.get('info', {})
        start_coordinate = info.get('start_coordinate', [0, 0])
        original_shape = info.get('shape', [0, 0])

        components = []
        for name, data in json_data.items():
            if name == 'info':
                continue
            coords = [tuple(coord) for coord in data["label"]]
            components.append(coords)
        return cls(components=components, start_coordinate=start_coordinate, original_shape=original_shape)
        
    def get_labels_dict(self):
        return {label.name: label for label in self.labels}

    def get_components_list(self):
        return self.components