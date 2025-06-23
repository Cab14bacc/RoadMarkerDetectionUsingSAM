import json

class MapJsonData:
    def __init__(self, components, start_coordinate, original_shape, bbox_list=None):
        self.start_coordinate = start_coordinate
        self.original_shape = original_shape
        self.components = components
        self.bbox_list = bbox_list
        if bbox_list is None:
            self.measure_bbox()

        self.labels = []
        for i, (component, bbox) in enumerate(zip(components, self.bbox_list)):
            self.labels.append({
                f"component_{i}": {
                    'label': [[int(x), int(y)] for (x, y) in component],
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                }
            })
            
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
        bbox_list = []
        for name, data in json_data.items():
            if name == 'info':
                continue
            coords = [tuple(coord) for coord in data["label"]]
            components.append(coords)
            bbox_list.append(data['bbox'])
        return cls(components=components, start_coordinate=start_coordinate, original_shape=original_shape, bbox_list=bbox_list)
    
    def measure_bbox(self):
        print("measure bbox for each component")
        self.bbox_list = []
        for i, component in enumerate(self.components):
            xs = [pt[0] for pt in component]
            ys = [pt[1] for pt in component]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox = [min_x, min_y, max_x, max_y]
            self.bbox_list.append(bbox)

    def get_labels_dict(self):
        return {label.name: label for label in self.labels}

    def get_components_list(self):
        return self.components
    
    def get_start_coordinate(self):
        return self.start_coordinate
    
    def get_bbox_list(self):
        return self.bbox_list