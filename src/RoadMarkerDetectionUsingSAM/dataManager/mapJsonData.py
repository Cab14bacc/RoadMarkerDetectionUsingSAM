import json
import numpy as np
import warnings
from skimage.draw import polygon
import cv2
from colorsys import hsv_to_rgb
import random

class JsonData:
    """
    params:
        components: 
            List of (x, y) image coordinates.

        original_shape: 
            Shape of Image

        bbox_list: 
            List of bboxes of each component
    """
    def __init__(self, components, original_shape, bbox_list=None):
        self.original_shape = original_shape
        self.components = components
        self.bbox_list = bbox_list
        if bbox_list is None:
            self.measure_bbox()

        self.labels = []
        self.labels = []
        
        for i, (component, bbox) in enumerate(zip(self.components, self.bbox_list)):
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
        original_shape = info.get('shape', [0, 0])

        components = []
        bbox_list = []
        for name, data in json_data.items():
            if name == 'info':
                continue
            coords = [tuple(coord) for coord in data["label"]]
            components.append(coords)
            bbox_list.append(data['bbox'])
        return cls(components=components, original_shape=original_shape, bbox_list=bbox_list)
    
    def measure_bbox(self):
        print("measure bbox for each component")
        self.bbox_list = []
        for i, component in enumerate(self.components):
            xs = [pt[0] for pt in component]
            ys = [pt[1] for pt in component]
            min_x, max_x = np.min(xs), np.max(xs)
            min_y, max_y = np.min(ys), np.max(ys)
            bbox = [min_x, min_y, max_x, max_y]
            self.bbox_list.append(bbox)

    def get_labels_dict(self):
        return {key: value for label in self.labels for key, value in label.items()}
    
    def get_components_list(self):
        return self.components
    
    def get_bbox_list(self):
        return self.bbox_list
    
    # shape: height, width
    def get_shape(self):
        return self.original_shape
    
    def to_mask(self, component_type=None, mask_type=None, debug_texts = None):
        """
        params:
            component_type: "polygon" or "spline"
            mask_type: "colored", "labeled", "boolean"
            debug_type: "Index", "Type"
        """

        assert debug_texts is None or \
            (isinstance(debug_texts, list) and len(debug_texts) == len(self.get_components_list()))

        if component_type is None:
            component_type = "polygon"

        if mask_type is None:
            mask_type = "colored"
        elif mask_type != "colored" and debug_texts is not None:
            warnings.warn("debug texts can only be used with colored mask type, ignoring debug_texts", RuntimeWarning)
            debug_texts = None



        height, width = self.original_shape[:2]
        
        if len(self.original_shape) < 2 or len(self.original_shape) >= 3: 
            warnings.warn("original_shape of json data expects 2 or 3 dimensions, assumes first 2 dimensions as (height, width)", RuntimeWarning)
        
        components = self.get_components_list()


        if mask_type == "colored":
            vis_mask = np.zeros(shape=(height, width, 3), dtype=np.uint8) 
            if len(components) <= 1:
                colors = [(255, 255, 255)]
            else:
                colors = [255 * np.array(hsv_to_rgb(i * (1 / (len(components) - 1)), 1, 1)) for i in range(len(components))]
        elif mask_type == "boolean":
            vis_mask = np.zeros(shape=(height, width), dtype=np.uint8) 
            colors = [1]
        elif mask_type == "labeled":
            vis_mask = np.zeros(shape=(height, width), dtype=np.int32) 
            colors = list(range(0, len(components), 1))


        
        if component_type == "polygon":
            self._to_polygon_mask(vis_mask, colors, debug_texts)
        elif component_type == "spline":
            self._to_spline_mask(vis_mask, colors, debug_texts)

        if mask_type == "boolean":
            vis_mask = vis_mask > 0
            
        # row_coords, col_coords = np.nonzero(vis_mask)
        # print(max(vis_mask[row_coords, col_coords]))
        # vis_mask = cv2.resize(vis_mask, np.array(vis_mask.shape[:2]) // 1)
        # cv2.imshow("visdjf;slfdkj", vis_mask)
        # cv2.waitKey()

        return vis_mask
    
    def _to_spline_mask(self, mask, colors, debug_texts = None):
        for comp_idx, coords in enumerate(self.components):
            color = colors[min(comp_idx, len(colors) - 1)]
            for i in range(len(coords) - 1):
                cv2.line(mask, tuple(coords[i]), tuple(coords[i+1]), color, thickness=1)
        
        return mask

    def _to_polygon_mask(self, mask, colors,  debug_texts = None):
        for comp_idx, coords in enumerate(self.components):
            color = colors[min(comp_idx, len(colors) - 1)]
            col_coords = np.array(coords)[:, 0]
            row_coords = np.array(coords)[:, 1]
            rr, cc = polygon(row_coords, col_coords, mask.shape[:2])
            mask[rr, cc] = color
        
        return mask
    

class MapJsonData(JsonData):
    def __init__(self, components, start_coordinate, original_shape, bbox_list=None):
        super().__init__(components, original_shape, bbox_list)
        self.start_coordinate = start_coordinate

            
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
    
    
    def get_start_coordinate(self):
        return self.start_coordinate
    

class SplineJsonData(JsonData):
    """
    params:
        components: 
            List of components. Each component is a List of (x, y) image coordinates.

        component_spline_indices: 
            List of spline correspondence info of each component. 
            A correspondence info consists of spline indices, and has the same size as <components>. 
            A component may consist of multiple splines.
            e.g. A spline correspondence info: [1,1,1,3,3], 
                first 3 points belong to spline 1, and the latter belong to spline 3.
        
        road_line_groups:
            List of grouping info of each component. A grouping info is a list of groups.
            Each group is a list of spline indices. 
            Each group is meant to denote a type of road line (see <road_line_types>).
            e.g. A grouping info [[1,3], [4,5]], means spline 1, 3 is in the same group forming a solid yellow line.
        
        road_line_types:
            List of type info of each component. A type info is a list of types, each corresponds to a group.
            e.g. A grouping info ["yellow solid", "yellow dashed"], 
                meant this component is a double yellow line, where one is dashed and another solid. 

        original_shape: 
            (height , width), shape of image which this spline data is based in. 

        bbox_list: 
            List of bboxes of each component
    """
    def __init__(self, components, component_spline_indices, road_line_groups, road_line_types, original_shape, bbox_list=None):
        super().__init__(components, original_shape, bbox_list)
        self.road_line_types = road_line_types
        self.component_spline_indices = component_spline_indices
        self.road_line_groups = road_line_groups

        if not (len(self.road_line_types) == len(self.components) == len(self.component_spline_indices)):
            raise ValueError(f"Lengths of spline_types (len: {len(self.road_line_types)}), components (len: {len(self.components)}), and component_spline_indices (len: {len(self.component_spline_indices)}) must be equal.")
        
        for label_idx, label in enumerate(self.labels):
            key = f"component_{label_idx}"
            label[key]["road_line_types"] = road_line_types[label_idx]
            label[key]["spline_indices"] = component_spline_indices[label_idx]
            label[key]["road_line_groups"] = road_line_groups[label_idx]

    def to_json_dict(self):
        merged = { 
            'info':
            {
                'shape':self.original_shape,
            }
        }

        for label in self.labels:
            merged.update(label)
        return merged

    @classmethod
    def from_file(cls, filepath):
        
        with open(filepath, "r") as f:
            json_data = json.load(f)

        info = json_data.get('info', {})
        original_shape = info.get('shape', [0, 0])

        components = []
        bbox_list = []
        road_line_types = []
        component_spline_indices = []
        road_line_groups = []
        for name, data in json_data.items():
            if name == 'info':
                continue
            coords = [tuple(coord) for coord in data["label"]]
            components.append(coords)
            bbox_list.append(data['bbox'])
            road_line_types.append(data["road_line_types"])
            component_spline_indices.append(data["spline_indices"])
            road_line_groups.append(data["road_line_groups"])
            
        return cls(components=components, component_spline_indices=component_spline_indices, road_line_groups=road_line_groups, road_line_types=road_line_types, original_shape=original_shape, bbox_list=bbox_list)
    
    def get_road_line_types(self):
        return self.road_line_types
    
    def get_component_spline_indices(self):
        return self.component_spline_indices
    
    def get_road_line_groups(self):
        return self.road_line_groups
    
    
    def _to_spline_mask(self, mask, colors,  debug_texts = None):
        assert isinstance(debug_texts, list) 
        
        debug_text = []
        for comp_idx, coords in enumerate(self.components):

            color = colors[min(comp_idx, len(colors) - 1)]
            spline_indicies = self.component_spline_indices[comp_idx]
            _,  indices_of_unique_vals = np.unique(spline_indicies, return_index=True) # should just use road line types to get unique indices
            unique_spline_indices = np.array(spline_indicies)[indices_of_unique_vals] # to guarantee stableness

            if debug_texts is not None:
                debug_text = debug_texts[comp_idx]
                if isinstance(debug_text, str):
                    debug_text = [debug_text]


            for j, spline_idx in enumerate(unique_spline_indices):
                spline_pt_indices = np.nonzero(spline_indicies == spline_idx)[0]

                cur_spline_text = debug_text[j] if j < len(debug_text) else ""

                for i in range(len(spline_pt_indices) - 1):
                    spline_pt1_idx = spline_pt_indices[i] 
                    spline_pt2_idx = spline_pt_indices[i + 1]
                    
                    if i == 0:
                        cv2.putText(mask, cur_spline_text, np.array([coords[spline_pt1_idx][0], min(coords[spline_pt1_idx][1] + 100 * random.random(), self.original_shape[0])]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (127, 255, 255), 3, cv2.LINE_AA)
                    cv2.line(mask, tuple(coords[spline_pt1_idx]), tuple(coords[spline_pt2_idx]), color, thickness=2)
        
        return mask