import tifffile

class TiffLoader:
    def __init__(self, path):
        self.tiff_path = path
        self.image = tifffile.memmap(self.tiff_path)
        self._load_tags()

    def memory_map_loading(self):
        return self.image
    
    def get_partial_image(self, start, end):
        if start < 0 or end > self.image.shape[0]:
            raise ValueError("Start and end indices must be within the image dimensions.")
        return self.image[start:end, :, :]

    def _load_tags(self):
        with tifffile.TiffFile(self.tiff_path) as tif:
            tiff_info = tif.pages[0].tags
            self.start_coordinate = tiff_info.get('ModelTiepointTag').value[3:6]
            self.crs = tiff_info.get('GeoAsciiParamsTag').value
            self.scale = tiff_info.get('ModelPixelScaleTag').value

    def get_start_coordinate(self):
        return self.start_coordinate

    
    def get_coordinate_system(self):
        return self.crs

        
    def get_shape(self):
        return self.image.shape
    
    def get_pixel_scale(self):
        return self.scale


        