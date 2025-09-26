import os
from PIL import Image, TiffImagePlugin, TiffTags
import pyproj
def from_TWD97_to_WGS84(TWD97_Coord):
    """
    Convert coordinates from TWD97 (EPSG:3826, Taiwan Datum 1997 TM2 zone 121) to WGS84 (EPSG:4326).
    
    Args:
        TWD97_Coord (tuple or list): (northing, easting) coordinates in TWD97 system
        
    Returns:
        tuple: (latitude, longitude) coordinates in WGS84 system
    """
    
    wgs84 = pyproj.CRS("EPSG:4326")  # WGS84
    twd97 = pyproj.CRS("EPSG:3826")  # TWD97 TM2 zone 121
    
    transformer = pyproj.Transformer.from_crs(twd97, wgs84, always_xy=True)
    
    northing, easting = TWD97_Coord
    
    lon, lat = transformer.transform(easting, northing)
        
    return lat, lon

def print_tiff_tags(tiff_path):
    with Image.open(tiff_path) as img:
        print(f"TIFF file: {tiff_path}")
        print("Tags:")
        for tag, value in img.tag_v2.items():
            tag_name = TiffTags.TAGS.get(tag, tag)
            # if tag_name in ["ModelPixelScaleTag", "ImageWidth", "ImageLength", "ModelTiepointTag"]:
            if tag_name in ["ImageWidth", "ImageLength"]:
                # print(f"{tag} ({tag_name}): {value}")
                print(f"{value * 5}")

        tie_point_real = img.tag_v2[33922][3:5]
        scale = img.tag_v2[33550][0]
        half_width = img.tag_v2[256] // 2
        half_height = img.tag_v2[257] // 2

        center_x = half_width * scale + tie_point_real[0]
        center_y = - half_height * scale + tie_point_real[1]

        # print("center WGS84 coord: ", from_TWD97_to_WGS84((center_y, center_x)))
        # print(from_TWD97_to_WGS84((center_y, center_x)))
        center_lat, center_lon = from_TWD97_to_WGS84((center_y, center_x))
        print(center_lat)
        print(center_lon)
        print(center_lat, ",",center_lon)

if __name__ == "__main__":
    import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python print_tiff_tags.py <tiff_file>")
    # else:
    image_dir = r"c:\Users\Leo\Documents\GitRepos\OSMAPI\InputImages\新生南路路口"
    num_image = 8
    for i in range(1, num_image + 1):
        i = f"{i:04d}"
        i += ".tif"

        print_tiff_tags(os.path.join(image_dir,i))

