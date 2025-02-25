import geopandas as gpd
import rasterio
from shapely.geometry import box

def trim_shp(stream_shp, dem_file, output_folder):
    shp = gpd.read_file(stream_shp)
    with rasterio.open(dem_file) as dem:
        dem_crs = dem.crs
        dem_bounds = dem.bounds

    # Reproject the shapefile to match the DEM's CRS if necessary
    if shp.crs != dem_crs:
        shp = shp.to_crs(dem_crs)

    # Create a bounding box from the DEM bounds
    dem_bbox = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

    # Trim the shapefile based on the DEM bounds
    trimmed_shp = shp[shp.intersects(dem_bbox)]

    trimmed_shp.to_file(output_path)
    print(f"Shapefile has been trimmed based on the DEM and saved to {output_path}")
