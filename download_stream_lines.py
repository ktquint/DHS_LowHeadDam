# Most recent, worked but really slow (2/7/25)
import geopandas as gpd
import os


def save_streamline_as_shapefile(gpkg_path, linkno, output_path):
    # Read the .gpkg file
    gdf = gpd.read_file(gpkg_path)

    # Filter the GeoDataFrame for the specified LINKNO
    streamline = gdf[gdf['LINKNO'] == linkno]

    if streamline.empty:
        print(f"No streamline found with LINKNO {linkno}")
        return

    # Save the filtered GeoDataFrame as a shapefile
    streamline.to_file(output_path, driver='ESRI Shapefile')
    print(f"Streamline data saved successfully to {output_path}")


# Example usage
gpkg_path = 'http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/streams_106.gpkg'
linkno = 120034993  # Replace with your LINKNO value
output_path = 'data/streamline.shp'

save_streamline_as_shapefile(gpkg_path, linkno, output_path)

# This program takes a while to run from the one time I ran it; however, it seems to successfully save the streamline
# associated with the linkno as a .shp file


'''
import geopandas as gpd
import os


def save_streamline_as_shapefile(gpkg_path, linkno, output_path):
    # Read the .gpkg file
    gdf = gpd.read_file(gpkg_path)

    # Filter the GeoDataFrame for the specified LINKNO
    streamline = gdf[gdf['LINKNO'] == linkno]

    if streamline.empty:
        print(f"No streamline found with LINKNO {linkno} in {gpkg_path}")
        return

    # Rename columns to shorter names
    column_mapping = {
        'TDXHydroRegion': 'TDXHydroRe',
        'TopologicalOrder': 'Topologica',
        'LengthGeodesicMeters': 'LengthGeod',
        'TerminalLink': 'TerminalLi'
    }
    streamline = streamline.rename(columns=column_mapping)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define the schema with appropriate field widths and precisions, including 'fid'
    schema = {
        'geometry': 'LineString',
        'properties': {
            'fid': 'int',
            'LINKNO': 'int',
            'TDXHydroRe': 'str:10',
            'Topologica': 'str:10',
            'LengthGeod': 'float:15.2',
            'TerminalLi': 'str:10',
            'USContArea': 'float:15.2',
            'DSContArea': 'float:15.2'
        }
    }

    # Add 'fid' column to the GeoDataFrame
    streamline = streamline.reset_index().rename(columns={'index': 'fid'})

    # Save the filtered GeoDataFrame as a shapefile with the defined schema
    streamline.to_file(output_path, driver='ESRI Shapefile', schema=schema)
    print(f"Streamline data saved successfully to {output_path}")


# Example usage
gpkg_path = 'http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/streams_105.gpkg'
linkno = 120362351  # Replace with your LINKNO value
output_path = 'data/streamline2.shp'

save_streamline_as_shapefile(gpkg_path, linkno, output_path)
'''
# The green code is what I was working on to try to improve the output; if needed, I can work on it more in the future
