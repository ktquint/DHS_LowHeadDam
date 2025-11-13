import xarray as xr
import numpy as np
import os
import shutil
from dask.diagnostics import ProgressBar


# Coordinates to match (latitude, longitude)
hf_coords = [[36.12085558, -95.98829985]]  # example near Tulsa, OK

# --- Load dataset ---
print("Opening NWM dataset...")
ds = xr.open_zarr(
    's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr',
    storage_options={'anon': True}
)

# --- Load coordinate arrays ---
print("Loading coordinate data...")
lats = ds['latitude'].values
lons = ds['longitude'].values
feature_ids = ds['feature_id'].values

# --- Find nearest feature for each coordinate ---
def find_nearest_feature(lat, lon, lats, lons, feature_ids):
    """Return the feature_id of the grid cell closest to (lat, lon)."""
    distance = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    idx = np.argmin(distance)
    return feature_ids[idx], lats[idx], lons[idx]

nearest_features = [find_nearest_feature(lat, lon, lats, lons, feature_ids) for lat, lon in hf_coords]

# Unpack results
nearest_feature_ids = [f[0] for f in nearest_features]
matched_lats = [f[1] for f in nearest_features]
matched_lons = [f[2] for f in nearest_features]

print("\nNearest features found:")
for i, (fid, lat, lon) in enumerate(nearest_features):
    print(f"  Point {i}: feature_id={fid}, lat={lat:.5f}, lon={lon:.5f}")

# --- Select streamflow data ---
print(f"\nSelecting streamflow data for {len(nearest_feature_ids)} feature(s)...")
streamflow_subset = ds['streamflow'].sel(feature_id=nearest_feature_ids)

# --- Resample to daily average ---
print("Resampling from hourly to daily average...")
daily_streamflow = streamflow_subset.resample(time='1D').mean()

# --- Rename and build final dataset ---
daily_streamflow = daily_streamflow.rename({'feature_id': 'rivid'})
final_ds = xr.Dataset({'streamflow': daily_streamflow})
final_ds = final_ds.assign_coords({
    'latitude': ('rivid', matched_lats),
    'longitude': ('rivid', matched_lons)
})

# --- Add metadata ---
final_ds.attrs = {
    'title': 'Selected Coordinates Daily Streamflow (cms)',
    'source': 'NOAA National Water Model v3.0 Retrospective',
    'units': 'cubic meters per second (cms)',
    'temporal_resolution': 'daily average',
    'original_resolution': 'hourly',
    'created_by': 'SABER preparation script',
    'selected_points': len(nearest_feature_ids)
}

final_ds['streamflow'].attrs = {
    'long_name': 'Daily Average Streamflow',
    'units': 'cubic feet per second',
    'standard_name': 'water_volume_transport_in_river_channel'
}

final_ds['rivid'].attrs = {
    'long_name': 'River Reach ID (NWM feature_id)',
    'description': 'Unique identifier for each river reach selected by nearest lat/lon'
}

# --- Save output ---
output_path = '../data/nwm_daily_retrospective.zarr'
print(f"\nSaving to: {output_path}")

if os.path.exists(output_path):
    shutil.rmtree(output_path)

with ProgressBar():
    final_ds.to_zarr(output_path, zarr_format=2, consolidated=True)

print("\n✅ Successfully saved daily retrospective data!")
print(f"Final dataset shape: {final_ds.streamflow.shape}")
print(f"Time range: {final_ds.time.min().values} to {final_ds.time.max().values}")
print(f"Number of points: {len(final_ds.rivid)}")
print("Units: CFS (cubic feet per second)")
print("Temporal resolution: Daily average")
