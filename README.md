# DHS_LowHeadDam
- all code associated with the DHS research project on low head dams.

# PREPARE DATA
  ## download_dem.py
  - this program reads in a lat/long, then creates a bounding box that is used to download a 3DEP USGS DEM.
  ## download_land.py
  - this does the same as download_dem.py, but for land use.
  - we'll either get the land use data from NLCD, or some global database; it shouldn't matter for the rating curves—we may even try to override this part of ARC.
  ## download_stream.py
  - same as download_dem.py, but we'll download either the NHDPlus or TDX Hydro (GEOGLOWS) streams.
  - for either one we'll need the stream ID to isolate the segment we want.
  - GEOGLOWS has lat/long embedded into the URL, so a quick python script can create a list of urls that go directly to each stream.
    ### streamflow.py
    - this will take the GEOGLOWS LINKNO to retrieve a NetCDF file containing the return period flows.
  ## trim_raster.py
  - this program will trim the rasters (land use and dem [potentially stream if we have to] down to size).
  ## arc_inputs.py
  - this program will create the input files used in the Automated_Rating_Curve_Generator.py
    - input files (DEM, STRM, LAND, FLOW) & output files (VDT database) need to be specified
  
# DATA ANALYSIS
  ## risk_calc.py
  - this program will calculate the conjugate and flip depths and read the VDT file from ARC to create plots of each rating curve and to produce a range of dangerous flows/depths.
  
