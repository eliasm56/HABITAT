import geopandas as gpd
import os
import pandas as pd
import turning_function
from shapely.geometry import Polygon, MultiPolygon
from rtree import index  # R-tree spatial index for fast nearest neighbor search

# Paths
reference_dir = r"D:\building_footprints\fairbanks"
predicted_path = r"D:\HABITAT\maps\postprocessed\merged\alaska\AK_build_eqArea.shp"

print("Loading predicted dataset...")
# Load predicted dataset
predicted_gdf = gpd.read_file(predicted_path)

# Get CRS of predicted dataset
predicted_crs = predicted_gdf.crs
print(f"Predicted dataset loaded. CRS: {predicted_crs}")

# Remove empty and invalid geometries
predicted_gdf = predicted_gdf[predicted_gdf.geometry.notnull()]
predicted_gdf = predicted_gdf[predicted_gdf.is_valid]
if predicted_gdf.empty:
    print("Error: The predicted dataset contains only empty or invalid geometries. Exiting.")
    exit()

# Function to clip predicted buildings to reference area
def clip_to_reference(reference_gdf, predicted_gdf):
    """Clips the predicted dataset to the bounding box of the reference dataset."""
    reference_bounds = reference_gdf.total_bounds  # minx, miny, maxx, maxy
    predicted_gdf = predicted_gdf.cx[
        reference_bounds[0] : reference_bounds[2], reference_bounds[1] : reference_bounds[3]
    ]
    return predicted_gdf

# Function to handle MultiPolygons (extracts largest polygon)
def extract_largest_polygon(geometry):
    """Ensures geometry is a Polygon, selecting the largest if MultiPolygon."""
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)  # Select largest polygon
    return geometry  # Already a Polygon

# Function to compute Turning Function Distance
def compute_turning_function_distance(reference_gdf, predicted_gdf):
    """
    Computes the mean Turning Function Distance between reference and predicted footprints.
    """
    if reference_gdf.empty or predicted_gdf.empty:
        print("  Warning: One or both datasets are empty. Skipping Turning Function computation.")
        return None

    # Build spatial index for fast nearest neighbor search
    spatial_index = index.Index()
    for i, geom in enumerate(predicted_gdf.geometry):
        spatial_index.insert(i, geom.bounds, obj=geom)

    # Compute Turning Function Distance for nearest matches
    turning_distances = []
    for _, ref_row in reference_gdf.iterrows():
        ref_geom = ref_row.geometry
        possible_matches = list(spatial_index.intersection(ref_geom.bounds))

        if possible_matches:
            nearest_pred = min(
                (predicted_gdf.iloc[i].geometry for i in possible_matches),
                key=lambda g: ref_geom.distance(g),
                default=None,
            )

            if nearest_pred is not None:
                # Ensure we work with single Polygons
                ref_geom = extract_largest_polygon(ref_geom)
                nearest_pred = extract_largest_polygon(nearest_pred)

                # Convert to Nx2 list of points for turning_function package
                ref_coords = list(ref_geom.exterior.coords)
                pred_coords = list(nearest_pred.exterior.coords)

                # Compute Turning Function Distance
                try:
                    distance, _, _, _ = turning_function.distance(ref_coords, pred_coords, brute_force_updates=False)
                    turning_distances.append(distance)
                except Exception as e:
                    print(f"  Error computing Turning Function Distance: {e}")

    # Compute mean Turning Function Distance
    return sum(turning_distances) / len(turning_distances) if turning_distances else None

# Function to compute accuracy metrics
def compute_accuracy_metrics(reference_gdf, predicted_gdf):
    print("  Performing spatial operations for Turning Function Distance assessment...")

    # Remove invalid geometries
    reference_gdf = reference_gdf[reference_gdf.geometry.notnull()]
    reference_gdf = reference_gdf[reference_gdf.is_valid]
    predicted_gdf = predicted_gdf[predicted_gdf.geometry.notnull()]
    predicted_gdf = predicted_gdf[predicted_gdf.is_valid]

    # Skip if either dataset is empty
    if reference_gdf.empty or predicted_gdf.empty:
        print("  Warning: One or both datasets are empty after removing null and invalid geometries. Skipping this location.")
        return {"Turning Function Distance": None}
    
    # Clip predicted footprints to match reference study area
    predicted_gdf = clip_to_reference(reference_gdf, predicted_gdf)

    # Compute Turning Function Distance
    turning_function_dist = compute_turning_function_distance(reference_gdf, predicted_gdf)
    turning_function_dist = turning_function_dist if turning_function_dist is not None else -1  # Assign -1 if it fails

    print(f"  Turning Function Distance: {turning_function_dist:.4f}")
    return {"Turning Function Distance": turning_function_dist}

# Iterate through subdirectories
results = {}
subdirs = os.listdir(reference_dir)
print(f"Found {len(subdirs)} subdirectories in {reference_dir}.")

for subdir in subdirs:
    ref_path = os.path.join(reference_dir, subdir)
    if os.path.isdir(ref_path):
        print(f"Processing reference dataset in: {subdir}...")

        # Find the shapefile in the directory
        for file in os.listdir(ref_path):
            if file.endswith(".shp"):
                ref_shapefile = os.path.join(ref_path, file)
                print(f"  Found shapefile: {file}. Loading...")

                reference_gdf = gpd.read_file(ref_shapefile)

                # Reproject to predicted CRS if needed
                if reference_gdf.crs != predicted_crs:
                    print("  Reprojecting reference dataset to match predicted dataset CRS...")
                    reference_gdf = reference_gdf.to_crs(predicted_crs)
                    print("  Reprojection complete.")

                # Perform accuracy assessment
                metrics = compute_accuracy_metrics(reference_gdf, predicted_gdf)
                results[subdir] = metrics
                print(f"  Turning Function assessment completed for {subdir}.\n")
                break  # Process only one shapefile per directory

print("Processing complete. Compiling results...")

# Save results to CSV
print("Saving results to CSV...")
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv("D:/PhD_main/chapter_1/results/fairbanks_turning_function_results.csv", index=True)
print("Results saved as CAN_turning_function_results.csv")

print("Turning Function assessment results displayed.")
