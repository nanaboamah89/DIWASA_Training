from __future__ import annotations

import os
import time
import random
import pandas as pd
import numpy as np
import xarray as xr           # Works with labeled multi-dimensional arrays (e.g., satellite data)
import geopandas as gpd       # Manages and analyzes vector geospatial data (shapefiles, GeoJSON)
import rioxarray              # Integrates rasterio with xarray for georeferenced raster data handling
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from rasterio.plot import show as rio_show
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling

from pystac_client import Client # Connects to STAC catalogs to search and access satellite data
import odc.stac                  # Loads STAC data into analysis-ready datacubes for processing
import folium                    # used by display_data

# ---------------------------------------------------------------------
# Utilities (single, non-duplicated)
# ---------------------------------------------------------------------
def _maybe_import_tqdm(progress: bool):
    if not progress:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def _maybe_dask_progress(progress: bool):
    if not progress:
        return nullcontext()
    try:
        from dask.diagnostics import ProgressBar  # type: ignore
        return ProgressBar()
    except Exception:
        # If dask ProgressBar isn't available, just no-op
        return nullcontext()


def _maybe_tqdm(iterable, total=None, desc: str = "", unit: str = "", enable: bool = True):
    """Wrapper for tqdm if available."""
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


@contextmanager
def _aws_env(unsigned_s3: bool, aws_region: str):
    """Temporarily set AWS-related env vars for unsigned S3 reads."""
    prev_no_sign = os.environ.get("AWS_NO_SIGN_REQUEST")
    prev_region = os.environ.get("AWS_REGION")

    try:
        if unsigned_s3:
            os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
        if aws_region:
            os.environ["AWS_REGION"] = aws_region
        yield
    finally:
        if prev_no_sign is None:
            os.environ.pop("AWS_NO_SIGN_REQUEST", None)
        else:
            os.environ["AWS_NO_SIGN_REQUEST"] = prev_no_sign

        if prev_region is None:
            os.environ.pop("AWS_REGION", None)
        else:
            os.environ["AWS_REGION"] = prev_region


def display_data(dataset, rows, color_code, column):
    """
    Display a subset of a dataset on an interactive map.

    Parameters
    ----------
    dataset : GeoDataFrame
        The dataset to visualize.
    rows : int
        Number of rows (records) to display.
    color_code : dict
        Dictionary mapping class names or codes to color values (e.g., {"forest": "green"}).
    column : str
        Name of the column containing land cover or classification information.

    Returns
    -------
    folium.Map
        An interactive map with points colored according to their class.
    """
    # Select subset of dataset
    points = dataset.head(rows)

    # Ensure column exists
    if column not in points.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {list(points.columns)}")

    # Map class to color
    points["color"] = points[column].map(color_code)

    # Display unique classes (optional)
    print("Unique classes found:", points[column].unique())

    # Visualize
    return points.explore(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Imagery ©2024 Landsat/Copernicus, Map data ©2024 Google",
        popup=True,
        marker_kwds=dict(radius=10, fillOpacity=0.8, weight=5),
        color=points["color"]
    )


def connect_stac(
    collection_id: str,
    bbox: Tuple[float, float, float, float],
    time_range: str,
    bands: Sequence[str],
    resolution: Union[float, Tuple[float, float]],
    *,
    stac_url: str = "https://explorer.digitalearth.africa/stac",
    groupby: str = "solar_day",
    compute: bool = True,
    chunks: Optional[Dict[str, int]] = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,  # seconds (base delay)
    unsigned_s3: bool = True,
    aws_region: str = "af-south-1",
    progress: bool = True,  # <-- turn progress bars on/off
) -> xr.Dataset:
    """
    Load a median composite from a STAC collection (DE Africa default) over a bbox/time window,
    with optional progress bars.
    """
    # --- Basic validation ---
    if not bands:
        raise ValueError("`bands` cannot be empty.")
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise ValueError("`bbox` must be a 4-tuple: (minx, miny, maxx, maxy).")
    if isinstance(resolution, (tuple, list)) and len(resolution) != 2:
        raise ValueError("`resolution` must be a float or a (xres, yres) tuple.")

    chunks = chunks or {}
    tqdm_bar = _maybe_import_tqdm(progress)

    with _aws_env(unsigned_s3=unsigned_s3, aws_region=aws_region):
        # Retry loop with exponential backoff + jitter
        for attempt in range(1, max_retries + 1):
            try:
                # Prepare a step progress bar (6 main steps)
                total_steps = 6
                pbar_ctx = nullcontext()
                if tqdm_bar is not None:
                    pbar_ctx = tqdm_bar(total=total_steps, desc=f"STAC Attempt {attempt}/{max_retries}", unit="step")

                with pbar_ctx as pbar:
                    # Step 1: Open STAC
                    if tqdm_bar is None:
                        print(f"[connect_stac] Opening STAC: {stac_url}")
                    catalog = Client.open(stac_url)
                    if pbar: pbar.update(1)

                    # Step 2: Search items
                    if tqdm_bar is None:
                        print(f"[connect_stac] Searching: collection={collection_id}, time={time_range}, bbox={bbox}")
                    search = catalog.search(
                        collections=[collection_id],
                        bbox=bbox,
                        datetime=time_range,
                        limit=100,
                    )
                    items = list(search.get_items())
                    if tqdm_bar is None:
                        print(f"[connect_stac] Found {len(items)} item(s)")
                    if pbar: pbar.update(1)

                    if not items:
                        raise ValueError(
                            "No STAC items found. Check your collection_id, time_range, and bbox."
                        )

                    # Step 3: Load with odc.stac
                    if tqdm_bar is None:
                        print("[connect_stac] Loading items with odc.stac.load ...")
                    ds = odc.stac.load(
                        items,
                        bands=list(bands),
                        bbox=bbox,
                        resolution=resolution,
                        groupby=groupby,
                        chunks=chunks,
                    )
                    if pbar: pbar.update(1)

                    # Step 4: Validate time dimension
                    if "time" not in ds.dims or ds.sizes.get("time", 0) == 0:
                        raise ValueError("Loaded dataset has no time dimension or zero time slices.")
                    if pbar: pbar.update(1)

                    # Step 5: Median reduction
                    if tqdm_bar is None:
                        print("[connect_stac] Computing median composite (lazy)...")
                    composite = ds.median(dim="time")
                    if pbar: pbar.update(1)

                    # Step 6: Compute (optional) with Dask progress
                    if compute:
                        if tqdm_bar is None:
                            print("[connect_stac] Computing (this may take a while)...")
                        with _maybe_dask_progress(progress):
                            composite = composite.compute()
                        if tqdm_bar is None:
                            print("[connect_stac] Composite computed (in-memory).")
                    else:
                        if tqdm_bar is None:
                            print("[connect_stac] Returning lazy Dask graph (not computed).")
                    if pbar: pbar.update(1)

                return composite

            except Exception as exc:
                # If last attempt, re-raise
                if attempt == max_retries:
                    if tqdm_bar is not None:
                        tqdm_bar.write(f"[connect_stac] Attempt {attempt}/{max_retries} failed: {exc}")
                        tqdm_bar.write("[connect_stac] Max retries reached. Raising the last exception.")
                    else:
                        print(f"[connect_stac] Attempt {attempt}/{max_retries} failed: {exc}")
                        print("[connect_stac] Max retries reached. Raising the last exception.")
                    raise

                # Backoff with jitter
                backoff = retry_delay * (2 ** (attempt - 1))
                sleep_for = backoff + random.uniform(0, retry_delay / 2.0)
                if tqdm_bar is not None:
                    tqdm_bar.write(f"[connect_stac] Attempt {attempt}/{max_retries} failed: {exc}")
                    tqdm_bar.write(f"[connect_stac] Retrying in {sleep_for:.1f} seconds ...")
                else:
                    print(f"[connect_stac] Attempt {attempt}/{max_retries} failed: {exc}")
                    print(f"[connect_stac] Retrying in {sleep_for:.1f} seconds ...")
                time.sleep(sleep_for)


def clip_roi(
    roi: Union[gpd.GeoDataFrame, gpd.GeoSeries, BaseGeometry],
    composite: xr.Dataset,
    *,
    assume_wgs84_if_missing: bool = True,
    drop: bool = True,
    all_touched: bool = False,
    buffer_m: float = 0.0,
) -> xr.Dataset:
    """
    Clip an xarray/rioxarray dataset to a region of interest (ROI) with progress feedback.
    """
    steps = [
        "Validate CRS",
        "Normalize ROI to GeoDataFrame",
        "Reproject ROI to composite CRS",
        "Apply optional buffer",
        "Dissolve geometry",
        "Clip dataset",
    ]

    with tqdm(total=len(steps), desc="Clipping Progress", unit="step") as pbar:
        # Step 1: CRS validation
        if not hasattr(composite, "rio"):
            raise ImportError("Composite dataset lacks `.rio` accessor. Install rioxarray.")
        if composite.rio.crs is None:
            raise ValueError("Composite CRS missing. Use `composite.rio.write_crs(...)`.")
        pbar.update(1)

        # Step 2: Normalize ROI
        if isinstance(roi, (gpd.GeoSeries, BaseGeometry)):
            roi_gdf = gpd.GeoDataFrame(geometry=[roi] if isinstance(roi, BaseGeometry) else roi)
        elif isinstance(roi, gpd.GeoDataFrame):
            roi_gdf = roi.copy()
        else:
            raise TypeError("`roi` must be a GeoDataFrame, GeoSeries, or shapely geometry.")
        pbar.update(1)

        # Step 3: Reproject to composite CRS
        if roi_gdf.crs is None:
            if assume_wgs84_if_missing:
                roi_gdf = roi_gdf.set_crs(epsg=4326)
            else:
                raise ValueError("ROI CRS missing. Provide CRS or set `assume_wgs84_if_missing=True`.")
        roi_gdf = roi_gdf.to_crs(composite.rio.crs)
        pbar.update(1)

        # Step 4: Optional buffer
        if buffer_m and buffer_m != 0.0:
            roi_gdf["geometry"] = roi_gdf.geometry.buffer(buffer_m)
        pbar.update(1)

        # Step 5: Dissolve geometry
        roi_geom = roi_gdf.unary_union
        if roi_geom.is_empty:
            raise ValueError("ROI geometry is empty after reprojection/buffer.")
        pbar.update(1)

        # Step 6: Clip
        clipped = composite.rio.clip([roi_geom], roi_gdf.crs, drop=drop, all_touched=all_touched)
        pbar.update(1)

    print("Clipping complete. Bands:", list(clipped.data_vars))
    return clipped


def save_roi(
    composite: xr.Dataset,
    filename: str,
    bands: list[str],
    *,
    overwrite: bool = True,
    show_progress: bool = True,
) -> xr.DataArray:
    """
    Save a composite as a multi-band GeoTIFF with progress feedback.
    """
    comp_tif = f"{filename}.tif"
    print(f"Saving composite to: {comp_tif}")

    # Validate bands
    missing = [b for b in bands if b not in composite.data_vars]
    if missing:
        raise ValueError(f"Bands not found in composite: {missing}")

    # --- Stack bands into a single DataArray ---
    tqdm_iter = tqdm(bands, desc="Stacking bands", unit="band") if show_progress else bands
    band_list = [composite[b] for b in tqdm_iter]
    stack = xr.concat(band_list, dim="band")
    stack = stack.assign_coords(band=np.arange(1, len(bands) + 1))
    stack.rio.write_crs(composite.rio.crs, inplace=True)

    # --- Save to GeoTIFF ---
    if show_progress:
        print("Writing GeoTIFF (this may take time for large composites)...")

    stack.rio.to_raster(comp_tif, overwrite=overwrite)

    print(f"Saved composite to: {comp_tif}")
    return stack


def sample_lulc_from_composite(
    gdf_lulc: gpd.GeoDataFrame,
    composite: xr.Dataset,
    comp_tif: str,
    bands: list[str],
    *,
    column: str = "landcover",
    show_progress: bool = True,
    max_pixels_per_poly: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample LULC training pixels from a composite GeoTIFF using polygon GCPs.
    """
    print("Sampling LULC training pixels from GCP polygons...")

    # --- Validate class column ---
    if column not in gdf_lulc.columns:
        raise ValueError(
            f"'{column}' column not found in GCP GeoDataFrame. "
            f"Available columns: {list(gdf_lulc.columns)}"
        )

    # Rename to 'class' for internal consistency
    gdf_lulc = gdf_lulc.rename(columns={column: "class"})

    # Ensure CRS exists and matches composite CRS
    if gdf_lulc.crs is None:
        gdf_lulc = gdf_lulc.set_crs(epsg=4326)
    if not hasattr(composite, "rio") or composite.rio.crs is None:
        raise ValueError("Composite is missing a valid CRS (composite.rio.crs is None).")
    gdf_lulc = gdf_lulc.to_crs(composite.rio.crs)

    # --- Sample pixels under polygons ---
    X_parts, y_parts = [], []
    comp_tif = f'{comp_tif}.tif'
    with rasterio.open(comp_tif) as src:
        nodata_val = src.nodata
        iterator = _maybe_tqdm(
            gdf_lulc.iterrows(),
            total=len(gdf_lulc),
            desc="Sampling polygons",
            enable=show_progress,
        )

        for idx, row in iterator:
            geom = [row.geometry.__geo_interface__]
            try:
                out_img, _ = rio_mask(src, geom, crop=True)
            except Exception as e:
                print(f"Polygon {idx} failed in rio_mask: {e}")
                continue

            # Reshape to (pixels, bands)
            arr = out_img.reshape(out_img.shape[0], -1).T

            # Mask invalid pixels (nodata, NaN, or all zeros)
            if nodata_val is not None:
                mask_valid = ~np.any(arr == nodata_val, axis=1)
            else:
                mask_valid = ~np.any(np.isnan(arr), axis=1)
                mask_valid &= ~np.all(arr == 0, axis=1)

            arr_valid = arr[mask_valid]
            if arr_valid.size == 0:
                continue

            # Optional per-polygon sampling cap
            if max_pixels_per_poly and arr_valid.shape[0] > max_pixels_per_poly:
                sel = np.random.choice(arr_valid.shape[0], size=max_pixels_per_poly, replace=False)
                arr_valid = arr_valid[sel]

            X_parts.append(arr_valid)
            y_parts.extend([int(row["class"])] * arr_valid.shape[0])

    # --- Combine results ---
    if not X_parts:
        print("No valid pixels found.")
        return np.zeros((0, len(bands)), dtype=float), np.array([], dtype=int)

    X_lulc = np.vstack(X_parts)
    y_lulc = np.array(y_parts, dtype=int)

    # --- Summary ---
    labels, counts = np.unique(y_lulc, return_counts=True)
    print(f"LULC samples: {X_lulc.shape}, Labels: {labels.tolist()}, Counts: {counts.tolist()}")
    return X_lulc, y_lulc


def predict_lulc_map(
    model: RandomForestClassifier,
    comp_tif: str,
    out_path: str = "lulc_map",
    *,
    chunk_size: Optional[int] = None,
    show_progress: bool = True,
) -> str:
    """
    Predict a full LULC map from a trained Random Forest model using the composite GeoTIFF.
    """
    print("Predicting full LULC map (this may use significant memory for large composites)...")

    comp_tif = f'{comp_tif}.tif'
    out_path = f'{out_path}.tif'
    with rasterio.open(comp_tif) as src:
        meta = src.meta.copy()
        bands_count = src.count

        out_meta = meta.copy()
        out_meta.update({"count": 1, "dtype": "float32"})
        with rasterio.open(out_path, "w", **out_meta) as dst:
            windows = list(src.block_windows())
            iterator = _maybe_tqdm(windows, total=len(windows), desc="Predicting tiles", unit="block", enable=show_progress)

            for _, win in iterator:
                block = src.read(window=win)  # (bands, h, w)
                h, w = win.height, win.width
                flat = block.reshape(bands_count, -1).T  # (h*w, bands)

                # Valid mask
                if src.nodata is not None:
                    valid = ~np.all(flat == src.nodata, axis=1)
                else:
                    valid = ~np.all(flat == 0, axis=1)
                valid &= ~np.any(np.isnan(flat), axis=1)

                preds = np.full(flat.shape[0], np.nan, dtype=np.float32)
                if valid.any():
                    vidx = np.where(valid)[0]
                    if chunk_size is None:
                        preds[vidx] = model.predict(flat[vidx])
                    else:
                        for start in range(0, vidx.size, chunk_size):
                            sub = vidx[start:start + chunk_size]
                            preds[sub] = model.predict(flat[sub])

                # IMPORTANT: write 2-D array when band index is provided
                dst.write(preds.reshape(h, w), 1, window=win)

    print(f"Saved LULC map to: {out_path}")
    return out_path



# --------------------------- helpers -------------------------------------------
def _is_geographic(crs) -> bool:
    try:
        return bool(getattr(crs, "is_geographic", False))
    except Exception:
        return False


def _extent_from_transform(transform, height: int, width: int) -> Tuple[float, float, float, float]:
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e
    x0, x1 = sorted([left, right])
    y0, y1 = sorted([bottom, top])
    return x0, y0, x1, y1


def _nice_round_length(value_m: float) -> float:
    """Round to a 'nice' number (1, 2, 5 × 10^n)."""
    if value_m <= 0:
        return 0
    power = int(np.floor(np.log10(value_m)))
    base = value_m / (10 ** power)
    for n in (1, 2, 5, 10):
        if base <= n:
            return n * (10 ** power)
    return 10 * (10 ** power)


def _add_scalebar(ax, extent, crs, *, bar_frac=0.25, height_frac=0.01, loc="lower left", pad=0.02):
    x0, y0, x1, y1 = extent
    width_units = x1 - x0
    height_units = y1 - y0
    bar_units = width_units * bar_frac

    is_geo = _is_geographic(crs)
    if is_geo:
        lat_center = (y0 + y1) / 2.0
        meters_per_degree = (
            111132.92 - 559.82 * np.cos(2 * np.radians(lat_center))
            + 1.175 * np.cos(4 * np.radians(lat_center))
            - 0.0023 * np.cos(6 * np.radians(lat_center))
        )
        bar_meters = bar_units * meters_per_degree
        label_units_m = _nice_round_length(bar_meters)
        bar_units = label_units_m / meters_per_degree
    else:
        label_units_m = _nice_round_length(bar_units)

    pad_x = width_units * pad
    pad_y = height_units * pad
    bx0, by0 = (x0 + pad_x, y0 + pad_y) if "left" in loc else (x1 - pad_x - bar_units, y0 + pad_y)

    rect = mpatches.Rectangle(
        (bx0, by0),
        bar_units,
        height_units * height_frac,
        facecolor="white",
        edgecolor="black",
        alpha=0.7,
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(rect)

    label = f"{int(label_units_m)} m" if label_units_m < 1000 else f"{label_units_m/1000:.1f} km"
    ax.text(
        bx0 + bar_units / 2.0,
        by0 + height_units * height_frac * 1.6,
        label,
        ha="center",
        va="bottom",
        color="black",
        fontsize=9,
        zorder=6,
    )


def _add_north_arrow(ax, *, loc="upper left", size=0.08, pad=0.02, color="black"):
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    x0, y0 = (pad, 1 - pad) if "left" in loc else (1 - pad, 1 - pad)
    ha, va = ("left", "top") if "left" in loc else ("right", "top")

    ax.annotate(
        "",
        xy=(x0, y0),
        xytext=(x0, y0 - size),
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=10, headlength=10),
        zorder=7,
    )
    ax.text(x0, y0 + 0.01, "N", transform=trans, ha=ha, va="bottom", color=color, fontsize=10, zorder=7)


# --------------------------- main function -------------------------------------
def plot_lulc_map(
    lulc_tif: str,
    *,
    class_info: Dict[int, Tuple[str, str]],  # {id: (label, color)}
    title: str = "LULC Map",
    legend_loc: str = "lower right",
    figsize: Tuple[float, float] = (7, 7),
    grid: bool = True,
    save_path: Optional[str] = None,
    enable_scalebar: bool = True,
    scalebar_loc: str = "lower left",
    scalebar_frac: float = 0.25,
    enable_north_arrow: bool = True,
    north_arrow_loc: str = "upper left",
):
    """
    Open and plot a LULC GeoTIFF with unified class info, legend, scalebar & north arrow.
    """
    with rasterio.open(lulc_tif) as src:
        lulc_map = src.read(1)
        transform = src.transform
        crs = src.crs
        H, W = lulc_map.shape

    # Extract label & color arrays
    classes = sorted(class_info.keys())
    labels = [class_info[cid][0] for cid in classes]
    colors = [class_info[cid][1] for cid in classes]

    cmap = mcolors.ListedColormap(colors)
    bounds = classes + [classes[-1] + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)

    masked = np.ma.masked_invalid(lulc_map)

    fig, ax = plt.subplots(figsize=figsize)
    rio_show(masked, transform=transform, cmap=cmap, norm=norm, ax=ax)

    # Build legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(classes))]
    ax.legend(
        handles=patches,
        loc=legend_loc,
        fontsize=8,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        title="LULC Classes",
        title_fontsize=9,
    )

    ax.set_xlabel("Longitude (°)" if _is_geographic(crs) else "Easting (m)")
    ax.set_ylabel("Latitude (°)" if _is_geographic(crs) else "Northing (m)")
    if grid:
        ax.grid(True, color="white", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    extent = _extent_from_transform(transform, H, W)
    if enable_scalebar:
        _add_scalebar(ax, extent, crs, bar_frac=scalebar_frac, loc=scalebar_loc)
    if enable_north_arrow:
        _add_north_arrow(ax, loc=north_arrow_loc)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f" Saved figure to: {save_path}")

    return fig, ax


# ---------------------------------------------------------------------
# Helper: sample pixels under polygons
# ---------------------------------------------------------------------
def _sample_from_polygons(
    gdf,
    raster_path: str,
    bands_count: Optional[int] = None,
    *,
    show_progress: bool = True,
    max_pixels_per_poly: Optional[int] = None,
):
    """For each polygon, mask raster and collect (pixels, bands) + class labels."""
    X_parts, y_parts = [], []
    with rasterio.open(raster_path) as src:
        nodata_val = src.nodata
        iterator = _maybe_tqdm(
            gdf.iterrows(),
            total=len(gdf),
            desc="Sampling polygons",
            unit="poly",
            enable=show_progress,
        )

        for idx, row in iterator:
            geom = [row.geometry.__geo_interface__]
            try:
                out_img, _ = rio_mask(src, geom, crop=True)
            except Exception as e:
                print(f"Polygon {idx} failed in rio_mask: {e}")
                continue

            arr = out_img.reshape(out_img.shape[0], -1).T  # (pixels, bands)

            if nodata_val is not None:
                mask_valid = ~np.any(arr == nodata_val, axis=1)
            else:
                mask_valid = ~np.any(np.isnan(arr), axis=1)
                mask_valid &= ~np.all(arr == 0, axis=1)

            arr_valid = arr[mask_valid]
            if arr_valid.size == 0:
                continue

            if max_pixels_per_poly and arr_valid.shape[0] > max_pixels_per_poly:
                sel = np.random.choice(arr_valid.shape[0], size=max_pixels_per_poly, replace=False)
                arr_valid = arr_valid[sel]

            X_parts.append(arr_valid)
            y_parts.extend([int(row["class"])] * arr_valid.shape[0])

    if not X_parts:
        return np.zeros((0, bands_count or 0), dtype=float), np.array([], dtype=int)

    X = np.vstack(X_parts)
    y = np.array(y_parts, dtype=int)
    return X, y


# ---------------------------------------------------------------------
# Main: build mask → write masked composite → sample irrigation classes
# ---------------------------------------------------------------------
def irrigation_process(
    *,
    lulc_tif: str,
    ag_class: int,
    gdf_irrig,
    comp_tif: str,
    bands: list[str],
    out_masked_tif: str = "composite_agri_only.tif",
    class_col: str = "landcover",
    show_progress: bool = True,
    max_pixels_per_poly: Optional[int] = None,
    row_block: int = 2048,
    nodata: Optional[float] = None,
) -> Dict[str, Any]:
    """
    From a LULC GeoTIFF, create an agriculture mask aligned to the composite raster,
    write a masked composite (non-ag -> NoData), and sample irrigated/rainfed pixels
    ONLY inside agriculture using polygon GCPs.
    """
    comp_tif = f'{comp_tif}.tif'
    out_masked_tif = f'{out_masked_tif}.tif'
    
    # --- Read composite metadata (target grid) ---
    with rasterio.open(comp_tif) as comp_src:
        H, W = comp_src.height, comp_src.width
        comp_crs = comp_src.crs
        comp_transform = comp_src.transform
        comp_meta = comp_src.meta.copy()
        if nodata is None:
            if comp_src.nodata is not None:
                out_nodata = comp_src.nodata
            else:
                is_float = np.issubdtype(np.dtype(comp_src.dtypes[0]), np.floating)
                out_nodata = np.nan if is_float else 0
        else:
            out_nodata = nodata

    # --- Read and reproject LULC to composite grid ---
    with rasterio.open(lulc_tif) as lulc_src:
        lulc_aligned = np.empty((H, W), dtype=lulc_src.dtypes[0])
        reproject(
            source=rasterio.band(lulc_src, 1),
            destination=lulc_aligned,
            src_transform=lulc_src.transform,
            src_crs=lulc_src.crs,
            dst_transform=comp_transform,
            dst_crs=comp_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,  # categorical
        )

    # --- Build agriculture mask ---
    agri_mask = (lulc_aligned == ag_class)
    n_ag = int(np.nansum(agri_mask))
    print(f"[AgriMask] Agricultural pixels: {n_ag:,}")
    if n_ag == 0:
        print(f"No agricultural pixels found (class {ag_class}).")

    # --- Write masked composite ---
    print(f"[AgriMask] Writing masked composite to: {out_masked_tif}")
    with rasterio.open(comp_tif) as src, rasterio.open(
        out_masked_tif, "w", **{**comp_meta, "nodata": out_nodata}
    ) as dst:
        windows = list(src.block_windows())
        iterator = _maybe_tqdm(windows, total=len(windows), desc="Writing masked composite", unit="block", enable=show_progress)
        for _, win in iterator:
            block = src.read(window=win)
            m = agri_mask[win.row_off:win.row_off+win.height, win.col_off:win.col_off+win.width]
            if np.isnan(out_nodata):
                block[:, ~m] = np.nan
            else:
                block[:, ~m] = out_nodata
            dst.write(block, window=win)
    print(f"Masked composite written: {out_masked_tif}")

    # --- Sample irrigated/rainfed pixels ---
    if class_col not in gdf_irrig.columns:
        raise ValueError(f"'{class_col}' column not found in irrigation polygons. Available: {list(gdf_irrig.columns)}")
    gdf_local = gdf_irrig.rename(columns={class_col: "class"})
    if gdf_local.crs is None:
        gdf_local = gdf_local.set_crs(epsg=4326)
    gdf_local = gdf_local.to_crs(comp_crs)

    print("[Sample] Sampling irrigated/rainfed pixels (agriculture only)...")
    X_irrig, y_irrig = _sample_from_polygons(
        gdf=gdf_local,
        raster_path=comp_tif,
        bands_count=len(bands),
        show_progress=show_progress,
        max_pixels_per_poly=max_pixels_per_poly,
    )

    if y_irrig.size:
        labels, counts = np.unique(y_irrig, return_counts=True)
        print(f"[Sample] Done. X: {X_irrig.shape}, Labels: {labels.tolist()}, Counts: {counts.tolist()}")
    else:
        print("No irrigated/rainfed samples found (empty result).")

    return {
        "agri_mask": agri_mask,
        "masked_tif": out_masked_tif,
        "X_irrig": X_irrig,
        "y_irrig": y_irrig,
    }


def predict_irrigated_rainfed_map(
    model,
    comp_tif: str,
    agri_mask: np.ndarray,                 # shape (H, W), aligned to comp_tif
    out_path: str = "irrigated_rainfed.tif",
    *,
    chunk_size: Optional[int] = 500_000,   # None => predict all at once
    show_progress: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Predict irrigated vs rainfed map only inside agricultural pixels.
    """
    print("Applying irrigated/rainfed model to agricultural pixels...")
    comp_tif= f'{comp_tif}.tif'
    with rasterio.open(comp_tif) as src:
        meta = src.meta.copy()
        bands = src.count
        H, W = src.height, src.width

        if agri_mask.shape != (H, W):
            raise ValueError(f"`agri_mask` shape {agri_mask.shape} does not match raster {(H, W)}")

        out_meta = meta.copy()
        out_meta.update({"count": 1, "dtype": "float32"})
        irrig_map = np.full((H, W), np.nan, dtype=np.float32)

        windows = list(src.block_windows())
        it = _maybe_tqdm(windows, total=len(windows), desc="Predicting agri blocks", unit="block", enable=show_progress)
        for _, win in it:
            block = src.read(window=win)  # (bands, h, w)
            flat = block.reshape(bands, -1).T

            m = agri_mask[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
            m_flat = m.ravel()

            # Valid mask: exclude nodata/all-zero/NaN
            if src.nodata is not None:
                valid = ~np.all(flat == src.nodata, axis=1)
            else:
                valid = ~np.all(flat == 0, axis=1)
            valid &= ~np.any(np.isnan(flat), axis=1)
            valid_ag = valid & m_flat

            preds = np.full(flat.shape[0], np.nan, dtype=np.float32)
            if valid_ag.any():
                idx = np.where(valid_ag)[0]
                if chunk_size is None:
                    preds[idx] = model.predict(flat[idx])
                else:
                    for start in range(0, idx.size, chunk_size):
                        sub = idx[start : start + chunk_size]
                        preds[sub] = model.predict(flat[sub])

            irrig_map[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width] = preds.reshape(win.height, win.width)

        out_path = f'{out_path}.tif'
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(irrig_map, 1)

    print(f"Saved irrigated/rainfed map to: {out_path}")
    return out_path, irrig_map


def plot_irrigated_rainfed_map(
    irrig_tif: str,
    *,
    class_info: Dict[int, Tuple[str, str]],   # {id: (label, color)}
    title: str = "Irrigated vs Rainfed Agricultural Land",
    legend_loc: str = "lower right",
    figsize: Tuple[float, float] = (7, 7),
    grid: bool = True,
    mask_non_target_as: int = 0,              # remap values not in class_info to this id
):
    """
    Open and plot the irrigated/rainfed GeoTIFF with a legend from a unified class dict.
    """
    with rasterio.open(irrig_tif) as src:
        arr = src.read(1)  # (H, W)
        transform = src.transform
        crs = src.crs

    # Normalize values to known classes: map anything else to "mask_non_target_as"
    known = set(class_info.keys())
    arr_norm = np.where(np.isin(arr, list(known)), arr, mask_non_target_as)

    # Build cmap and boundaries
    ids_sorted = sorted(class_info.keys())
    labels = [class_info[i][0] for i in ids_sorted]
    colors = [class_info[i][1] for i in ids_sorted]

    cmap = mcolors.ListedColormap(colors)
    bounds = ids_sorted + [ids_sorted[-1] + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    rio_show(arr_norm, transform=transform, cmap=cmap, norm=norm, ax=ax)

    # Legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(ids_sorted))]
    ax.legend(
        handles=patches,
        loc=legend_loc,
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        title="Legend",
        title_fontsize=10,
    )

    # Axis labels (CRS-aware)
    is_geo = bool(getattr(crs, "is_geographic", False))
    ax.set_xlabel("Longitude (°)" if is_geo else "Easting (m)")
    ax.set_ylabel("Latitude (°)" if is_geo else "Northing (m)")

    if grid:
        ax.grid(True, color="white", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig, ax


def export_classification_summary(
    *,
    out_dir: str = "Results",
    # model outputs
    y_te,
    y_pred_te,
    yte_i,
    ypred_i,
    # accuracy metrics
    accuracy_lulc: float,
    kappa_lulc: float,
    accuracy_i: float,
    kappa_i: float,
    # key raster paths
    comp_tif: str,
    out_lulc_tif: str,
    out_irrig_tif: str,
    # class mapping for area computation
    class_mapping: dict[int, str],
    pixel_size: float = 10.0,  # meters
) -> str:
    """
    Save confusion matrices, area statistics, and accuracy/Kappa metrics
    for both LULC and irrigated/rainfed classifications.
    """
    comp_tif = f'{comp_tif}.tif'
    out_irrig_tif = f'{out_irrig_tif}.tif'

    # === 1. Confusion matrices ===
    lulc_cm = confusion_matrix(y_te, y_pred_te)
    irrig_cm = confusion_matrix(yte_i, ypred_i)

    lulc_cm_path = os.path.join(out_dir, "lulc_confusion_matrix.csv")
    irrig_cm_path = os.path.join(out_dir, "irrig_confusion_matrix.csv")
    pd.DataFrame(lulc_cm).to_csv(lulc_cm_path, index=False)
    pd.DataFrame(irrig_cm).to_csv(irrig_cm_path, index=False)
    print(f" Confusion matrices saved in: {out_dir}")

    # === 2. Area computation from irrigated/rainfed map ===
    with rasterio.open(out_irrig_tif) as src:
        arr = src.read(1)
        pixel_area = (pixel_size ** 2) / 10_000.0  # ha per pixel
        total_valid = np.isfinite(arr).sum()

        area_data = []
        for class_val, label in class_mapping.items():
            count = int((arr == class_val).sum())
            area_ha = round(count * pixel_area, 2)
            pct = round((count / total_valid * 100) if total_valid > 0 else 0.0, 2)
            area_data.append({"Class": label, "Area_ha": area_ha, "Percent": pct})

    area_df = pd.DataFrame(area_data)

    # === 3. Append metrics ===
    area_df["Acc_LULC"] = round(float(accuracy_lulc), 4)
    area_df["Kappa_LULC"] = round(float(kappa_lulc), 4)
    area_df["Acc_Irrig"] = round(float(accuracy_i), 4)
    area_df["Kappa_Irrig"] = round(float(kappa_i), 4)

    # === 4. Save unified CSV ===
    area_csv_path = os.path.join(out_dir, "area_summary.csv")
    area_df.to_csv(area_csv_path, index=False)

    # === 5. Print summary ===
    print("\  Workflow finished successfully. Outputs:")
    print(f" - Composite (multiband): {comp_tif}")
    print(f" - LULC map: {out_lulc_tif}")
    print(f" - Rainfed map: {out_irrig_tif}")
    print(f" - LULC confusion CSV: {lulc_cm_path}")
    print(f" - Irrigated/Rainfed confusion CSV: {irrig_cm_path}")
    print(f" - Unified Area+Metrics CSV: {area_csv_path}")

    print("\n Area + metrics summary:")
    print(area_df.to_string(index=False))

    return area_csv_path

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
import s3fs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rasterio.enums import Resampling

def compare_map(
    other_map_path_or_url: str,
    aoi_path: str = None,
    iwmi_url_or_path: str = None,           # if None, auto-pick first IWMI COG from S3
    iwmi_bucket_prefix: str = "iwmi-datasets/Cropland_partition/Irrigated_area",
    resampling_for_align: str = "nearest",   # 'nearest' for classes; 'bilinear' for continuous
    class_info_pred: dict = None,            # e.g., {0:("Other","#fff"), 1:("Irrigated","#682215"), 2:("Rainfed","#eaf30c")}
    class_info_iwmi: dict = None,            # e.g., {0:("Irrigated","#682215"), 1:("Rainfed","#eaf30c")}
    title_iwmi: str = "IWMI Classified Map",
    title_pred: str = "Predicted Classified Map",
    sharey: bool = True,
    out_dir: str = "Results",
    out_iwmi_name: str = "iwmi_aligned_clip.tif",
    out_pred_name: str = "pred_aligned_clip.tif"
):
    """
    """

    # ------------- helpers defined inline (kept inside this single function) -------------
    def list_iwmi_urls(prefix):
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "af-south-1"})
        tifs = fs.glob(f"{prefix}/**/*.tif")
        return [f"https://{t.replace('iwmi-datasets/', 'iwmi-datasets.s3.af-south-1.amazonaws.com/')}" for t in tifs]

    def load_da(path_or_url):
        return xr.open_dataarray(path_or_url, engine="rasterio")

    def single_band(r):
        return r.isel(band=0) if "band" in r.dims else r

    def clip_to_aoi(da, aoi):
        gdf = gpd.read_file(aoi)
        gdf = gdf.to_crs(da.rio.crs)
        return da.rio.clip(gdf.geometry, crs=gdf.crs, drop=True, invert=False)

    def align_to(ref, src, resampling="nearest"):
        method = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}.get(resampling, Resampling.nearest)
        aligned = src.rio.reproject_match(ref, resampling=method)
        return aligned.rio.write_crs(ref.rio.crs)

    def to_int_safe(da):
        r = single_band(da)
        v = r.values.copy()
        nan_mask = np.isnan(v)
        v_int = np.where(nan_mask, 0, np.rint(v)).astype(np.int32)
        v_int = np.where(nan_mask, np.nan, v_int)
        out = r.copy(deep=True); out.values = v_int
        out = out.rio.write_crs(r.rio.crs).rio.write_nodata(np.nan)
        return out

    def merge_class_info(info1, info2):
        merged = {}
        if info1:
            merged.update(info1)
        if info2:
            merged.update(info2)
        return merged

    def build_cmap_norm(class_info):
        keys = sorted(class_info.keys())
        labels = [class_info[k][0] for k in keys]
        colors = [class_info[k][1] for k in keys]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        bounds = np.array(keys + [keys[-1] + 1]) - 0.5
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = [mpatches.Patch(color=c, label=l) for l, c in zip(labels, colors)]
        return cmap, norm, patches

    # ------------- load IWMI & OTHER -------------
    if iwmi_url_or_path is None:
        urls = list_iwmi_urls(iwmi_bucket_prefix)
        if not urls:
            raise RuntimeError("No IWMI GeoTIFFs found in the S3 prefix. Provide iwmi_url_or_path explicitly.")
        iwmi_url_or_path = urls[0]

    iwmi = load_da(iwmi_url_or_path)
    pred = load_da(other_map_path_or_url)

    # ------------- optional AOI clip -------------
    if aoi_path:
        iwmi = clip_to_aoi(iwmi, aoi_path)
        pred = clip_to_aoi(pred, aoi_path)

    # ------------- align to same grid/units -------------
    # Choose the predicted map as reference grid (you can invert if needed)
    iwmi_aligned = align_to(pred, iwmi, resampling=resampling_for_align)
    pred_aligned = single_band(pred)  # ensure single-band for plotting & int conversion

    # ------------- convert to safe integers (keep NaNs) -------------
    iwmi_int = to_int_safe(iwmi_aligned)
    pred_int = to_int_safe(pred_aligned)

    # ------------- build ONE merged legend -------------
    merged_info = merge_class_info(class_info_iwmi or {}, class_info_pred or {})
    if not merged_info:
        # default fallback if none provided
        merged_info = {0: ("Class 0", "#cccccc"), 1: ("Class 1", "#682215"), 2: ("Class 2", "#eaf30c")}
    cmap, norm, patches = build_cmap_norm(merged_info)

    # Build filtered, deduplicated legend (only Irrigated + Rainfed)
    desired_labels = {"Irrigated Agriculture", "Rainfed Agriculture"}
    
    # Keep only these from the merged patches
    filtered_patches = []
    seen = set()
    for p in patches:
        label = p.get_label()
        if label in desired_labels and label not in seen:
            filtered_patches.append(p)
            seen.add(label)

    # ------------- plot side-by-side with one legend -------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=sharey)
    im1 = iwmi_int.plot(ax=axes[0], cmap=cmap, norm=norm, add_colorbar=False)
    im2 = pred_int.plot(ax=axes[1], cmap=cmap, norm=norm, add_colorbar=False)
    axes[0].set_title(title_iwmi, fontsize=12)
    axes[1].set_title(title_pred, fontsize=12)
    fig.legend(handles=filtered_patches, loc='lower center', ncol=min(len(patches), 5),
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/compare_map.png')
    plt.show()

    return {
        "iwmi_int": iwmi_int,
        "pred_int": pred_int,
    }





















