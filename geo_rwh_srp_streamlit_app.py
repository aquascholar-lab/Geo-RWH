# ================================================================
# Geo-RWH-SRP+ : Smart GIS Web Portal for Rainwater Harvesting
# and Spring Rejuvenation Planning
# ================================================================
# Major functions:
#   1. Upload 10 classified GeoTIFF suitability factor layers
#   2. Apply AHP default weights or user-adjusted weights
#   3. Perform weighted overlay
#   4. Classify final suitability into 4 classes
#   5. Classification methods: Equal interval, Quantile, Jenks, Manual
#   6. Interactive map with dark legend, north arrow, scale and spring CSV points
#   7. Area statistics in sq. km and percentage
#   8. ROC-AUC validation using spring/non-spring point shapefile ZIP
# ================================================================
# Run:
#   conda create -n geo_rwh python=3.10 -y
#   conda activate geo_rwh
#   pip install -r requirements_geo_rwh_srp.txt
#   streamlit run geo_rwh_srp_streamlit_app.py
# ================================================================

import os
import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import rowcol
import geopandas as gpd
import folium
from folium.plugins import Fullscreen, MeasureControl, MousePosition, MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from PIL import Image
from pyproj import Transformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Geo-RWH-SRP+ Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# App constants
# -----------------------------
FACTOR_SPECS = [
    {"key": "cn", "label": "Curve Number (CN)", "hint": "cn.tif"},
    {"key": "rain_monsoon", "label": "Rainfall", "hint": "rain_monsoon.tif"},
    {"key": "geology", "label": "Geology", "hint": "geology.tif"},
    {"key": "slope", "label": "Slope", "hint": "slope.tif"},
    {"key": "twi", "label": "Topographic Wetness Index (TWI)", "hint": "twi.tif"},
    {"key": "ddensity", "label": "Drainage Density", "hint": "ddensity.tif"},
    {"key": "d_str", "label": "Stream Distance", "hint": "d_str.tif"},
    {"key": "soil", "label": "Soil", "hint": "soil.tif"},
    {"key": "lulc", "label": "LULC", "hint": "lulc.tif"},
    {"key": "lineaments", "label": "Lineament", "hint": "lineaments.tif"},
]

# Default AHP weights taken from the provided table/image. Values are in percent.
DEFAULT_WEIGHTS = {
    "cn": 29.04,
    "rain_monsoon": 20.99,
    "geology": 15.08,
    "slope": 10.78,
    "twi": 7.62,
    "ddensity": 5.27,
    "d_str": 3.53,
    "soil": 3.53,
    "lulc": 2.41,
    "lineaments": 1.75,
}

# Final map has four classes only.
CLASS_INFO = {
    0: {"name": "NoData", "color": "#00000000"},
    1: {"name": "Unsuitable", "color": "#d73027"},
    2: {"name": "Moderately Suitable", "color": "#fee08b"},
    3: {"name": "Highly Suitable", "color": "#91cf60"},
    4: {"name": "Very Highly Suitable", "color": "#1a9850"},
}

# RGBA values for PNG overlay: index 0 transparent, 1-4 classes.
CLASS_RGBA = np.array([
    [0, 0, 0, 0],          # 0 NoData
    [215, 48, 39, 215],    # 1 Unsuitable
    [254, 224, 139, 215],  # 2 Moderate
    [145, 207, 96, 215],   # 3 High
    [26, 152, 80, 215],    # 4 Very High
], dtype=np.uint8)


# -----------------------------
# Helper functions
# -----------------------------
def save_uploaded_file(uploaded_file, out_dir: str, name_prefix: str = "") -> str:
    """Save a Streamlit UploadedFile to a temporary path and return path."""
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{name_prefix}{Path(uploaded_file.name).stem}{suffix}".replace(" ", "_")
    out_path = os.path.join(out_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def get_reference_profile(ref_path: str):
    """Read reference raster metadata."""
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "bounds": src.bounds,
            "profile": profile,
        }
    return meta


def read_and_align_raster(
    src_path: str,
    ref_meta: dict,
    valid_min: float = 1,
    valid_max: float = 5,
    clean_to_1_5: bool = True,
    invert_scale: bool = False,
):
    """
    Read a classified raster and align it to the reference raster.
    Classified rasters are expected to use values 1-5, where 5 = highly suitable.
    Values outside 1-5 can be treated as NoData.
    """
    dst_shape = (ref_meta["height"], ref_meta["width"])
    dst = np.full(dst_shape, np.nan, dtype="float32")

    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32")
        src_nodata = src.nodata

        same_grid = (
            src.crs == ref_meta["crs"]
            and src.transform.almost_equals(ref_meta["transform"])
            and src.height == ref_meta["height"]
            and src.width == ref_meta["width"]
        )

        if same_grid:
            dst = src_arr
            if src_nodata is not None:
                dst = np.where(dst == src_nodata, np.nan, dst)
        else:
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )

    if clean_to_1_5:
        dst = np.where((dst >= valid_min) & (dst <= valid_max), dst, np.nan)

    if invert_scale:
        valid = np.isfinite(dst)
        dst[valid] = (valid_min + valid_max) - dst[valid]

    return dst.astype("float32")


def normalize_weights(weight_dict: dict) -> dict:
    total = float(sum(weight_dict.values()))
    if total <= 0:
        raise ValueError("At least one factor weight must be greater than zero.")
    return {k: float(v) / total for k, v in weight_dict.items()}


def weighted_overlay(arrays: dict, norm_weights: dict, missing_policy: str = "Strict: use only pixels valid in all layers"):
    """Perform weighted overlay on 1-5 classified layers."""
    keys = list(arrays.keys())
    shape = arrays[keys[0]].shape
    numerator = np.zeros(shape, dtype="float32")
    denominator = np.zeros(shape, dtype="float32")

    if missing_policy.startswith("Strict"):
        valid_all = np.ones(shape, dtype=bool)
        for k in keys:
            valid_all &= np.isfinite(arrays[k])
            numerator += np.where(np.isfinite(arrays[k]), arrays[k], 0) * norm_weights[k]
        score = numerator.copy()
        score[~valid_all] = np.nan
    else:
        for k in keys:
            valid = np.isfinite(arrays[k])
            numerator += np.where(valid, arrays[k], 0) * norm_weights[k]
            denominator += np.where(valid, norm_weights[k], 0)
        score = np.divide(numerator, denominator, out=np.full(shape, np.nan, dtype="float32"), where=denominator > 0)

    return score.astype("float32")


def _custom_jenks_breaks(values: np.ndarray, n_classes: int = 4, sample_size: int = 5000):
    """
    Fisher-Jenks natural breaks fallback implementation.
    For speed, large rasters are sampled reproducibly before break estimation.
    Returns n_classes + 1 break values, including minimum and maximum.
    """
    data = np.asarray(values, dtype="float64")
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("No valid values available for Jenks classification.")

    if data.size > sample_size:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=sample_size, replace=False)

    data = np.sort(data)
    unique_vals = np.unique(data)
    if unique_vals.size <= n_classes:
        return np.linspace(float(data.min()), float(data.max()), n_classes + 1)

    n_data = len(data)
    lower_class_limits = np.zeros((n_data + 1, n_classes + 1), dtype=np.int32)
    variance_combinations = np.full((n_data + 1, n_classes + 1), np.inf, dtype="float64")

    for i in range(1, n_classes + 1):
        lower_class_limits[1, i] = 1
        variance_combinations[1, i] = 0

    for l in range(2, n_data + 1):
        sum_values = 0.0
        sum_squares = 0.0
        weight = 0.0
        variance = 0.0

        for m in range(1, l + 1):
            lower_class_limit = l - m + 1
            val = data[lower_class_limit - 1]
            weight += 1
            sum_values += val
            sum_squares += val * val
            variance = sum_squares - (sum_values * sum_values) / weight
            previous_class_limit = lower_class_limit - 1

            if previous_class_limit != 0:
                for j in range(2, n_classes + 1):
                    test_variance = variance + variance_combinations[previous_class_limit, j - 1]
                    if variance_combinations[l, j] >= test_variance:
                        lower_class_limits[l, j] = lower_class_limit
                        variance_combinations[l, j] = test_variance

        lower_class_limits[l, 1] = 1
        variance_combinations[l, 1] = variance

    breaks = np.zeros(n_classes + 1, dtype="float64")
    breaks[n_classes] = data[-1]
    breaks[0] = data[0]

    k = n_data
    count_num = n_classes
    while count_num >= 2:
        idx = int(lower_class_limits[k, count_num] - 2)
        breaks[count_num - 1] = data[max(idx, 0)]
        k = int(lower_class_limits[k, count_num] - 1)
        count_num -= 1

    return breaks


def _sample_for_classification(valid_values: np.ndarray, sample_size: int = 50000) -> np.ndarray:
    """Return a representative sample for slow classification routines such as Jenks."""
    valid_values = np.asarray(valid_values, dtype="float64")
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size <= sample_size:
        return valid_values

    # Random sample + fixed quantiles gives fast and stable breaks for large rasters.
    rng = np.random.default_rng(42)
    random_part = rng.choice(valid_values, size=sample_size, replace=False)
    quantile_part = np.nanpercentile(valid_values, np.linspace(0, 100, 101))
    sample = np.concatenate([random_part, quantile_part, [np.nanmin(valid_values), np.nanmax(valid_values)]])
    return sample[np.isfinite(sample)]


def jenks_thresholds(values: np.ndarray, n_classes: int = 4, sample_size: int = 50000):
    """Return internal Jenks thresholds for n_classes.

    Important: Jenks can become very slow on full-resolution rasters with millions
    of pixels. Therefore, this function always estimates Jenks breaks from a
    representative sample. This prevents the Streamlit page from freezing.
    """
    valid_values = values[np.isfinite(values)].astype("float64")
    if valid_values.size == 0:
        raise ValueError("No valid values available for Jenks classification.")

    sampled_values = _sample_for_classification(valid_values, sample_size=sample_size)

    # Prefer jenkspy if installed, but run it only on the sample.
    try:
        import jenkspy
        try:
            breaks = np.array(jenkspy.jenks_breaks(sampled_values, n_classes=n_classes), dtype="float64")
        except TypeError:
            # Some jenkspy versions use nb_class instead of n_classes.
            breaks = np.array(jenkspy.jenks_breaks(sampled_values, nb_class=n_classes), dtype="float64")
    except Exception:
        # Fallback is intentionally smaller because it is O(n²).
        breaks = _custom_jenks_breaks(sampled_values, n_classes=n_classes, sample_size=5000)

    internal = np.array(breaks[1:-1], dtype="float64")
    if len(internal) != n_classes - 1 or len(np.unique(np.round(internal, 8))) != n_classes - 1:
        # Duplicate thresholds may occur if the raster has too few unique values.
        internal = np.nanpercentile(valid_values, [25, 50, 75]).astype("float64")
    return internal


def calculate_thresholds(score: np.ndarray, method: str):
    """Calculate three thresholds for four suitability classes."""
    valid = score[np.isfinite(score)]
    if valid.size == 0:
        raise ValueError("No valid pixels available for classification.")

    if method == "Equal interval":
        min_v = float(np.nanmin(valid))
        max_v = float(np.nanmax(valid))
        thresholds = np.linspace(min_v, max_v, 5)[1:-1]
    elif method == "Quantile":
        thresholds = np.nanpercentile(valid, [25, 50, 75])
    elif method == "Jenks natural breaks":
        thresholds = jenks_thresholds(score, n_classes=4)
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    thresholds = np.array(thresholds, dtype="float64")
    if len(np.unique(np.round(thresholds, 8))) < 3:
        thresholds = np.linspace(float(np.nanmin(valid)), float(np.nanmax(valid)), 5)[1:-1]
    return thresholds


def classify_suitability(score: np.ndarray, thresholds):
    """Classify continuous suitability score into four classes using three thresholds."""
    valid = np.isfinite(score)
    cls = np.zeros(score.shape, dtype="uint8")
    thresholds = np.array(thresholds, dtype="float64")

    if len(thresholds) != 3:
        raise ValueError("Four-class classification requires exactly three threshold values.")
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError("Manual thresholds must be in increasing order: Break 1 < Break 2 < Break 3.")

    cls[valid] = np.digitize(score[valid], bins=thresholds, right=True).astype("uint8") + 1
    return cls


def write_geotiff(out_path: str, arr: np.ndarray, ref_meta: dict, dtype: str, nodata):
    profile = ref_meta["profile"].copy()
    profile.update(
        driver="GTiff",
        height=ref_meta["height"],
        width=ref_meta["width"],
        count=1,
        dtype=dtype,
        crs=ref_meta["crs"],
        transform=ref_meta["transform"],
        nodata=nodata,
        compress="lzw",
    )

    if np.issubdtype(np.dtype(dtype), np.floating):
        out_arr = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    else:
        out_arr = np.asarray(arr).astype(dtype)
        out_arr = np.where(np.isfinite(out_arr), out_arr, nodata).astype(dtype)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)


def create_class_png(class_array: np.ndarray, out_png: str):
    """Create transparent PNG overlay from classified array."""
    safe_cls = np.nan_to_num(class_array, nan=0).astype("uint8")
    safe_cls = np.clip(safe_cls, 0, 4)
    rgba = CLASS_RGBA[safe_cls]
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_png)


def add_legend(map_obj):
    legend_items = "".join(
        [
            f"""
            <div style='display:flex;align-items:center;margin-bottom:5px;color:#111111;font-weight:600;'>
                <span style='background:{CLASS_INFO[i]['color']};width:18px;height:18px;display:inline-block;border:1px solid #333;margin-right:8px;'></span>
                <span>{i}. {CLASS_INFO[i]['name']}</span>
            </div>
            """
            for i in [4, 3, 2, 1]
        ]
    )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 32px;
        left: 32px;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        color: #111111;
        padding: 12px 14px;
        border: 2px solid #333333;
        border-radius: 8px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
    ">
        <div style="font-weight:800;color:#111111;margin-bottom:7px;">RWH/Spring Rejuvenation Suitability</div>
        {legend_items}
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def add_north_arrow(map_obj):
    north_html = """
    <div style="
        position: fixed;
        top: 95px;
        right: 28px;
        z-index: 9999;
        background: rgba(255,255,255,0.94);
        color: #111111;
        padding: 8px 10px;
        border: 2px solid #333;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
        N<br>
        <span style="font-size: 30px; line-height: 20px;color:#111111;">▲</span>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(north_html))


def raster_bounds_wgs84(ref_meta: dict):
    bounds_wgs84 = transform_bounds(
        ref_meta["crs"],
        "EPSG:4326",
        ref_meta["bounds"].left,
        ref_meta["bounds"].bottom,
        ref_meta["bounds"].right,
        ref_meta["bounds"].top,
        densify_pts=21,
    )
    west, south, east, north = bounds_wgs84
    return west, south, east, north


def make_folium_map(class_png: str, ref_meta: dict, springs_df: pd.DataFrame | None = None, label_col: str | None = None):
    """Build interactive folium map with overlay, legend, north arrow, scale and optional spring CSV points."""
    west, south, east, north = raster_bounds_wgs84(ref_meta)
    center = [(south + north) / 2, (west + east) / 2]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)

    folium.raster_layers.ImageOverlay(
        image=class_png,
        bounds=[[south, west], [north, east]],
        opacity=0.78,
        name="Suitability classes",
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(m)

    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#222222",
        weight=1,
        fill=False,
        tooltip="Analysis extent",
    ).add_to(m)

    if springs_df is not None and not springs_df.empty:
        cluster = MarkerCluster(name="Spring locations from CSV", overlay=True, control=True).add_to(m)
        for idx, row in springs_df.iterrows():
            lat = row.get("_map_lat")
            lon = row.get("_map_lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            label = f"Spring {idx + 1}"
            if label_col and label_col in springs_df.columns and pd.notna(row[label_col]):
                label = str(row[label_col])
            popup_lines = [f"<b>{label}</b>"]
            for col in springs_df.columns:
                if col.startswith("_map_"):
                    continue
                value = row[col]
                if pd.notna(value):
                    popup_lines.append(f"<b>{col}</b>: {value}")
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=5,
                color="#0033cc",
                weight=2,
                fill=True,
                fill_color="#00ccff",
                fill_opacity=0.9,
                popup=folium.Popup("<br>".join(popup_lines), max_width=300),
                tooltip=label,
            ).add_to(cluster)

    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon:").add_to(m)
    add_legend(m)
    add_north_arrow(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_area_chart(class_array: np.ndarray, pixel_area_m2: float):
    rows = []
    for i in [1, 2, 3, 4]:
        count = int(np.sum(class_array == i))
        area_km2 = count * pixel_area_m2 / 1_000_000
        rows.append({"Class": i, "Suitability": CLASS_INFO[i]["name"], "Pixels": count, "Area_sq_km": area_km2})
    df = pd.DataFrame(rows)
    total_area = df["Area_sq_km"].sum()
    if total_area > 0:
        df["Area_percent"] = df["Area_sq_km"] / total_area * 100
    else:
        df["Area_percent"] = 0.0
    df["Area_sq_km"] = df["Area_sq_km"].round(3)
    df["Area_percent"] = df["Area_percent"].round(2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["Suitability"], df["Area_sq_km"])
    ax.set_ylabel("Area (sq. km)")
    ax.set_xlabel("Suitability class")
    ax.set_title("Area under each suitability class")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return df, fig


def extract_zip_to_temp(uploaded_zip, out_dir: str) -> str:
    zip_path = save_uploaded_file(uploaded_zip, out_dir, name_prefix="validation_")
    extract_dir = os.path.join(out_dir, "validation_points")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    shp_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.lower().endswith(".shp")]
    if not shp_files:
        shp_files = [str(p) for p in Path(extract_dir).rglob("*.shp")]
    if not shp_files:
        raise ValueError("No .shp file found in the uploaded ZIP.")
    return shp_files[0]


def sample_scores_at_points(gdf: gpd.GeoDataFrame, score: np.ndarray, class_array: np.ndarray, ref_meta: dict):
    if gdf.crs is None:
        raise ValueError("Validation shapefile has no CRS. Please define its projection before uploading.")

    gdf_r = gdf.to_crs(ref_meta["crs"])
    xs = gdf_r.geometry.x.values
    ys = gdf_r.geometry.y.values

    sampled_score = []
    sampled_class = []
    inside = []

    for x, y in zip(xs, ys):
        try:
            r, c = rowcol(ref_meta["transform"], x, y)
            ok = 0 <= r < ref_meta["height"] and 0 <= c < ref_meta["width"]
            if ok and np.isfinite(score[r, c]):
                sampled_score.append(float(score[r, c]))
                sampled_class.append(int(class_array[r, c]))
                inside.append(True)
            else:
                sampled_score.append(np.nan)
                sampled_class.append(0)
                inside.append(False)
        except Exception:
            sampled_score.append(np.nan)
            sampled_class.append(0)
            inside.append(False)

    out = gdf.copy()
    out["suit_score"] = sampled_score
    out["suit_class"] = sampled_class
    out["inside_valid_raster"] = inside
    return out


def create_roc_plot(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC curve, AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random model")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC validation using spring/non-spring points")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return auc, fpr, tpr, thresholds, fig


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=300, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


def auto_detect_column(columns, candidates):
    lower_map = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in columns:
        c_low = str(c).lower()
        for cand in candidates:
            if cand.lower() in c_low:
                return c
    return columns[0] if len(columns) else None


def prepare_spring_csv_for_map(df: pd.DataFrame, x_col: str, y_col: str, coord_mode: str, ref_meta: dict):
    out = df.copy()
    xs = pd.to_numeric(out[x_col], errors="coerce")
    ys = pd.to_numeric(out[y_col], errors="coerce")

    if coord_mode == "Same CRS as raster":
        transformer = Transformer.from_crs(ref_meta["crs"], "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xs.values, ys.values)
    else:
        lon, lat = xs.values, ys.values

    out["_map_lon"] = lon
    out["_map_lat"] = lat
    out = out[np.isfinite(out["_map_lon"]) & np.isfinite(out["_map_lat"])].copy()
    out = out[(out["_map_lat"].between(-90, 90)) & (out["_map_lon"].between(-180, 180))].copy()
    return out


# -----------------------------
# Header
# -----------------------------
st.title("💧 Geo-RWH-SRP+ Dashboard")
st.markdown(
    """
    **A Smart GIS Web Portal for Rainwater Harvesting and Spring Rejuvenation Planning**  
    Upload classified factor rasters, use AHP/default weights, run weighted overlay, classify suitability, display spring locations, and validate using spring/non-spring point data.
    """
)

with st.expander("Important input assumption", expanded=False):
    st.info(
        "Each uploaded factor GeoTIFF should be a reclassified suitability raster with values from 1 to 5, "
        "where 1 = least suitable and 5 = most suitable. Values outside 1-5, including 65535, are treated as NoData when the cleaning option is enabled."
    )


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("1. Upload factor rasters")
st.sidebar.caption("Upload all 10 classified GeoTIFF layers.")

work_dir = tempfile.mkdtemp(prefix="geo_rwh_srp_")

uploaded_paths = {}
weights = {}
invert_flags = {}

for spec in FACTOR_SPECS:
    st.sidebar.markdown(f"**{spec['label']}**")
    up = st.sidebar.file_uploader(
        f"Upload {spec['hint']}",
        type=["tif", "tiff"],
        key=f"upload_{spec['key']}",
        label_visibility="collapsed",
    )
    if up is not None:
        uploaded_paths[spec["key"]] = save_uploaded_file(up, work_dir, name_prefix=f"{spec['key']}_")

st.sidebar.divider()
st.sidebar.header("2. Factor weights")
st.sidebar.caption("Default AHP weights are from the uploaded table. Values are normalized automatically.")

for spec in FACTOR_SPECS:
    weights[spec["key"]] = st.sidebar.slider(
        spec["label"],
        min_value=0.00,
        max_value=100.00,
        value=float(DEFAULT_WEIGHTS[spec["key"]]),
        step=0.01,
        format="%.2f",
        key=f"wt_{spec['key']}",
    )

weight_total = sum(weights.values())
st.sidebar.caption(f"Current raw weight total: **{weight_total:.2f}**")

with st.sidebar.expander("Advanced raster options", expanded=False):
    clean_to_1_5 = st.checkbox("Treat all values outside 1-5 as NoData", value=True)
    missing_policy = st.radio(
        "NoData handling in overlay",
        [
            "Strict: use only pixels valid in all layers",
            "Flexible: ignore missing layers and rescale weights per pixel",
        ],
        index=0,
    )
    st.markdown("**Invert 1-5 suitability scale if any input layer is reversed**")
    for spec in FACTOR_SPECS:
        invert_flags[spec["key"]] = st.checkbox(f"Invert {spec['label']}", value=False, key=f"inv_{spec['key']}")

run_button = st.sidebar.button("🚀 Run Weighted Overlay", type="primary", use_container_width=True)


# -----------------------------
# Main workflow
# -----------------------------
all_uploaded = len(uploaded_paths) == len(FACTOR_SPECS)

if not all_uploaded:
    missing = [s["hint"] for s in FACTOR_SPECS if s["key"] not in uploaded_paths]
    st.warning("Please upload all 10 classified GeoTIFF layers to run the dashboard.")
    st.write("Missing layers:", ", ".join(missing))
    st.stop()

if sum(weights.values()) <= 0:
    st.error("Please assign at least one non-zero weight.")
    st.stop()

if run_button or all_uploaded:
    try:
        with st.spinner("Reading, aligning and overlaying rasters..."):
            # Use first factor as reference raster.
            ref_key = FACTOR_SPECS[0]["key"]
            ref_meta = get_reference_profile(uploaded_paths[ref_key])

            arrays = {}
            for spec in FACTOR_SPECS:
                arrays[spec["key"]] = read_and_align_raster(
                    uploaded_paths[spec["key"]],
                    ref_meta,
                    valid_min=1,
                    valid_max=5,
                    clean_to_1_5=clean_to_1_5,
                    invert_scale=invert_flags[spec["key"]],
                )

            norm_w = normalize_weights(weights)
            score = weighted_overlay(arrays, norm_w, missing_policy=missing_policy)

        st.success("Weighted overlay completed successfully.")

        c1, c2, c3, c4 = st.columns(4)
        valid_pixels = int(np.sum(np.isfinite(score)))
        pixel_area_m2 = abs(ref_meta["transform"].a * ref_meta["transform"].e)
        valid_area_km2 = valid_pixels * pixel_area_m2 / 1_000_000
        c1.metric("Valid analysis pixels", f"{valid_pixels:,}")
        c2.metric("Valid area", f"{valid_area_km2:,.2f} sq. km")
        c3.metric("Mean score", f"{np.nanmean(score):.2f}")
        c4.metric("Score range", f"{np.nanmin(score):.2f}–{np.nanmax(score):.2f}")

        st.subheader("Normalized weights used in overlay")
        wdf = pd.DataFrame(
            [
                {
                    "Factor": next(s["label"] for s in FACTOR_SPECS if s["key"] == k),
                    "Raw weight (%)": round(weights[k], 2),
                    "Normalized weight": round(v, 4),
                    "Percent contribution": round(v * 100, 2),
                }
                for k, v in norm_w.items()
            ]
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        # -----------------------------
        # Threshold and classification controls
        # -----------------------------
        st.subheader("Classification threshold setting")
        class_method = st.radio(
            "Select class threshold method",
            ["Equal interval", "Quantile", "Jenks natural breaks", "Manual edit"],
            horizontal=True,
            index=2,
            help="Jenks natural breaks is selected by default. Manual edit allows you to define all three breaks for the four-class map.",
        )

        score_min = float(np.nanmin(score))
        score_max = float(np.nanmax(score))

        if class_method == "Manual edit":
            with st.spinner("Preparing default threshold values..."):
                default_manual = calculate_thresholds(score, "Jenks natural breaks")
            st.caption("Enter three increasing threshold values. The four classes will be: ≤B1, B1–B2, B2–B3, and >B3.")
            bcol1, bcol2, bcol3 = st.columns(3)
            b1 = bcol1.number_input(
                "Break 1: Unsuitable upper limit",
                min_value=score_min,
                max_value=score_max,
                value=float(default_manual[0]),
                step=0.01,
                format="%.2f",
            )
            b2 = bcol2.number_input(
                "Break 2: Moderate upper limit",
                min_value=score_min,
                max_value=score_max,
                value=float(default_manual[1]),
                step=0.01,
                format="%.2f",
            )
            b3 = bcol3.number_input(
                "Break 3: High upper limit",
                min_value=score_min,
                max_value=score_max,
                value=float(default_manual[2]),
                step=0.01,
                format="%.2f",
            )
            thresholds = np.array([b1, b2, b3], dtype="float64")
        else:
            with st.spinner(f"Calculating {class_method} thresholds..."):
                thresholds = calculate_thresholds(score, class_method)

        if not (thresholds[0] < thresholds[1] < thresholds[2]):
            st.error("Thresholds must be in increasing order: Break 1 < Break 2 < Break 3.")
            st.stop()

        st.caption(
            f"Thresholds used: {thresholds[0]:.2f}, {thresholds[1]:.2f}, {thresholds[2]:.2f}"
        )

        class_array = classify_suitability(score, thresholds)

        out_score_tif = os.path.join(work_dir, "Geo_RWH_SRP_weighted_suitability_score.tif")
        out_class_tif = os.path.join(work_dir, "Geo_RWH_SRP_suitability_classes_4class.tif")
        out_class_png = os.path.join(work_dir, "Geo_RWH_SRP_suitability_overlay_4class.png")

        write_geotiff(out_score_tif, score, ref_meta, dtype="float32", nodata=-9999)
        write_geotiff(out_class_tif, class_array, ref_meta, dtype="uint8", nodata=0)
        create_class_png(class_array, out_class_png)

        th_df = pd.DataFrame(
            {
                "Class": [1, 2, 3, 4],
                "Suitability": [CLASS_INFO[i]["name"] for i in [1, 2, 3, 4]],
                "Score interval": [
                    f"≤ {thresholds[0]:.2f}",
                    f"> {thresholds[0]:.2f} to ≤ {thresholds[1]:.2f}",
                    f"> {thresholds[1]:.2f} to ≤ {thresholds[2]:.2f}",
                    f"> {thresholds[2]:.2f}",
                ],
            }
        )
        st.dataframe(th_df, use_container_width=True, hide_index=True)

        # -----------------------------
        # Map and area statistics
        # -----------------------------
        tab_map, tab_stats, tab_validation, tab_download = st.tabs(
            ["🗺️ Final Suitability Map", "📊 Area Statistics", "✅ ROC/AUC Validation", "⬇️ Downloads"]
        )

        with tab_map:
            st.subheader("Final RWH and Spring Rejuvenation Suitability Map")
            st.markdown("Upload a spring-location CSV if you want to display spring points on the final map.")
            springs_df_for_map = None
            label_col = None

            springs_csv = st.file_uploader(
                "Upload springs location CSV for map display",
                type=["csv"],
                key="springs_location_csv",
                help="CSV should contain coordinate columns such as latitude/longitude or x/y.",
            )
            if springs_csv is not None:
                try:
                    spring_df = pd.read_csv(springs_csv)
                    if spring_df.empty:
                        st.warning("Uploaded spring CSV is empty.")
                    else:
                        numeric_cols = list(spring_df.columns)
                        default_x = auto_detect_column(numeric_cols, ["longitude", "lon", "long", "x", "easting"])
                        default_y = auto_detect_column(numeric_cols, ["latitude", "lat", "y", "northing"])
                        map_cols = st.columns(4)
                        x_col = map_cols[0].selectbox("Longitude/X column", numeric_cols, index=numeric_cols.index(default_x))
                        y_col = map_cols[1].selectbox("Latitude/Y column", numeric_cols, index=numeric_cols.index(default_y))
                        coord_mode = map_cols[2].selectbox(
                            "CSV coordinate system",
                            ["Latitude/Longitude (EPSG:4326)", "Same CRS as raster"],
                            index=0,
                        )
                        label_options = ["None"] + numeric_cols
                        label_guess = auto_detect_column(numeric_cols, ["name", "id", "spring", "site"])
                        label_index = label_options.index(label_guess) if label_guess in label_options else 0
                        label_selected = map_cols[3].selectbox("Point label column", label_options, index=label_index)
                        label_col = None if label_selected == "None" else label_selected

                        springs_df_for_map = prepare_spring_csv_for_map(spring_df, x_col, y_col, coord_mode, ref_meta)
                        st.success(f"{len(springs_df_for_map):,} spring location point(s) will be shown on the map.")
                        with st.expander("Preview spring CSV points", expanded=False):
                            st.dataframe(springs_df_for_map.head(20), use_container_width=True)
                except Exception as e:
                    st.warning("Spring CSV could not be displayed on the map. Please check coordinate columns.")
                    st.exception(e)

            m = make_folium_map(out_class_png, ref_meta, springs_df=springs_df_for_map, label_col=label_col)
            st_folium(m, width=None, height=680)
            st.caption("The map includes four suitability classes, dark legend font, north arrow, scale bar, coordinate display, measuring tool and optional spring CSV points.")

        with tab_stats:
            st.subheader("Area statistics")
            area_df, area_fig = plot_area_chart(class_array, pixel_area_m2)
            st.dataframe(area_df, use_container_width=True, hide_index=True)
            st.pyplot(area_fig, clear_figure=False)

        with tab_validation:
            st.subheader("Validate with spring and non-spring points")
            st.markdown(
                "Upload a ZIP containing `.shp`, `.shx`, `.dbf`, and `.prj`. "
                "The shapefile should contain both spring and non-spring points in one attribute field, such as `VALUE` with 1 = spring and 0 = non-spring."
            )
            val_zip = st.file_uploader("Upload validation point shapefile as ZIP", type=["zip"], key="validation_zip")

            if val_zip is not None:
                try:
                    shp_path = extract_zip_to_temp(val_zip, work_dir)
                    gdf = gpd.read_file(shp_path)

                    if gdf.empty:
                        st.error("The validation shapefile is empty.")
                    else:
                        non_geom_cols = [c for c in gdf.columns if c != "geometry"]
                        default_idx = non_geom_cols.index("VALUE") if "VALUE" in non_geom_cols else 0
                        label_field = st.selectbox("Select spring/non-spring label field", non_geom_cols, index=default_idx)

                        unique_vals = sorted([str(v) for v in gdf[label_field].dropna().unique()])
                        default_pos_idx = unique_vals.index("1") if "1" in unique_vals else len(unique_vals) - 1
                        positive_value = st.selectbox(
                            "Select value representing spring/presence",
                            unique_vals,
                            index=max(default_pos_idx, 0),
                        )

                        sampled = sample_scores_at_points(gdf, score, class_array, ref_meta)
                        valid_sampled = sampled[np.isfinite(sampled["suit_score"])].copy()

                        if valid_sampled.empty:
                            st.error("No validation points fall on valid suitability pixels.")
                        else:
                            valid_sampled["y_true"] = (
                                valid_sampled[label_field].astype(str) == str(positive_value)
                            ).astype(int)

                            n_pos = int(valid_sampled["y_true"].sum())
                            n_neg = int(len(valid_sampled) - n_pos)

                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Total valid points", f"{len(valid_sampled):,}")
                            col_b.metric("Spring/presence points", f"{n_pos:,}")
                            col_c.metric("Non-spring/absence points", f"{n_neg:,}")

                            if n_pos == 0 or n_neg == 0:
                                st.error("ROC-AUC requires both presence and absence points. Please check the selected field/value.")
                            else:
                                auc, fpr, tpr, roc_thresholds, roc_fig = create_roc_plot(
                                    valid_sampled["y_true"].values,
                                    valid_sampled["suit_score"].values,
                                )
                                st.metric("AUC", f"{auc:.3f}")
                                st.pyplot(roc_fig, clear_figure=False)

                                st.markdown("**Binary validation using classes 3 and 4 as suitable**")
                                y_pred = (valid_sampled["suit_class"] >= 3).astype(int).values
                                y_true = valid_sampled["y_true"].values
                                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                                cm_df = pd.DataFrame(
                                    cm,
                                    index=["Observed non-spring", "Observed spring"],
                                    columns=["Predicted unsuitable/moderate", "Predicted high/very high"],
                                )
                                st.dataframe(cm_df, use_container_width=True)

                                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                                report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Metric/Class"})
                                st.dataframe(report_df, use_container_width=True, hide_index=True)

                                st.download_button(
                                    "Download validation samples as CSV",
                                    data=dataframe_to_csv_download(valid_sampled.drop(columns="geometry")),
                                    file_name="Geo_RWH_SRP_validation_samples.csv",
                                    mime="text/csv",
                                )
                                st.download_button(
                                    "Download ROC curve PNG",
                                    data=figure_to_png_bytes(roc_fig),
                                    file_name="Geo_RWH_SRP_ROC_AUC.png",
                                    mime="image/png",
                                )

                except Exception as e:
                    st.exception(e)
            else:
                st.info("Upload validation ZIP to compute ROC-AUC.")

        with tab_download:
            st.subheader("Download outputs")
            with open(out_score_tif, "rb") as f:
                st.download_button(
                    "Download weighted suitability score GeoTIFF",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_weighted_suitability_score.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            with open(out_class_tif, "rb") as f:
                st.download_button(
                    "Download classified suitability GeoTIFF - 4 classes",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_suitability_classes_4class.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            with open(out_class_png, "rb") as f:
                st.download_button(
                    "Download map overlay PNG - 4 classes",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_suitability_overlay_4class.png",
                    mime="image/png",
                    use_container_width=True,
                )
            st.download_button(
                "Download normalized weights CSV",
                data=dataframe_to_csv_download(wdf),
                file_name="Geo_RWH_SRP_normalized_weights.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download class threshold CSV",
                data=dataframe_to_csv_download(th_df),
                file_name="Geo_RWH_SRP_class_thresholds.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if "area_df" in locals():
                st.download_button(
                    "Download area statistics CSV",
                    data=dataframe_to_csv_download(area_df),
                    file_name="Geo_RWH_SRP_area_statistics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    except Exception as e:
        st.error("Processing failed. Please check that all rasters are valid GeoTIFFs and have proper CRS information.")
        st.exception(e)
