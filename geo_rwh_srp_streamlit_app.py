# ================================================================
# Geo-RWH-SRP+ : Smart GIS Web Portal for Rainwater Harvesting
# and Spring Rejuvenation Planning
# ================================================================
# Author: Prepared for GIS-based weighted overlay + ROC/AUC validation
# Run:
#   pip install streamlit rasterio geopandas folium streamlit-folium \
#               matplotlib scikit-learn pandas numpy pillow shapely pyproj
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
from folium.plugins import Fullscreen, MeasureControl, MousePosition
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
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
    {"key": "cn", "label": "Curve Number", "hint": "cn.tif"},
    {"key": "d_str", "label": "Distance from Stream", "hint": "d_str.tif"},
    {"key": "ddensity", "label": "Drainage Density", "hint": "ddensity.tif"},
    {"key": "geology", "label": "Geology", "hint": "geology.tif"},
    {"key": "lineaments", "label": "Lineaments", "hint": "lineaments.tif"},
    {"key": "lulc", "label": "Land Use/Land Cover", "hint": "lulc.tif"},
    {"key": "rain_monsoon", "label": "Monsoon Rainfall", "hint": "rain_monsoon.tif"},
    {"key": "slope", "label": "Slope", "hint": "slope.tif"},
    {"key": "soil", "label": "Soil", "hint": "soil.tif"},
    {"key": "twi", "label": "Topographic Wetness Index", "hint": "twi.tif"},
]

CLASS_INFO = {
    0: {"name": "NoData", "color": "#00000000"},
    1: {"name": "Unsuitable", "color": "#d7191c"},
    2: {"name": "Low Suitable", "color": "#fdae61"},
    3: {"name": "Moderately Suitable", "color": "#ffffbf"},
    4: {"name": "Highly Suitable", "color": "#a6d96a"},
    5: {"name": "Very Highly Suitable", "color": "#1a9641"},
}

# RGBA values for PNG overlay: index 0 transparent, 1-5 classes
CLASS_RGBA = np.array([
    [0, 0, 0, 0],          # 0 NoData
    [215, 25, 28, 210],    # 1 Unsuitable
    [253, 174, 97, 210],   # 2 Low
    [255, 255, 191, 210],  # 3 Moderate
    [166, 217, 106, 210],  # 4 High
    [26, 150, 65, 210],    # 5 Very High
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
    Any value outside 1-5 can optionally be treated as NoData.
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


def classify_suitability(score: np.ndarray, method: str = "Equal interval"):
    """Classify continuous suitability score into five classes."""
    valid = np.isfinite(score)
    cls = np.zeros(score.shape, dtype="uint8")

    if not np.any(valid):
        raise ValueError("No valid pixels available for classification.")

    if method == "Quantile":
        thresholds = np.nanpercentile(score[valid], [20, 40, 60, 80])
        # In case of duplicate thresholds, use equal interval fallback.
        if len(np.unique(np.round(thresholds, 6))) < 4:
            thresholds = np.array([1.8, 2.6, 3.4, 4.2], dtype="float32")
    else:
        # Suitable for 1-5 weighted score.
        thresholds = np.array([1.8, 2.6, 3.4, 4.2], dtype="float32")

    cls[valid] = np.digitize(score[valid], bins=thresholds, right=True).astype("uint8") + 1
    return cls, thresholds


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
        out_arr = np.where(np.isfinite(arr), arr, nodata).astype(dtype)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)


def create_class_png(class_array: np.ndarray, out_png: str):
    """Create transparent PNG overlay from classified array."""
    safe_cls = np.nan_to_num(class_array, nan=0).astype("uint8")
    safe_cls = np.clip(safe_cls, 0, 5)
    rgba = CLASS_RGBA[safe_cls]
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_png)


def add_legend(map_obj):
    legend_items = "".join(
        [
            f"""
            <div style='display:flex;align-items:center;margin-bottom:4px;'>
                <span style='background:{CLASS_INFO[i]['color']};width:18px;height:18px;display:inline-block;border:1px solid #555;margin-right:8px;'></span>
                <span>{i}. {CLASS_INFO[i]['name']}</span>
            </div>
            """
            for i in [5, 4, 3, 2, 1]
        ]
    )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 32px;
        left: 32px;
        z-index: 9999;
        background: white;
        padding: 12px 14px;
        border: 2px solid #555;
        border-radius: 8px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
        <b>RWH/Spring Rejuvenation Suitability</b><br>
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
        background: rgba(255,255,255,0.92);
        padding: 8px 10px;
        border: 2px solid #333;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    ">
        N<br>
        <span style="font-size: 30px; line-height: 20px;">▲</span>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(north_html))


def make_folium_map(class_png: str, ref_meta: dict):
    """Build interactive folium map with overlay, legend, north arrow and scale."""
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

    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon:").add_to(m)
    add_legend(m)
    add_north_arrow(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def plot_area_chart(class_array: np.ndarray, pixel_area_m2: float):
    rows = []
    for i in [1, 2, 3, 4, 5]:
        count = int(np.sum(class_array == i))
        area_km2 = count * pixel_area_m2 / 1_000_000
        rows.append({"Class": i, "Suitability": CLASS_INFO[i]["name"], "Pixels": count, "Area_km2": area_km2})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["Suitability"], df["Area_km2"])
    ax.set_ylabel("Area (sq. km)")
    ax.set_xlabel("Suitability class")
    ax.set_title("Area under each suitability class")
    ax.tick_params(axis="x", rotation=35)
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
        # Search recursively if shapefile is inside a folder.
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


# -----------------------------
# Header
# -----------------------------
st.title("💧 Geo-RWH-SRP+ Dashboard")
st.markdown(
    """
    **A Smart GIS Web Portal for Rainwater Harvesting and Spring Rejuvenation Planning**  
    Upload classified factor rasters, assign weights, run weighted overlay, classify suitability, map the result, and validate using spring/non-spring point data.
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
st.sidebar.caption("Weights are automatically normalized before weighted overlay.")

for spec in FACTOR_SPECS:
    weights[spec["key"]] = st.sidebar.slider(
        spec["label"],
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        key=f"wt_{spec['key']}",
    )

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
    class_method = st.radio("Classification method", ["Equal interval", "Quantile"], index=0)
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
            class_array, thresholds = classify_suitability(score, method=class_method)

            out_score_tif = os.path.join(work_dir, "Geo_RWH_SRP_weighted_suitability_score.tif")
            out_class_tif = os.path.join(work_dir, "Geo_RWH_SRP_suitability_classes.tif")
            out_class_png = os.path.join(work_dir, "Geo_RWH_SRP_suitability_overlay.png")

            write_geotiff(out_score_tif, score, ref_meta, dtype="float32", nodata=-9999)
            write_geotiff(out_class_tif, class_array, ref_meta, dtype="uint8", nodata=0)
            create_class_png(class_array, out_class_png)

        # -----------------------------
        # Results summary
        # -----------------------------
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
                    "Raw weight": weights[k],
                    "Normalized weight": round(v, 4),
                    "Percent contribution": round(v * 100, 2),
                }
                for k, v in norm_w.items()
            ]
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        st.subheader("Classification thresholds")
        th_df = pd.DataFrame(
            {
                "Class": [1, 2, 3, 4, 5],
                "Suitability": [CLASS_INFO[i]["name"] for i in [1, 2, 3, 4, 5]],
                "Score interval": [
                    f"≤ {thresholds[0]:.3f}",
                    f"> {thresholds[0]:.3f} to ≤ {thresholds[1]:.3f}",
                    f"> {thresholds[1]:.3f} to ≤ {thresholds[2]:.3f}",
                    f"> {thresholds[2]:.3f} to ≤ {thresholds[3]:.3f}",
                    f"> {thresholds[3]:.3f}",
                ],
            }
        )
        st.dataframe(th_df, use_container_width=True, hide_index=True)

        # -----------------------------
        # Map and area statistics
        # -----------------------------
        tab_map, tab_stats, tab_validation, tab_download = st.tabs(
            ["🗺️ Suitability Map", "📊 Area Statistics", "✅ ROC/AUC Validation", "⬇️ Downloads"]
        )

        with tab_map:
            st.subheader("RWH and Spring Rejuvenation Suitability Map")
            m = make_folium_map(out_class_png, ref_meta)
            st_folium(m, width=None, height=650)
            st.caption("The map includes a class legend, north arrow, scale bar, coordinate display and measuring tool.")

        with tab_stats:
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

                                # Optional binary evaluation using suitability threshold >= 3.4, i.e. high and very high.
                                st.markdown("**Binary validation using classes 4 and 5 as suitable**")
                                y_pred = (valid_sampled["suit_class"] >= 4).astype(int).values
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
                    "Download classified suitability GeoTIFF",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_suitability_classes.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            with open(out_class_png, "rb") as f:
                st.download_button(
                    "Download map overlay PNG",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_suitability_overlay.png",
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

    except Exception as e:
        st.error("Processing failed. Please check that all rasters are valid GeoTIFFs and have proper CRS information.")
        st.exception(e)
