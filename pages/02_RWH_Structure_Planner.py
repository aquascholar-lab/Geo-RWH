# ================================================================
# Geo-RWH-SRP+ : RWH Structure Recommendation Page
# Add this file inside a Streamlit app folder as:
#   pages/02_RWH_Structure_Planner.py
# ================================================================

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RWH Structure Planner",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Structure rule table
# ------------------------------------------------------------
# Slope category: A=0-5, B=5-10, C=10-15, D=15-30, E=30-50, F=>50
# CN category: A=<55, B=56-70, C=71-85, D=>85
# LULC classes: 1=Barren, 2=Forest, 3=Agricultural, 4=Vegetation,
#               5=River Bed, 6=Hilly Shadow, 7=Settlement, 8=Snow Cover
# Stream order: 0 for non-stream/upland pixels; 1,2,3,4 for channel pixels.

STRUCTURE_RULES = [
    {
        "code": 1,
        "abbr": "FP",
        "name": "Farm Pond",
        "slope": ["A"],
        "cn": ["B"],
        "lulc": [3],
        "stream": [0],
        "color": "#1f78b4",
    },
    {
        "code": 2,
        "abbr": "PT",
        "name": "Percolation Tank / Pond",
        "slope": ["B"],
        "cn": ["A", "B"],
        "lulc": [1, 3],
        "stream": [0],
        "color": "#33a02c",
    },
    {
        "code": 3,
        "abbr": "NB",
        "name": "Nala Bunds",
        "slope": ["B"],
        "cn": ["A", "B"],
        "lulc": [1, 4],
        "stream": [1, 2],
        "color": "#b2df8a",
    },
    {
        "code": 4,
        "abbr": "ChDm",
        "name": "Check Dams",
        "slope": ["A", "B", "C"],
        "cn": ["C"],
        "lulc": [1, 2, 4, 5],
        "stream": [2, 3, 4],
        "color": "#e31a1c",
    },
    {
        "code": 5,
        "abbr": "GLP",
        "name": "Gully Plug",
        "slope": ["D"],
        "cn": ["C", "D"],
        "lulc": [1, 2, 4, 5],
        "stream": [1],
        "color": "#fb9a99",
    },
    {
        "code": 6,
        "abbr": "TERR",
        "name": "Terracing (Bench/Contour)",
        "slope": ["D"],
        "cn": ["B"],
        "lulc": [3],
        "stream": [0],
        "color": "#ff7f00",
    },
    {
        "code": 7,
        "abbr": "SCT",
        "name": "Staggered Contour Trenches",
        "slope": ["D", "E"],
        "cn": ["B"],
        "lulc": [1, 2, 4, 6],
        "stream": [0],
        "color": "#cab2d6",
    },
    {
        "code": 8,
        "abbr": "TT",
        "name": "Toe Trenches & Shoulder Bunds",
        "slope": ["D"],
        "cn": ["B"],
        "lulc": [3, 4],
        "stream": [0],
        "color": "#6a3d9a",
    },
    {
        "code": 9,
        "abbr": "ROOFTP",
        "name": "Rooftop Recharge Pits",
        "slope": ["A", "B"],
        "cn": ["D"],
        "lulc": [7],
        "stream": [0],
        "color": "#ffff99",
    },
    {
        "code": 10,
        "abbr": "VGBIO",
        "name": "Vegetative/Bioengineering Measures",
        "slope": ["E"],
        "cn": ["C", "D"],
        "lulc": [1, 2, 4, 6, 8],
        "stream": [0],
        "color": "#a6cee3",
    },
    {
        "code": 11,
        "abbr": "GrassPLA",
        "name": "Grass / Agave Plantations",
        "slope": ["F"],
        "cn": ["C", "D"],
        "lulc": [1, 3, 6, 8],
        "stream": [0],
        "color": "#b15928",
    },
]

SLOPE_CAT_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
CN_CAT_CODE = {"A": 1, "B": 2, "C": 3, "D": 4}
SLOPE_CAT_NAME = {1: "A: 0-5°", 2: "B: 5-10°", 3: "C: 10-15°", 4: "D: 15-30°", 5: "E: 30-50°", 6: "F: >50°"}
CN_CAT_NAME = {1: "A: <55", 2: "B: 56-70", 3: "C: 71-85", 4: "D: >85"}
LULC_NAME = {
    1: "Barren Land",
    2: "Forest",
    3: "Agricultural Land",
    4: "Vegetation",
    5: "River Bed",
    6: "Hilly Shadow",
    7: "Settlement",
    8: "Snow Cover",
}

STRUCTURE_INFO = {
    0: {"abbr": "None", "name": "No recommended structure", "color": "#00000000"},
}
for rule in STRUCTURE_RULES:
    STRUCTURE_INFO[rule["code"]] = {
        "abbr": rule["abbr"],
        "name": rule["name"],
        "color": rule["color"],
    }

# RGBA color lookup table for transparent PNG overlay.
def _hex_to_rgba(hex_color: str, alpha: int = 220):
    hex_color = hex_color.replace("#", "")
    if len(hex_color) == 8:  # transparent no-data
        return [0, 0, 0, 0]
    return [int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), alpha]

STRUCTURE_RGBA = np.array([_hex_to_rgba(STRUCTURE_INFO[i]["color"]) for i in range(0, 12)], dtype=np.uint8)


def save_uploaded_file(uploaded_file, out_dir: str, name_prefix: str = "") -> str:
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{name_prefix}{Path(uploaded_file.name).stem}{suffix}".replace(" ", "_")
    out_path = os.path.join(out_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def get_reference_profile(ref_path: str):
    import rasterio
    with rasterio.open(ref_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "bounds": src.bounds,
            "profile": src.profile.copy(),
        }


def read_and_align_raw_raster(src_path: str, ref_meta: dict, resampling_method: str = "nearest") -> np.ndarray:
    """Read and align raw raster to the reference grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    dst_shape = (ref_meta["height"], ref_meta["width"])
    dst = np.full(dst_shape, np.nan, dtype="float32")

    resampling = Resampling.bilinear if resampling_method == "bilinear" else Resampling.nearest

    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        # Common NoData values from unsigned rasters.
        arr = np.where(np.isin(arr, [65535, 32767, -9999, -99999]), np.nan, arr)

        same_grid = (
            src.crs == ref_meta["crs"]
            and src.transform.almost_equals(ref_meta["transform"])
            and src.height == ref_meta["height"]
            and src.width == ref_meta["width"]
        )

        if same_grid:
            dst = arr.astype("float32")
        else:
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return dst.astype("float32")


def classify_slope_to_category(slope: np.ndarray) -> np.ndarray:
    """Classify slope in degrees into A-F category codes."""
    out = np.zeros(slope.shape, dtype="uint8")
    valid = np.isfinite(slope)
    out[valid & (slope >= 0) & (slope <= 5)] = SLOPE_CAT_CODE["A"]
    out[valid & (slope > 5) & (slope <= 10)] = SLOPE_CAT_CODE["B"]
    out[valid & (slope > 10) & (slope <= 15)] = SLOPE_CAT_CODE["C"]
    out[valid & (slope > 15) & (slope <= 30)] = SLOPE_CAT_CODE["D"]
    out[valid & (slope > 30) & (slope <= 50)] = SLOPE_CAT_CODE["E"]
    out[valid & (slope > 50)] = SLOPE_CAT_CODE["F"]
    return out


def classify_cn_to_category(cn: np.ndarray) -> np.ndarray:
    """Classify Curve Number into A-D category codes."""
    out = np.zeros(cn.shape, dtype="uint8")
    valid = np.isfinite(cn)
    out[valid & (cn < 55)] = CN_CAT_CODE["A"]
    # 55 is included in B to avoid a no-category gap for continuous rasters.
    out[valid & (cn >= 55) & (cn <= 70)] = CN_CAT_CODE["B"]
    out[valid & (cn > 70) & (cn <= 85)] = CN_CAT_CODE["C"]
    out[valid & (cn > 85)] = CN_CAT_CODE["D"]
    return out


def prepare_lulc(lulc: np.ndarray) -> np.ndarray:
    out = np.rint(lulc).astype("float32")
    out = np.where(np.isfinite(out) & (out >= 1) & (out <= 8), out, 0)
    return out.astype("uint8")


def prepare_stream_order(stream_order: np.ndarray, max_order: int = 4) -> np.ndarray:
    out = np.rint(stream_order).astype("float32")
    out = np.where(np.isfinite(out) & (out >= 0), out, 0)
    out = np.where(out > max_order, max_order, out)
    return out.astype("uint8")


def recommend_structures(slope_cat: np.ndarray, cn_cat: np.ndarray, lulc_class: np.ndarray, stream_order: np.ndarray) -> np.ndarray:
    """Apply first-match rule priority to create a structure recommendation raster."""
    valid = (slope_cat > 0) & (cn_cat > 0) & (lulc_class > 0) & np.isfinite(stream_order)
    structure = np.zeros(slope_cat.shape, dtype="uint8")

    for rule in STRUCTURE_RULES:
        slope_codes = [SLOPE_CAT_CODE[x] for x in rule["slope"]]
        cn_codes = [CN_CAT_CODE[x] for x in rule["cn"]]
        mask = (
            valid
            & (structure == 0)
            & np.isin(slope_cat, slope_codes)
            & np.isin(cn_cat, cn_codes)
            & np.isin(lulc_class, rule["lulc"])
            & np.isin(stream_order, rule["stream"])
        )
        structure[mask] = rule["code"]
    return structure


def write_geotiff(out_path: str, arr: np.ndarray, ref_meta: dict, dtype: str = "uint8", nodata=0):
    import rasterio
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
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def create_structure_png(structure_array: np.ndarray, out_png: str):
    from PIL import Image
    safe = np.nan_to_num(structure_array, nan=0).astype("uint8")
    safe = np.clip(safe, 0, 11)
    rgba = STRUCTURE_RGBA[safe]
    Image.fromarray(rgba, mode="RGBA").save(out_png)


def raster_bounds_wgs84(ref_meta: dict):
    from rasterio.warp import transform_bounds
    west, south, east, north = transform_bounds(
        ref_meta["crs"],
        "EPSG:4326",
        ref_meta["bounds"].left,
        ref_meta["bounds"].bottom,
        ref_meta["bounds"].right,
        ref_meta["bounds"].top,
        densify_pts=21,
    )
    return west, south, east, north


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
    from pyproj import Transformer
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


def add_structure_legend(map_obj):
    import folium
    legend_items = ""
    for rule in STRUCTURE_RULES:
        legend_items += f"""
        <div style='display:flex;align-items:center;margin-bottom:4px;color:#111111;font-weight:600;'>
            <span style='background:{rule['color']};width:17px;height:17px;display:inline-block;border:1px solid #333;margin-right:8px;'></span>
            <span>{rule['code']}. {rule['abbr']} - {rule['name']}</span>
        </div>
        """

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
        font-size: 12px;
        max-height: 340px;
        overflow-y: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
    ">
        <div style="font-weight:800;color:#111111;margin-bottom:7px;">Recommended RWH Structures</div>
        {legend_items}
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def add_north_arrow(map_obj):
    import folium
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


def make_structure_map(structure_png: str, ref_meta: dict, springs_df: pd.DataFrame | None = None, label_col: str | None = None):
    import folium
    from folium.plugins import Fullscreen, MarkerCluster, MeasureControl, MousePosition

    west, south, east, north = raster_bounds_wgs84(ref_meta)
    center = [(south + north) / 2, (west + east) / 2]

    m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)

    folium.raster_layers.ImageOverlay(
        image=structure_png,
        bounds=[[south, west], [north, east]],
        opacity=0.82,
        name="Recommended RWH structures",
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
                if not str(col).startswith("_map_") and pd.notna(row[col]):
                    popup_lines.append(f"<b>{col}</b>: {row[col]}")
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
    add_structure_legend(m)
    add_north_arrow(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def structure_area_statistics(structure_array: np.ndarray, pixel_area_m2: float) -> pd.DataFrame:
    rows = []
    total_pixels = int(np.sum(structure_array > 0))
    total_area_km2 = total_pixels * pixel_area_m2 / 1_000_000
    for rule in STRUCTURE_RULES:
        count = int(np.sum(structure_array == rule["code"]))
        area_km2 = count * pixel_area_m2 / 1_000_000
        percent = (area_km2 / total_area_km2 * 100) if total_area_km2 > 0 else 0
        rows.append(
            {
                "Code": rule["code"],
                "Abbreviation": rule["abbr"],
                "Recommended structure": rule["name"],
                "Pixels": count,
                "Area_sq_km": round(area_km2, 3),
                "Area_percent": round(percent, 2),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------
st.title("🏗️ RWH Structure Recommendation Planner")
st.markdown(
    """
    This page recommends the **type of Rainwater Harvesting / Spring Rejuvenation structure** for each pixel based on raw **Slope**, **Curve Number**, **LULC**, and **Stream Order** layers.
    """
)

with st.expander("Input classification rules used by this module", expanded=False):
    st.markdown(
        """
        **Slope categories:** A = 0–5°, B = 5–10°, C = 10–15°, D = 15–30°, E = 30–50°, F = >50°.  
        **CN categories:** A = <55, B = 56–70, C = 71–85, D = >85.  
        **LULC classes:** 1 Barren Land, 2 Forest, 3 Agricultural Land, 4 Vegetation, 5 River Bed, 6 Hilly Shadow, 7 Settlement, 8 Snow Cover.  
        **Stream order:** 0 is treated as non-stream/upland pixel; 1–4 are treated as stream-order pixels.  
        Where more than one rule can match, the rule priority follows the table order provided by you.
        """
    )
    rules_df = pd.DataFrame(
        [
            {
                "Code": r["code"],
                "Abbreviation": r["abbr"],
                "Structure": r["name"],
                "Slope Category": ", ".join(r["slope"]),
                "CN Category": ", ".join(r["cn"]),
                "LULC Classes": ", ".join(map(str, r["lulc"])),
                "Stream Order": ", ".join(map(str, r["stream"])),
            }
            for r in STRUCTURE_RULES
        ]
    )
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

work_dir = tempfile.mkdtemp(prefix="rwh_structure_planner_")

st.sidebar.header("Upload raw input rasters")
st.sidebar.caption("Slope and CN should be raw continuous rasters. LULC and stream order should be class rasters.")

slope_file = st.sidebar.file_uploader("Upload raw slope raster", type=["tif", "tiff"], key="raw_slope")
cn_file = st.sidebar.file_uploader("Upload raw Curve Number raster", type=["tif", "tiff"], key="raw_cn")
lulc_file = st.sidebar.file_uploader("Upload LULC class raster", type=["tif", "tiff"], key="raw_lulc")
stream_file = st.sidebar.file_uploader("Upload stream order raster", type=["tif", "tiff"], key="raw_stream_order")

assume_zero_stream = st.sidebar.checkbox(
    "If stream order raster is not uploaded, assume stream order = 0 everywhere",
    value=False,
    help="Use this only when you want to recommend only non-channel/upland structures. Channel structures such as Nala Bunds, Check Dams and Gully Plugs require a stream-order raster.",
)

st.sidebar.divider()
st.sidebar.header("Raster alignment")
resample_continuous = st.sidebar.selectbox(
    "Resampling for Slope and CN if grids differ",
    ["bilinear", "nearest"],
    index=0,
)
resample_categorical = st.sidebar.selectbox(
    "Resampling for LULC and Stream Order if grids differ",
    ["nearest"],
    index=0,
)

run = st.sidebar.button("🚀 Generate RWH Structure Map", type="primary", use_container_width=True)

if slope_file is None or cn_file is None or lulc_file is None:
    st.warning("Please upload raw Slope, raw CN, and LULC raster files.")
    st.info("For complete rule-based structure recommendation, also upload a Stream Order raster. Otherwise, tick the option to assume stream order = 0 everywhere.")
    st.stop()

if stream_file is None and not assume_zero_stream:
    st.warning("Stream Order raster is required for complete structure recommendation because several rules depend on stream order.")
    st.stop()

if run or (slope_file and cn_file and lulc_file and (stream_file or assume_zero_stream)):
    try:
        with st.spinner("Reading raw rasters and applying structure-recommendation rules..."):
            slope_path = save_uploaded_file(slope_file, work_dir, "slope_")
            cn_path = save_uploaded_file(cn_file, work_dir, "cn_")
            lulc_path = save_uploaded_file(lulc_file, work_dir, "lulc_")

            ref_meta = get_reference_profile(slope_path)

            slope_raw = read_and_align_raw_raster(slope_path, ref_meta, resampling_method=resample_continuous)
            cn_raw = read_and_align_raw_raster(cn_path, ref_meta, resampling_method=resample_continuous)
            lulc_raw = read_and_align_raw_raster(lulc_path, ref_meta, resampling_method=resample_categorical)

            if stream_file is not None:
                stream_path = save_uploaded_file(stream_file, work_dir, "stream_order_")
                stream_raw = read_and_align_raw_raster(stream_path, ref_meta, resampling_method=resample_categorical)
            else:
                stream_raw = np.zeros((ref_meta["height"], ref_meta["width"]), dtype="float32")

            slope_cat = classify_slope_to_category(slope_raw)
            cn_cat = classify_cn_to_category(cn_raw)
            lulc_class = prepare_lulc(lulc_raw)
            stream_order = prepare_stream_order(stream_raw)

            structure = recommend_structures(slope_cat, cn_cat, lulc_class, stream_order)

            out_structure_tif = os.path.join(work_dir, "Geo_RWH_SRP_recommended_RWH_structures.tif")
            out_slope_cat_tif = os.path.join(work_dir, "Geo_RWH_SRP_slope_categories.tif")
            out_cn_cat_tif = os.path.join(work_dir, "Geo_RWH_SRP_CN_categories.tif")
            out_structure_png = os.path.join(work_dir, "Geo_RWH_SRP_recommended_RWH_structures_overlay.png")

            write_geotiff(out_structure_tif, structure, ref_meta, dtype="uint8", nodata=0)
            write_geotiff(out_slope_cat_tif, slope_cat, ref_meta, dtype="uint8", nodata=0)
            write_geotiff(out_cn_cat_tif, cn_cat, ref_meta, dtype="uint8", nodata=0)
            create_structure_png(structure, out_structure_png)

        st.success("RWH structure recommendation completed successfully.")

        pixel_area_m2 = abs(ref_meta["transform"].a * ref_meta["transform"].e)
        total_rec_pixels = int(np.sum(structure > 0))
        total_rec_area_km2 = total_rec_pixels * pixel_area_m2 / 1_000_000

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pixels with recommendation", f"{total_rec_pixels:,}")
        c2.metric("Recommended area", f"{total_rec_area_km2:,.2f} sq. km")
        c3.metric("Structure types found", f"{len(np.unique(structure[structure > 0]))}")
        c4.metric("Raster size", f"{ref_meta['width']} × {ref_meta['height']}")

        tab_map, tab_stats, tab_categories, tab_download = st.tabs(
            ["🗺️ Structure Map", "📊 Structure Area Statistics", "🧩 Category Summary", "⬇️ Downloads"]
        )

        with tab_map:
            st.subheader("Recommended RWH Structure Map")
            st.markdown("Optionally upload a spring-location CSV to display existing spring locations on the structure-recommendation map.")

            springs_df_for_map = None
            label_col = None
            springs_csv = st.file_uploader(
                "Upload springs location CSV for map display",
                type=["csv"],
                key="structure_springs_csv",
                help="CSV should contain latitude/longitude, lon/lat, x/y, or easting/northing columns.",
            )
            if springs_csv is not None:
                try:
                    spring_df = pd.read_csv(springs_csv)
                    if not spring_df.empty:
                        cols = list(spring_df.columns)
                        default_x = auto_detect_column(cols, ["longitude", "lon", "long", "x", "easting"])
                        default_y = auto_detect_column(cols, ["latitude", "lat", "y", "northing"])
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        x_col = cc1.selectbox("Longitude/X column", cols, index=cols.index(default_x))
                        y_col = cc2.selectbox("Latitude/Y column", cols, index=cols.index(default_y))
                        coord_mode = cc3.selectbox("CSV coordinate system", ["Latitude/Longitude (EPSG:4326)", "Same CRS as raster"], index=0)
                        label_options = ["None"] + cols
                        label_guess = auto_detect_column(cols, ["name", "id", "spring", "site"])
                        label_index = label_options.index(label_guess) if label_guess in label_options else 0
                        label_selected = cc4.selectbox("Point label column", label_options, index=label_index)
                        label_col = None if label_selected == "None" else label_selected
                        springs_df_for_map = prepare_spring_csv_for_map(spring_df, x_col, y_col, coord_mode, ref_meta)
                        st.success(f"{len(springs_df_for_map):,} spring location point(s) will be shown on the map.")
                    else:
                        st.warning("Uploaded spring CSV is empty.")
                except Exception as e:
                    st.warning("Spring CSV could not be displayed. Please check coordinate columns.")
                    st.exception(e)

            from streamlit_folium import st_folium
            fmap = make_structure_map(out_structure_png, ref_meta, springs_df=springs_df_for_map, label_col=label_col)
            st_folium(fmap, width=None, height=720)
            st.caption("The map includes recommended structure classes, dark legend, north arrow, scale bar, coordinate display and measuring tool.")

        with tab_stats:
            st.subheader("Area statistics of recommended structures")
            stats_df = structure_area_statistics(structure, pixel_area_m2)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            try:
                import matplotlib.pyplot as plt
                plot_df = stats_df[stats_df["Pixels"] > 0].copy()
                if not plot_df.empty:
                    fig, ax = plt.subplots(figsize=(9, 4.8))
                    ax.bar(plot_df["Abbreviation"], plot_df["Area_sq_km"])
                    ax.set_xlabel("Recommended structure")
                    ax.set_ylabel("Area (sq. km)")
                    ax.set_title("Area under each recommended RWH structure")
                    ax.tick_params(axis="x", rotation=30)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=False)
            except Exception:
                pass

        with tab_categories:
            st.subheader("Generated intermediate categories")
            col1, col2, col3 = st.columns(3)

            slope_counts = pd.Series(slope_cat.ravel()).value_counts().sort_index()
            slope_df = pd.DataFrame(
                {
                    "Slope category code": slope_counts.index,
                    "Category": [SLOPE_CAT_NAME.get(int(i), "NoData") for i in slope_counts.index],
                    "Pixels": slope_counts.values,
                }
            )
            cn_counts = pd.Series(cn_cat.ravel()).value_counts().sort_index()
            cn_df = pd.DataFrame(
                {
                    "CN category code": cn_counts.index,
                    "Category": [CN_CAT_NAME.get(int(i), "NoData") for i in cn_counts.index],
                    "Pixels": cn_counts.values,
                }
            )
            lulc_counts = pd.Series(lulc_class.ravel()).value_counts().sort_index()
            lulc_df = pd.DataFrame(
                {
                    "LULC class": lulc_counts.index,
                    "Class name": [LULC_NAME.get(int(i), "NoData/Other") for i in lulc_counts.index],
                    "Pixels": lulc_counts.values,
                }
            )
            col1.markdown("**Slope category summary**")
            col1.dataframe(slope_df, use_container_width=True, hide_index=True)
            col2.markdown("**CN category summary**")
            col2.dataframe(cn_df, use_container_width=True, hide_index=True)
            col3.markdown("**LULC class summary**")
            col3.dataframe(lulc_df, use_container_width=True, hide_index=True)

            st.markdown("**Rule table used for recommendations**")
            st.dataframe(rules_df, use_container_width=True, hide_index=True)

        with tab_download:
            st.subheader("Download structure recommendation outputs")
            with open(out_structure_tif, "rb") as f:
                st.download_button(
                    "Download recommended RWH structures GeoTIFF",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_recommended_RWH_structures.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            with open(out_structure_png, "rb") as f:
                st.download_button(
                    "Download recommended structures map overlay PNG",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_recommended_RWH_structures_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with open(out_slope_cat_tif, "rb") as f:
                st.download_button(
                    "Download slope category GeoTIFF",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_slope_categories.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            with open(out_cn_cat_tif, "rb") as f:
                st.download_button(
                    "Download CN category GeoTIFF",
                    data=f.read(),
                    file_name="Geo_RWH_SRP_CN_categories.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            st.download_button(
                "Download structure area statistics CSV",
                data=dataframe_to_csv_download(stats_df),
                file_name="Geo_RWH_SRP_structure_area_statistics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download rule table CSV",
                data=dataframe_to_csv_download(rules_df),
                file_name="Geo_RWH_SRP_structure_rule_table.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error("Structure recommendation failed. Please check raster CRS, raster format, and input value ranges.")
        st.exception(e)
