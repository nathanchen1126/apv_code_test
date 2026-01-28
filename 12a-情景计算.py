from __future__ import annotations

import warnings
import time
from pathlib import Path
from typing import Iterable, List
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

warnings.filterwarnings("ignore")

# -------------------- 路径配置 --------------------
NDVI_MAX_PATH = Path(r"D:\pv\scenario\ndvi_max_2023.tif")
APV_PATH = Path(r"D:\pv\result\result_merge_postprocess\apv_2023_all_merge_block.shp")
CLIMATE_PATH = Path(r"D:\pv\data\Chinese_climate\Chinese_climate.shp")
GRW_PATH = Path(r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg")
GRW_CROP_PATH = Path(r"D:\pv\grw_microsoft\grw_2023_crop.shp")
APV_RATE_XLSX = Path(r"D:\pv\scenario\apv_efficiency_rate.xlsx")
SCENARIO_XLSX = Path(r"D:\pv\scenario\ndvi_climate_change.xlsx")

# -------------------- ★ 空间参数（可调试） --------------------
IMPACT_RADIUS   = 6000    # 影响区半径
BG_INNER_RADIUS = 6000    # 背景区下限
BG_OUTER_RADIUS = 10000   # 背景区上限

# -------------------- 工具函数 --------------------
@contextmanager
def step_timer(desc: str):
    print(f"[{desc}] 正在启动...")
    start = time.time()
    yield
    print(f"[{desc}] 完成. 耗时: {time.time() - start:.2f} 秒")
    print("-" * 40)

def safe_read_vector(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    for enc in [None, "utf-8", "gbk", "gb18030"]:
        try:
            return gpd.read_file(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取 {path}")

def clean_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if not gdf.is_valid.all():
        gdf.loc[~gdf.is_valid, "geometry"] = gdf.loc[~gdf.is_valid, "geometry"].buffer(0)
    return gdf

def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

# -------------------- 核心工具 --------------------
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def low_memory_zonal_mean(
    geoms: Iterable,
    raster_path: Path,
    nodata_val: float = -9999
) -> List[float]:
    geom_list = list(geoms)
    results = [np.nan] * len(geom_list)

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width

        for i, geom in enumerate(tqdm(geom_list, desc="  Analyzing", unit="poly")):
            if geom is None or geom.is_empty:
                continue
            try:
                minx, miny, maxx, maxy = geom.bounds
                row_start, col_start = src.index(minx, maxy, op=float)
                row_stop, col_stop = src.index(maxx, miny, op=float)

                r1 = clamp(int(np.floor(row_start)), 0, height)
                r2 = clamp(int(np.ceil(row_stop)), 0, height)
                c1 = clamp(int(np.floor(col_start)), 0, width)
                c2 = clamp(int(np.ceil(col_stop)), 0, width)

                if r2 <= r1 or c2 <= c1:
                    continue

                window = Window(c1, r1, c2 - c1, r2 - r1)
                data = src.read(1, window=window, masked=True)
                if data.size == 0:
                    continue

                transform = src.window_transform(window)
                geom_mask = geometry_mask(
                    [geom],
                    out_shape=data.shape,
                    transform=transform,
                    invert=True,
                    all_touched=(data.shape[0] < 2 or data.shape[1] < 2)
                )

                valid = geom_mask & (~data.mask)
                if valid.any():
                    results[i] = float(data.data[valid].mean())
            except Exception:
                results[i] = np.nan

    return results

# -------------------- Buffer --------------------
def ring_buffer(series: gpd.GeoSeries, inner: float, outer: float) -> gpd.GeoSeries:
    outer_buf = series.buffer(outer, resolution=8)
    inner_buf = series.buffer(inner, resolution=8)
    ring = outer_buf.difference(inner_buf)
    ring = gpd.GeoSeries(ring, index=series.index, crs=series.crs)
    ring.loc[ring.is_empty | ring.isna()] = None
    return ring

# -------------------- APV 效率 --------------------
def compute_apv_efficiency(ndvi_path: Path):
    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs

    apv = clean_geoms(safe_read_vector(APV_PATH))
    climate = clean_geoms(safe_read_vector(CLIMATE_PATH))
    if climate.crs != apv.crs:
        climate = climate.to_crs(apv.crs)

    apv = gpd.sjoin(apv, climate, how="left", predicate="intersects")
    zone_col = next((c for c in apv.columns if "climate" in c.lower()), "climate_zone")
    apv["climate_zone"] = apv[zone_col].fillna("Unknown")

    # ★ PARAM: APV 影响区
    apv_impact = apv.geometry.buffer(IMPACT_RADIUS)
    apv_impact_geo = gpd.GeoSeries(apv_impact, crs=apv.crs).to_crs(raster_crs)
    apv["ndvi_apv"] = low_memory_zonal_mean(apv_impact_geo, ndvi_path)

    # ★ PARAM: APV 背景区
    ring = ring_buffer(apv.geometry, BG_INNER_RADIUS, BG_OUTER_RADIUS)
    apv["ndvi_base"] = np.nan
    valid = ring.notna()
    if valid.any():
        apv.loc[valid, "ndvi_base"] = low_memory_zonal_mean(
            ring[valid].to_crs(raster_crs), ndvi_path
        )

    apv["rate_ndvi"] = apv["ndvi_apv"] / apv["ndvi_base"]
    apv["rate_ndvi"] = apv["rate_ndvi"].replace([np.inf, -np.inf], np.nan)

    out = apv[["climate_zone", "ndvi_apv", "ndvi_base", "rate_ndvi"]]
    ensure_dir(APV_RATE_XLSX)
    out.to_excel(APV_RATE_XLSX, index=False)
    return out

def derive_scenarios(apv_rates: pd.DataFrame):
    g = apv_rates.dropna(subset=["rate_ndvi"]).groupby("climate_zone")["rate_ndvi"]
    return pd.DataFrame({
        "climate_zone": g.mean().index,
        "rate_top": g.quantile(0.9).values,
        "rate_med": g.quantile(0.5).values,
        "rate_low": g.quantile(0.1).values
    })

# -------------------- CAPV → APV 模拟 --------------------
def simulate_capv(ndvi_path: Path, scenario_rates: pd.DataFrame):
    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs

    capv = prepare_capv_source()
    capv_m = capv.to_crs(epsg=3857) if capv.crs.is_geographic else capv

    climate = safe_read_vector(CLIMATE_PATH).to_crs(capv_m.crs)
    capv_m = gpd.sjoin(capv_m, climate, how="left", predicate="intersects")
    zone_col = next((c for c in capv_m.columns if "climate" in c.lower()), "climate_zone")
    capv_m["climate_zone"] = capv_m[zone_col].fillna("Unknown")

    # ★ PARAM: CAPV current = 影响区
    capv_impact = capv_m.geometry.buffer(IMPACT_RADIUS)
    capv_impact_geo = gpd.GeoSeries(capv_impact, crs=capv_m.crs).to_crs(raster_crs)
    capv_m["ndvi_capv_current"] = low_memory_zonal_mean(capv_impact_geo, ndvi_path)

    # ★ PARAM: CAPV 背景区
    ring = ring_buffer(capv_m.geometry, BG_INNER_RADIUS, BG_OUTER_RADIUS)
    capv_m["ndvi_capv_ref"] = np.nan
    valid = ring.notna()
    if valid.any():
        capv_m.loc[valid, "ndvi_capv_ref"] = low_memory_zonal_mean(
            ring[valid].to_crs(raster_crs), ndvi_path
        )

    capv_m = capv_m.merge(scenario_rates, on="climate_zone", how="left")
    for lvl in ["top", "med", "low"]:
        capv_m[f"sim_{lvl}"] = capv_m["ndvi_capv_ref"] * capv_m[f"rate_{lvl}"]

    summary = capv_m.groupby("climate_zone").agg(
        ndvi_capv_current=("ndvi_capv_current", "mean"),
        sim_top=("sim_top", "mean"),
        sim_med=("sim_med", "mean"),
        sim_low=("sim_low", "mean"),
        count=("geometry", "count")
    ).reset_index()

    ensure_dir(SCENARIO_XLSX)
    summary.to_excel(SCENARIO_XLSX, index=False)

# -------------------- CAPV 数据准备 --------------------
def prepare_capv_source() -> gpd.GeoDataFrame:
    if GRW_CROP_PATH.exists():
        capv = safe_read_vector(GRW_CROP_PATH)
    else:
        capv = clean_geoms(safe_read_vector(GRW_PATH))
        if "landcover_in_2018" in capv.columns:
            capv["lc"] = capv["landcover_in_2018"].astype(str).str.extract(r"(\d+)")[0].astype(float)
            capv = capv[capv["lc"].isin([10, 20])].drop(columns=["lc"])
        year = pd.to_numeric(capv["construction_year"], errors="coerce")
        capv = capv[year.notna() & (year <= 2023)]

    apv = clean_geoms(safe_read_vector(APV_PATH))
    if capv.crs != apv.crs:
        apv = apv.to_crs(capv.crs)

    capv = capv.loc[~capv.geometry.intersects(apv.geometry.unary_union)].copy()
    ensure_dir(GRW_CROP_PATH)
    capv.to_file(GRW_CROP_PATH)
    return capv

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    total_start = time.time()

    with step_timer("1/4 Prepare Vectors"):
        prepare_capv_source()

    with step_timer("2/4 APV Efficiency"):
        apv_rates = compute_apv_efficiency(NDVI_MAX_PATH)

    with step_timer("3/4 Scenarios"):
        scenario_rates = derive_scenarios(apv_rates)

    with step_timer("4/4 Simulation"):
        simulate_capv(NDVI_MAX_PATH, scenario_rates)

    print(f"\n全部流程结束. 总耗时: {time.time() - total_start:.2f} 秒")
