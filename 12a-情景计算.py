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
from rasterio.mask import mask

warnings.filterwarnings("ignore")

# -------------------- Paths --------------------
NDVI_MAX_PATH = Path(r"D:\pv\scenario\ndvi_max_2023.tif")

APV_PATH = Path(r"D:\pv\result\result_merge_postprocess\apv_2023_all_merge_block.shp")
CLIMATE_PATH = Path(r"D:\pv\data\Chinese_climate\Chinese_climate.shp")
GRW_PATH = Path(r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg")

GRW_CROP_PATH = Path(r"D:\pv\grw_microsoft\grw_2023_crop.shp")
APV_RATE_XLSX = Path(r"D:\pv\scenario\apv_efficiency_rate.xlsx")
SCENARIO_XLSX = Path(r"D:\pv\scenario\ndvi_climate_change.xlsx")

INNER_BUF = 2000
OUTER_BUF = 6000

# -------------------- Utils --------------------
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

# -------------------- 核心函数（安全版） --------------------
def low_memory_zonal_mean(
    geoms: Iterable,
    raster_path: Path,
    nodata_val: float = -9999
) -> List[float]:

    geom_list = list(geoms)
    results = [np.nan] * len(geom_list)

    with rasterio.open(raster_path) as src:
        src_nodata = src.nodata if src.nodata is not None else nodata_val

        for i, geom in enumerate(
            tqdm(geom_list, desc="  Analyzing Polygons", unit="poly")
        ):
            if geom is None or geom.is_empty:
                continue

            try:
                out_image, _ = mask(
                    src,
                    [geom],
                    crop=True,
                    nodata=src_nodata,
                    all_touched=False
                )

                data = out_image[0]
                data = np.where(data == src_nodata, np.nan, data)

                if np.all(np.isnan(data)):
                    continue

                results[i] = float(np.nanmean(data))

            except Exception:
                results[i] = np.nan

    return results

# -------------------- Vector Prep --------------------
def prepare_capv_source() -> gpd.GeoDataFrame:
    if GRW_CROP_PATH.exists():
        return safe_read_vector(GRW_CROP_PATH)

    capv = clean_geoms(safe_read_vector(GRW_PATH))

    if "landcover_in_2018" in capv.columns:
        capv["lc"] = (
            capv["landcover_in_2018"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
            .astype(float)
        )
        capv = capv[capv["lc"].isin([10, 20])].drop(columns=["lc"])

    year = pd.to_numeric(capv["construction_year"], errors="coerce")
    capv = capv[year.notna() & (year <= 2023)]

    ensure_dir(GRW_CROP_PATH)
    capv.to_file(GRW_CROP_PATH, encoding="utf-8")

    return capv

def ring_buffer(series: gpd.GeoSeries, inner: float, outer: float) -> gpd.GeoSeries:
    ring = series.buffer(outer).difference(series.buffer(inner))
    ring = gpd.GeoSeries(ring, index=series.index, crs=series.crs)
    ring.loc[ring.isna() | ring.is_empty] = None
    return ring

# -------------------- APV Efficiency --------------------
def compute_apv_efficiency(ndvi_path: Path):

    with rasterio.open(ndvi_path) as src:
        ndvi_crs = src.crs

    apv = clean_geoms(safe_read_vector(APV_PATH)).to_crs(ndvi_crs)
    climate = clean_geoms(safe_read_vector(CLIMATE_PATH)).to_crs(ndvi_crs)

    apv = gpd.sjoin(apv, climate, how="left", predicate="intersects")
    zone_col = next(
        (c for c in apv.columns if "climate" in c.lower() or "zone" in c.lower()),
        "climate_zone"
    )
    apv["climate_zone"] = apv[zone_col].fillna("Unknown")
    apv = apv.loc[:, ~apv.columns.duplicated()]

    print("  > 计算 APV 内部 NDVI...")
    apv["ndvi_apv"] = low_memory_zonal_mean(apv.geometry, ndvi_path)

    print("  > 生成缓冲区...")
    apv_ring = ring_buffer(apv.geometry, INNER_BUF, OUTER_BUF)

    print("  > 计算背景 NDVI...")
    apv["ndvi_base"] = low_memory_zonal_mean(apv_ring, ndvi_path)

    apv["rate_ndvi"] = apv["ndvi_apv"] / apv["ndvi_base"]
    apv["rate_ndvi"] = apv["rate_ndvi"].replace([np.inf, -np.inf], np.nan)

    out = apv[["climate_zone", "ndvi_apv", "ndvi_base", "rate_ndvi"]]
    ensure_dir(APV_RATE_XLSX)
    out.to_excel(APV_RATE_XLSX, index=False)

    return out

# -------------------- Scenarios --------------------
def derive_scenarios(apv_rates: pd.DataFrame):
    g = apv_rates.dropna(subset=["rate_ndvi"]).groupby("climate_zone")["rate_ndvi"]
    return pd.DataFrame({
        "climate_zone": g.mean().index,
        "rate_top": g.quantile(0.9).values,
        "rate_med": g.quantile(0.5).values,
        "rate_low": g.quantile(0.1).values
    })

# -------------------- Simulation --------------------
def simulate_capv(ndvi_path: Path, scenario_rates: pd.DataFrame):

    with rasterio.open(ndvi_path) as src:
        ndvi_crs = src.crs

    capv = prepare_capv_source().to_crs(ndvi_crs)
    climate = safe_read_vector(CLIMATE_PATH).to_crs(ndvi_crs)

    capv = gpd.sjoin(capv, climate, how="left", predicate="intersects")
    zone_col = next(
        (c for c in capv.columns if "climate" in c.lower() or "zone" in c.lower()),
        "climate_zone"
    )
    capv["climate_zone"] = capv[zone_col].fillna("Unknown")
    capv = capv.loc[:, ~capv.columns.duplicated()]

    print("  > 计算 CAPV 当前 NDVI...")
    capv["ndvi_capv_current"] = low_memory_zonal_mean(capv.geometry, ndvi_path)

    print("  > 计算参考 NDVI...")
    capv_ring = ring_buffer(capv.geometry, INNER_BUF, OUTER_BUF)
    capv["ndvi_capv_ref"] = low_memory_zonal_mean(capv_ring, ndvi_path)

    capv = capv.merge(scenario_rates, on="climate_zone", how="left")

    for lvl in ["top", "med", "low"]:
        capv[f"sim_{lvl}"] = capv["ndvi_capv_ref"] * capv[f"rate_{lvl}"]

    summary = capv.groupby("climate_zone").agg(
        ndvi_capv_current=("ndvi_capv_current", "mean"),
        sim_top=("sim_top", "mean"),
        sim_med=("sim_med", "mean"),
        sim_low=("sim_low", "mean"),
        count=("geometry", "count")
    ).reset_index()

    ensure_dir(SCENARIO_XLSX)
    summary.to_excel(SCENARIO_XLSX, index=False)
    print(summary.head())

# -------------------- Main --------------------
if __name__ == "__main__":

    if not NDVI_MAX_PATH.exists():
        raise FileNotFoundError(NDVI_MAX_PATH)

    with step_timer("1/4 Prepare Vectors"):
        prepare_capv_source()

    with step_timer("2/4 APV Efficiency"):
        apv_rates = compute_apv_efficiency(NDVI_MAX_PATH)

    with step_timer("3/4 Scenarios"):
        scenario_rates = derive_scenarios(apv_rates)

    with step_timer("4/4 Simulation"):
        simulate_capv(NDVI_MAX_PATH, scenario_rates)
