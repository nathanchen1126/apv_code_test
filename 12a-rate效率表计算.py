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
GRW_CROP_PATH = Path(r"D:\pv\grw_microsoft\grw_2023_crop.shp")

OUT_APV = Path(r"D:\pv\scenario\apv_ndvimean.xlsx")
OUT_CAPV = Path(r"D:\pv\scenario\capv_ndvimean.xlsx")

# -------------------- 空间参数 --------------------
BUFFER_RADIUS = 6000   # ★ MOD：统一 6000 m

# -------------------- 工具函数（完全复用） --------------------
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def low_memory_zonal_mean(geoms, raster_path):
    results = [np.nan] * len(geoms)
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        for i, geom in enumerate(tqdm(geoms, desc="  Zonal mean", unit="poly")):
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
                mask = geometry_mask(
                    [geom],
                    out_shape=data.shape,
                    transform=transform,
                    invert=True,
                    all_touched=(data.shape[0] < 2 or data.shape[1] < 2)
                )

                valid = mask & (~data.mask)
                if valid.any():
                    results[i] = float(data.data[valid].mean())
            except Exception:
                results[i] = np.nan
    return results

def safe_read_vector(path: Path) -> gpd.GeoDataFrame:
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

# -------------------- 核心计算 --------------------
def compute_ndvi_6000m():

    with rasterio.open(NDVI_MAX_PATH) as src:
        raster_crs = src.crs

    # ---------- APV ----------
    apv = clean_geoms(safe_read_vector(APV_PATH))
    climate = clean_geoms(safe_read_vector(CLIMATE_PATH)).to_crs(apv.crs)

    apv = gpd.sjoin(apv, climate, how="left", predicate="intersects")
    zone_col = next((c for c in apv.columns if "climate" in c.lower()), "climate_zone")
    apv["area"] = apv[zone_col].fillna("Unknown")

    apv_buf = apv.geometry.buffer(BUFFER_RADIUS)
    apv_buf_geo = gpd.GeoSeries(apv_buf, crs=apv.crs).to_crs(raster_crs)

    apv["ndvi_mean_6000m"] = low_memory_zonal_mean(apv_buf_geo, NDVI_MAX_PATH)

    apv_out = apv[["area", "ndvi_mean_6000m"]]
    apv_out.to_excel(OUT_APV, index=False)

    # ---------- CAPV（已剔除 APV） ----------
    capv = clean_geoms(safe_read_vector(GRW_CROP_PATH))
    capv = capv.to_crs(apv.crs)

    capv = gpd.sjoin(capv, climate, how="left", predicate="intersects")
    zone_col = next((c for c in capv.columns if "climate" in c.lower()), "climate_zone")
    capv["area"] = capv[zone_col].fillna("Unknown")

    capv_buf = capv.geometry.buffer(BUFFER_RADIUS)
    capv_buf_geo = gpd.GeoSeries(capv_buf, crs=capv.crs).to_crs(raster_crs)

    capv["ndvi_mean_6000m"] = low_memory_zonal_mean(capv_buf_geo, NDVI_MAX_PATH)

    capv_out = capv[["area", "ndvi_mean_6000m"]]
    capv_out.to_excel(OUT_CAPV, index=False)

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    t0 = time.time()
    compute_ndvi_6000m()
    print(f"完成，用时 {time.time() - t0:.2f} 秒")
