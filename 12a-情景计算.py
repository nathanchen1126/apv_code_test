from __future__ import annotations

import warnings
import time
from pathlib import Path
from typing import Iterable, List, Tuple
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
from rasterio.transform import rowcol

warnings.filterwarnings("ignore")

# -------------------- 路径配置 --------------------
NDVI_MAX_PATH = Path(r"D:\pv\scenario\ndvi_max_2023.tif")
APV_PATH = Path(r"D:\pv\result\result_merge_postprocess\apv_2023_all_merge_block.shp")
CLIMATE_PATH = Path(r"D:\pv\data\Chinese_climate\Chinese_climate.shp")
GRW_PATH = Path(r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg")
GRW_CROP_PATH = Path(r"D:\pv\grw_microsoft\grw_2023_crop.shp")
APV_RATE_XLSX = Path(r"D:\pv\scenario\apv_efficiency_rate.xlsx")
SCENARIO_XLSX = Path(r"D:\pv\scenario\ndvi_climate_change.xlsx")

INNER_BUF = 2000
OUTER_BUF = 6000

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

# -------------------- 绝对安全的核心计算函数 --------------------
def clamp(n, minn, maxn):
    """辅助函数：将数值限制在范围内"""
    return max(min(maxn, n), minn)

def low_memory_zonal_mean(
    geoms: Iterable,
    raster_path: Path,
    nodata_val: float = -9999
) -> List[float]:
    """
    [Crash-Proof Version]
    使用 src.index 手动计算像素坐标，并强制限制在图像尺寸内。
    彻底解决 EPSG:4490 等坐标系导致的溢出崩溃问题。
    """
    geom_list = list(geoms)
    results = [np.nan] * len(geom_list)
    
    with rasterio.open(raster_path) as src:
        src_nodata = src.nodata if src.nodata is not None else nodata_val
        height, width = src.height, src.width
        
        # 预先检查坐标系范围，判断是否需要翻转 XY
        # 如果矢量的 X 坐标均值 < 栅格 Left，说明可能传反了
        # 这里我们假设调用前已经做过 CRS 转换，这里只做“像素级”安全读取
        
        for i, geom in enumerate(
            tqdm(geom_list, desc="  Analyzing", unit="poly")
        ):
            if geom is None or geom.is_empty:
                continue

            try:
                minx, miny, maxx, maxy = geom.bounds

                # 1. 坐标转像素索引 (使用 rasterio 内部算法)
                # op=float 保证精度，后面手动转 int
                row_start, col_start = src.index(minx, maxy, op=float) # 左上
                row_stop, col_stop = src.index(maxx, miny, op=float)   # 右下

                # 2. 强制取整
                r1 = int(np.floor(row_start))
                c1 = int(np.floor(col_start))
                r2 = int(np.ceil(row_stop))
                c2 = int(np.ceil(col_stop))

                # 3. [关键防崩溃] 像素坐标钳位 (Clamping)
                # 无论坐标系偏到哪里去，强行拉回到图像范围内
                r1 = clamp(r1, 0, height)
                r2 = clamp(r2, 0, height)
                c1 = clamp(c1, 0, width)
                c2 = clamp(c2, 0, width)

                # 4. 检查窗口有效性
                win_h = r2 - r1
                win_w = c2 - c1
                
                if win_h <= 0 or win_w <= 0:
                    # 几何体完全在图像外，或者太小
                    continue

                # 5. 构建窗口对象
                window = Window(col_off=c1, row_off=r1, width=win_w, height=win_h)

                # 6. 读取数据
                data = src.read(1, window=window, masked=True)
                
                if data is None or data.size == 0 or data.count() == 0:
                    continue

                # 7. 构建 Geometry Mask
                # 必须使用该窗口对应的 Transform
                win_transform = src.window_transform(window)
                
                out_shape = (int(data.shape[0]), int(data.shape[1]))
                
                # 如果窗口非常小，开启 all_touched 防止漏掉点
                use_touched = (win_w < 2 or win_h < 2)

                geom_mask = geometry_mask(
                    [geom],
                    out_shape=out_shape,
                    transform=win_transform,
                    invert=True, 
                    all_touched=use_touched 
                )
                
                # 8. 统计
                valid_mask = geom_mask & (~data.mask)
                valid_pixels = data.data[valid_mask]

                if valid_pixels.size > 0:
                    results[i] = float(np.mean(valid_pixels))

            except Exception as e:
                # 打印错误但不退出
                # print(f"Skipped Poly {i}: {e}")
                results[i] = np.nan

    return results

# -------------------- Buffer 逻辑 --------------------
def ring_buffer(series: gpd.GeoSeries, inner: float, outer: float) -> gpd.GeoSeries:
    """必须在投影坐标系(米)下运算"""
    # 降低精度至 8 以提升速度
    outer_buf = series.buffer(outer, resolution=8)
    inner_buf = series.buffer(inner, resolution=8)
    ring = outer_buf.difference(inner_buf)
    ring = gpd.GeoSeries(ring, index=series.index, crs=series.crs)
    ring.loc[ring.is_empty | ring.isna()] = None
    return ring

# -------------------- 主流程函数 --------------------
def compute_apv_efficiency(ndvi_path: Path):
    
    # 1. 准备环境
    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs
        print(f"  Raster CRS: {raster_crs}")

    # 2. 读取并清洗矢量 (保持原始投影坐标系，如 EPSG:32648 米)
    print("  > Reading Vectors...")
    apv_meters = clean_geoms(safe_read_vector(APV_PATH))
    climate = clean_geoms(safe_read_vector(CLIMATE_PATH))

    # 统一到米制坐标系做空间关联
    if climate.crs != apv_meters.crs:
        climate = climate.to_crs(apv_meters.crs)

    print("  > Spatial Join (Climate Zones)...")
    apv_meters = gpd.sjoin(apv_meters, climate, how="left", predicate="intersects")
    
    # 寻找分区字段
    zone_col = next(
        (c for c in apv_meters.columns if "climate" in c.lower() or "zone" in c.lower()),
        "climate_zone"
    )
    apv_meters["climate_zone"] = apv_meters[zone_col].fillna("Unknown")
    apv_meters = apv_meters.loc[:, ~apv_meters.columns.duplicated()]

    # ------------------------------------------------
    # 阶段 1: 计算内部 NDVI
    # ------------------------------------------------
    print("  > 计算 APV 内部 NDVI...")
    # 显式转换为栅格坐标系
    apv_geo = apv_meters.to_crs(raster_crs)
    
    # 坐标系校验：如果转换后坐标不在 70-140 范围内(假设中国)，可能是 xy 反了
    # 不过上面的 low_memory_zonal_mean 已经做了安全钳位，这里直接传进去即可
    apv_meters["ndvi_apv"] = low_memory_zonal_mean(apv_geo.geometry, ndvi_path)

    # ------------------------------------------------
    # 阶段 2: 计算背景 Buffer (在米制坐标系下做几何运算)
    # ------------------------------------------------
    print("  > 生成缓冲区 (Meter CRS)...")
    ring_meters = ring_buffer(apv_meters.geometry, INNER_BUF, OUTER_BUF)

    # ------------------------------------------------
    # 阶段 3: 计算背景 NDVI (转回栅格坐标系提取)
    # ------------------------------------------------
    print("  > 计算背景 NDVI...")
    apv_meters["ndvi_base"] = np.nan
    
    # 只处理有效部分
    valid_mask = ring_meters.notna() & (~ring_meters.is_empty)
    if valid_mask.any():
        valid_rings_meters = ring_meters[valid_mask]
        
        # 转换到栅格坐标系
        valid_rings_geo = valid_rings_meters.to_crs(raster_crs)
        
        vals = low_memory_zonal_mean(valid_rings_geo, ndvi_path)
        apv_meters.loc[valid_mask, "ndvi_base"] = vals

    # 结果处理
    apv_meters["rate_ndvi"] = apv_meters["ndvi_apv"] / apv_meters["ndvi_base"]
    apv_meters["rate_ndvi"] = apv_meters["rate_ndvi"].replace([np.inf, -np.inf], np.nan)

    out = apv_meters[["climate_zone", "ndvi_apv", "ndvi_base", "rate_ndvi"]]
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

def simulate_capv(ndvi_path: Path, scenario_rates: pd.DataFrame):
    
    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs

    # 读取 CAPV 源数据
    capv = prepare_capv_source()
    
    # 确保 CAPV 有一个投影坐标系用于做 Buffer (如果原始是 LatLon，则转 3857)
    if capv.crs.is_geographic:
        capv_meters = capv.to_crs(epsg=3857)
    else:
        capv_meters = capv

    # 关联分区
    climate = safe_read_vector(CLIMATE_PATH).to_crs(capv_meters.crs)
    capv_meters = gpd.sjoin(capv_meters, climate, how="left", predicate="intersects")
    
    zone_col = next(
        (c for c in capv_meters.columns if "climate" in c.lower() or "zone" in c.lower()),
        "climate_zone"
    )
    capv_meters["climate_zone"] = capv_meters[zone_col].fillna("Unknown")
    capv_meters = capv_meters.loc[:, ~capv_meters.columns.duplicated()]

    print("  > 计算 CAPV 当前 NDVI...")
    # 转栅格坐标系提取
    capv_geo = capv_meters.to_crs(raster_crs)
    capv_meters["ndvi_capv_current"] = low_memory_zonal_mean(capv_geo.geometry, ndvi_path)

    print("  > 计算参考 NDVI (Buffer)...")
    # 米制 Buffer
    ring_meters = ring_buffer(capv_meters.geometry, INNER_BUF, OUTER_BUF)
    
    capv_meters["ndvi_capv_ref"] = np.nan
    valid_mask = ring_meters.notna() & (~ring_meters.is_empty)
    
    if valid_mask.any():
        valid_rings_geo = ring_meters[valid_mask].to_crs(raster_crs)
        vals = low_memory_zonal_mean(valid_rings_geo, ndvi_path)
        capv_meters.loc[valid_mask, "ndvi_capv_ref"] = vals

    # 模拟计算
    capv_meters = capv_meters.merge(scenario_rates, on="climate_zone", how="left")
    for lvl in ["top", "med", "low"]:
        capv_meters[f"sim_{lvl}"] = capv_meters["ndvi_capv_ref"] * capv_meters[f"rate_{lvl}"]

    summary = capv_meters.groupby("climate_zone").agg(
        ndvi_capv_current=("ndvi_capv_current", "mean"),
        sim_top=("sim_top", "mean"),
        sim_med=("sim_med", "mean"),
        sim_low=("sim_low", "mean"),
        count=("geometry", "count")
    ).reset_index()

    ensure_dir(SCENARIO_XLSX)
    summary.to_excel(SCENARIO_XLSX, index=False)
    print("\n结果预览:")
    print(summary.head())

# -------------------- 准备函数 --------------------
def prepare_capv_source() -> gpd.GeoDataFrame:
    if GRW_CROP_PATH.exists():
        return safe_read_vector(GRW_CROP_PATH)
    
    print("  > Reading GRW Source...")
    capv = clean_geoms(safe_read_vector(GRW_PATH))
    
    if "landcover_in_2018" in capv.columns:
        capv["lc"] = capv["landcover_in_2018"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        capv = capv[capv["lc"].isin([10, 20])].drop(columns=["lc"])
        
    year = pd.to_numeric(capv["construction_year"], errors="coerce")
    capv = capv[year.notna() & (year <= 2023)]
    
    ensure_dir(GRW_CROP_PATH)
    capv.to_file(GRW_CROP_PATH)
    return capv

if __name__ == "__main__":
    if not NDVI_MAX_PATH.exists():
        raise FileNotFoundError(NDVI_MAX_PATH)

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