# ============================================================
# NDVI 2023：12个月栅格 -> 年最大值合成 (MVC) -> 市级统计
# 算法：取12个月中的最大值，然后统计各市的区域均值
# 输出：ndvi_max2023.csv
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr, osr

warnings.filterwarnings("ignore")
gdal.UseExceptions()

CONFIG = {
    "ndvi_dir": Path(r"D:\pv\data\ndvi"),
    "city_path": Path(r"D:\矢量地图\2023行政区划\市.shp"),
    "output_dir": Path(r"D:\pv\result"),
    "out_csv": "ndvi_max2023.csv",  # 输出文件名已更改
}

# =========================================================================
# 【关键配置】基于之前的调试，最大值约为10000，必须使用 0.0001 进行缩放
# =========================================================================
MANUAL_SCALE = 0.0001


def ensure_output_dir():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def list_ndvi_files():
    files = sorted(CONFIG["ndvi_dir"].glob("*.tif"))
    if len(files) != 12:
        raise RuntimeError(f"期望 12 个 NDVI tif，实际找到 {len(files)} 个：{CONFIG['ndvi_dir']}")
    return files


def open_raster(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开栅格：{path}")
    return ds


def read_band_as_float(ds, file_name="Unknown", print_debug=False):
    """
    读取波段，应用 Scale=0.0001，将 NoData 转为 NaN
    """
    band = ds.GetRasterBand(1)
    raw_arr = band.ReadAsArray()
    
    nodata = band.GetNoDataValue()
    scale = band.GetScale()
    offset = band.GetOffset()
    
    # 转换为浮点
    arr = raw_arr.astype(np.float32)

    # 处理 NoData -> NaN
    if nodata is not None:
        is_nodata = np.isclose(arr, nodata) if isinstance(nodata, float) else (arr == nodata)
        arr[is_nodata] = np.nan

    # 应用缩放 (优先使用手动配置 0.0001)
    current_scale = 1.0
    current_offset = 0.0

    if MANUAL_SCALE is not None:
        current_scale = MANUAL_SCALE
    else:
        if scale is not None: current_scale = scale
        if offset is not None: current_offset = offset

    if current_scale != 1.0 or current_offset != 0.0:
        arr = arr * current_scale + current_offset

    # 打印第一个文件的调试信息
    if print_debug:
        print("\n" + "="*60)
        print(f" [DEBUG INFO] 正在读取第一个文件用于检查: {file_name}")
        print(f"  > 强制缩放因子: {current_scale}")
        valid_pixels = arr[~np.isnan(arr)]
        if valid_pixels.size > 0:
            print(f"  > 单月数据范围 (期望 -1 ~ 1):")
            print(f"    Min: {np.min(valid_pixels):.4f}")
            print(f"    Max: {np.max(valid_pixels):.4f}")
        print("="*60 + "\n")

    return arr


def calc_ndvi_max_year(ndvi_files):
    """
    计算年最大值合成 (MVC)
    使用 np.fmax，它会逐像元对比，且自动忽略 NaN
    例如：fmax(0.8, NaN) = 0.8
    """
    print("1/4 读取12个月数据并计算【最大值合成 (MVC)】...")

    # 1. 读取第一个月作为初始“最大值”矩阵
    ref_ds = open_raster(ndvi_files[0])
    max_arr = read_band_as_float(ref_ds, file_name=ndvi_files[0].name, print_debug=True)
    
    rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize

    # 2. 循环遍历剩余月份，更新最大值
    for i, p in enumerate(ndvi_files[1:], start=1):
        # print(f"    正在处理第 {i+1}/12 个月...") # 如果觉得刷屏可以注释掉
        ds = open_raster(p)
        
        # 尺寸检查
        if ds.RasterXSize != cols or ds.RasterYSize != rows:
            raise RuntimeError(f"尺寸不匹配：{p.name}")
            
        current_arr = read_band_as_float(ds)
        
        # 核心逻辑：逐像元取最大值 (忽略 NaN)
        # 如果 max_arr 是 NaN 而 current 是 0.5，结果为 0.5
        # 如果两者都是数值，取大的
        max_arr = np.fmax(max_arr, current_arr)

    return max_arr, ref_ds


def gdf_to_ogr_layer(gdf: gpd.GeoDataFrame):
    mem_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdf.crs.to_wkt())
    layer = mem_ds.CreateLayer("geom", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("_id", ogr.OFTInteger))
    for idx, row in gdf.iterrows():
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("_id", int(idx))
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None
    return mem_ds, layer


def bounds_to_window(ds, bounds):
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    if inv_gt is None: return None
    minx, miny, maxx, maxy = bounds
    px0, py0 = gdal.ApplyGeoTransform(inv_gt, minx, maxy)
    px1, py1 = gdal.ApplyGeoTransform(inv_gt, maxx, miny)
    x0 = int(np.floor(min(px0, px1)))
    x1 = int(np.ceil(max(px0, px1)))
    y0 = int(np.floor(min(py0, py1)))
    y1 = int(np.ceil(max(py0, py1)))
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(ds.RasterXSize, x1)
    y1c = min(ds.RasterYSize, y1)
    if (x1c - x0c) <= 0 or (y1c - y0c) <= 0: return None
    return x0c, y0c, x1c - x0c, y1c - y0c


def city_stats_ndvi(city_gdf, raster_arr, ref_ds):
    """
    统计市级均值（基于最大值合成后的栅格）
    注意：这里计算的是“该市所有像元(年最大值)的空间平均值”
    """
    print("2/4 市级统计 (计算各市年最大NDVI的空间均值) ...")
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    results = []

    for idx, row in city_gdf.iterrows():
        win = bounds_to_window(ref_ds, row.geometry.bounds)
        if win is None:
            results.append(np.nan)
            continue
        xoff, yoff, xsize, ysize = win
        data = raster_arr[yoff:yoff + ysize, xoff:xoff + xsize]
        if data.size == 0:
            results.append(np.nan)
            continue

        mem_drv = gdal.GetDriverByName("MEM")
        mask_ds = mem_drv.Create("", xsize, ysize, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform((gt[0] + xoff * gt[1], gt[1], 0, gt[3] + yoff * gt[5], 0, gt[5]))
        mask_ds.SetProjection(proj)
        tmp = city_gdf.loc[[idx]]
        mem_vec, layer = gdf_to_ogr_layer(tmp)
        gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
        mask = mask_ds.GetRasterBand(1).ReadAsArray()

        valid = (mask == 1) & np.isfinite(data)
        if np.count_nonzero(valid) == 0:
            results.append(np.nan)
        else:
            # 这里的 nanmean 意思是：计算该城市范围内，所有像元“年最大值”的平均水平
            results.append(float(np.nanmean(data[valid])))

    city_gdf["ndvi_max2023"] = results
    return city_gdf.drop(columns=["geometry"])


def main():
    ndvi_files = list_ndvi_files()

    print("0/4 读取市级行政区 ...")
    city = gpd.read_file(CONFIG["city_path"]).reset_index(drop=True)
    ref_ds = open_raster(ndvi_files[0])
    srs = osr.SpatialReference(wkt=ref_ds.GetProjection())
    city = city.to_crs(srs.ExportToWkt())

    # 计算最大值合成
    max_arr, ref_ds = calc_ndvi_max_year(ndvi_files)
    
    # 全局统计检查
    valid_pixels = max_arr[np.isfinite(max_arr)]
    if valid_pixels.size > 0:
        print(f"\n[检查] 2023年【最大值合成】栅格范围:")
        print(f"  Min={valid_pixels.min():.4f}, Max={valid_pixels.max():.4f}, Mean={valid_pixels.mean():.4f}")
    
    # 区域统计
    result = city_stats_ndvi(city, max_arr, ref_ds)

    print("3/4 结果范围快速检查 ...")
    v = result["ndvi_max2023"].dropna()
    if len(v) > 0:
        print(f"  > 市级统计结果: min={v.min():.4f}, max={v.max():.4f}")
    else:
        print("  > 异常: 结果全为空")

    print("4/4 输出 CSV ...")
    ensure_output_dir()
    out_csv = CONFIG["output_dir"] / CONFIG["out_csv"]
    result.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已输出：{out_csv}")

if __name__ == "__main__":
    main()