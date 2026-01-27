# ============================================================
# 市级耕地面积统计（GDAL 稳定版）
# CLCD 2023 | value == 1 | 30m
# 输出：cropland_area2023.csv
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr, osr

warnings.filterwarnings("ignore")
gdal.UseExceptions()

# ================= 配置 =================
CONFIG = {
    "raster_path": Path(r"D:\pv\data\clcd\CLCD_v01_2023_albert.tif"),
    "city_path": Path(r"D:\矢量地图\2023行政区划\市.shp"),
    "output_dir": Path(r"D:\pv\result"),
}

PIXEL_AREA_M2 = 30 * 30


# ================= 工具函数 =================
def ensure_output_dir():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def open_raster():
    ds = gdal.Open(str(CONFIG["raster_path"]), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("无法打开栅格")
    return ds


def gdf_to_ogr_layer(gdf: gpd.GeoDataFrame):
    """
    GeoDataFrame → OGR 内存图层
    """
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


# ================= 核心统计 =================
def calc_city_area(city_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    ds = open_raster()
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nodata = band.GetNoDataValue()

    raster_xsize = ds.RasterXSize
    raster_ysize = ds.RasterYSize

    results = []

    for idx, row in city_gdf.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds

        # ===== 计算原始窗口 =====
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        xsize = int((maxx - minx) / gt[1]) + 1
        ysize = int((miny - maxy) / gt[5]) + 1

        # ===== 裁剪到栅格范围（关键）=====
        xoff0 = max(0, xoff)
        yoff0 = max(0, yoff)

        xend = min(raster_xsize, xoff + xsize)
        yend = min(raster_ysize, yoff + ysize)

        xsize0 = xend - xoff0
        ysize0 = yend - yoff0

        if xsize0 <= 0 or ysize0 <= 0:
            results.append(0.0)
            continue

        # ===== 读取栅格 =====
        data = band.ReadAsArray(xoff0, yoff0, xsize0, ysize0)
        if data is None:
            results.append(0.0)
            continue

        if nodata is not None:
            data = np.where(data == nodata, 0, data)

        # ===== 构建 mask =====
        mem_drv = gdal.GetDriverByName("MEM")
        mask_ds = mem_drv.Create("", xsize0, ysize0, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform((
            gt[0] + xoff0 * gt[1],
            gt[1],
            0,
            gt[3] + yoff0 * gt[5],
            0,
            gt[5],
        ))
        mask_ds.SetProjection(proj)

        tmp_gdf = city_gdf.loc[[idx]]
        mem_vec, layer = gdf_to_ogr_layer(tmp_gdf)

        gdal.RasterizeLayer(
            mask_ds,
            [1],
            layer,
            burn_values=[1],
        )

        mask = mask_ds.GetRasterBand(1).ReadAsArray()

        # ===== 统计 value == 1 =====
        valid = (mask == 1) & (data == 1)
        pixel_count = int(np.count_nonzero(valid))

        area_km2 = pixel_count * PIXEL_AREA_M2 / 1_000_000
        results.append(area_km2)

    city_gdf["cropland_area2023"] = results
    return city_gdf.drop(columns=["geometry"])


# ================= 主流程 =================
def main():
    print("1/3 读取市级行政区 ...")
    city = gpd.read_file(CONFIG["city_path"])
    city = city.reset_index(drop=True)

    # 对齐到栅格 CRS
    with gdal.Open(str(CONFIG["raster_path"])) as ds:
        srs = osr.SpatialReference(wkt=ds.GetProjection())
        city = city.to_crs(srs.ExportToWkt())

    print("2/3 使用 GDAL 进行市级统计 ...")
    result = calc_city_area(city)

    print("3/3 输出 CSV ...")
    ensure_output_dir()
    out_csv = CONFIG["output_dir"] / "cropland_area2023.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已输出：{out_csv}")
    print("完成！")


if __name__ == "__main__":
    main()
