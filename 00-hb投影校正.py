"""
修复华北结果 Shapefile 的投影问题：过滤掉超出 UTM 50N 合法范围的要素，
并重投影到 WGS84，输出到 D:\\pv\\result\\hb 目录，便于上传 GEE。
"""

import os
import geopandas as gpd

SRC_PATH = r"D:\pv\result\result_postprocess\APV_hb_2023_postprocess.shp"
OUT_DIR = r"D:\pv\result\hb"
OUT_NAME = "APV_hb_2023_postprocess_wgs84.shp"  

# 期望源坐标系：UTM 50N
SRC_CRS = "EPSG:32650"
TARGET_CRS = "EPSG:4326"


def clean_geoms(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """去除空/无效几何，必要时 buffer(0) 修复。"""
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    invalid = ~gdf.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    return gdf


def filter_utm50_domain(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    使用质心过滤掉超出 UTM 50N 合法范围的要素。
    - 东向：0~1,000,000 m（负值或过大的值视为异常）
    - 北向：0~10,000,000 m
    """
    cent = gdf.geometry.centroid
    mask = cent.x.between(0, 1_000_000) & cent.y.between(0, 10_000_000)
    dropped = len(gdf) - mask.sum()
    if dropped > 0:
        print(f"[filter] 去掉超域要素 {dropped} 条")
    return gdf[mask].copy()


def main():
    gdf = gpd.read_file(SRC_PATH)
    print(f"[info] 读取完成，条数 {len(gdf)}, CRS={gdf.crs}")

    if gdf.crs is None:
        print("[warn] 源数据缺少 CRS，强制设为 UTM 50N")
        gdf.set_crs(SRC_CRS, inplace=True)
    elif gdf.crs != SRC_CRS:
        print(f"[warn] 源 CRS 为 {gdf.crs}，按 UTM 50N 处理并投影到 WGS84")

    gdf = clean_geoms(gdf, "src")
    gdf = filter_utm50_domain(gdf)

    gdf_wgs = gdf.to_crs(TARGET_CRS)
    print(f"[info] 重投影到 {TARGET_CRS}，条数 {len(gdf_wgs)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    driver = "GPKG" if out_path.lower().endswith(".gpkg") else "ESRI Shapefile"
    gdf_wgs.to_file(out_path, driver=driver, encoding="utf-8", index=False)
    print(f"[ok] 已保存: {out_path} (driver={driver})")


if __name__ == "__main__":
    main()
