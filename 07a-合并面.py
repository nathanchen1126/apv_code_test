# -*- coding: utf-8 -*-
"""
合并全国 OSM / Greenhouse / Mine 面数据为一个 nega shapefile
"""

import os
import pandas as pd
import geopandas as gpd

# =========================
# 1) 路径配置
# =========================
OSM_PATH = r"D:\pv\data\national_osm_combined.shp"
GH_PATH  = r"D:\pv\data\greenhouse\greenhouse.shp"
MINE_PATH = r"D:\pv\data\mine_area\mine_area.shp"

OUT_PATH = r"D:\pv\data\national_nega_merged.shp"

TARGET_CRS = "EPSG:4326"
POLY_TYPES = {"Polygon", "MultiPolygon"}
FIX_INVALID_GEOMETRY = True


# =========================
# 2) 工具函数
# =========================
def safe_read(path: str) -> gpd.GeoDataFrame:
    for enc in (None, "utf-8", "gbk", "gb18030"):
        try:
            return gpd.read_file(path) if enc is None else gpd.read_file(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件：{path}")


def ensure_crs(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(f"[{name}] 缺少 CRS")
    if gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def clean_polygons(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0:
        return gdf

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(POLY_TYPES)].copy()

    if FIX_INVALID_GEOMETRY:
        try:
            invalid = ~gdf.is_valid
            if invalid.any():
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
                gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        except Exception:
            print(f"[WARN] [{name}] 几何修复失败，继续执行")

    return gdf


def standardize_fields(gdf: gpd.GeoDataFrame, name: str, code_value=None, default_fclass="") -> gpd.GeoDataFrame:
    gdf.columns = [c.lower() for c in gdf.columns]

    if code_value is not None:
        gdf["code"] = code_value
    elif "code" not in gdf.columns:
        raise KeyError(f"[{name}] 缺少字段 code")

    if "fclass" not in gdf.columns:
        gdf["fclass"] = default_fclass
    else:
        gdf["fclass"] = gdf["fclass"].fillna(default_fclass)

    gdf["code"] = pd.to_numeric(gdf["code"], errors="raise").astype("int32")

    return gdf[["code", "fclass", "geometry"]].copy()


# =========================
# 3) 主流程
# =========================
def main():
    print("========== 读取 OSM ==========")
    osm = safe_read(OSM_PATH)
    osm = ensure_crs(osm, "OSM")
    osm = clean_polygons(osm, "OSM")
    osm = standardize_fields(osm, "OSM")

    print(f"[INFO] OSM 要素数：{len(osm)}")

    print("\n========== 读取 Greenhouse ==========")
    gh = safe_read(GH_PATH)
    gh = ensure_crs(gh, "Greenhouse")
    gh = clean_polygons(gh, "Greenhouse")
    gh = standardize_fields(gh, "Greenhouse", code_value=1, default_fclass="greenhouse")

    print(f"[INFO] Greenhouse 要素数：{len(gh)}")

    print("\n========== 读取 Mine ==========")
    mine = safe_read(MINE_PATH)
    mine = ensure_crs(mine, "Mine")
    mine = clean_polygons(mine, "Mine")
    mine = standardize_fields(mine, "Mine", code_value=2, default_fclass="mine")

    print(f"[INFO] Mine 要素数：{len(mine)}")

    print("\n========== 合并并导出 ==========")
    merged = pd.concat([osm, gh, mine], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_file(OUT_PATH, driver="ESRI Shapefile", encoding="UTF-8", index=False)

    print(f"[OK] 合并完成：{OUT_PATH}")
    print(f"[INFO] 合并后总要素数：{len(merged)}")


if __name__ == "__main__":
    main()
