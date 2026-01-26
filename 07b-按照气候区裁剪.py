# -*- coding: utf-8 -*-
"""
基于气候区划，对【已合并好的全国 nega 数据】进行裁剪（Shapely 2.0 加速版）

- 输入：全国 nega.shp（已包含 OSM / GH / Mine）
- 输出：按 7 个气候区裁剪后的 {xx}_nega.shp

依赖：
  geopandas >= 0.13
  shapely >= 2.0
  rtree
"""

import os
import glob
import traceback
import geopandas as gpd
import shapely
from shapely import intersection


# =========================
# 1) 配置区
# =========================
NEGA_PATH = r"D:\pv\data\national_nega_merged.shp"
CLIMATE_DIR = r"D:\pv\data\Chinese_climate"
OUT_ROOT = r"D:\pv\data"

TARGET_CRS = "EPSG:4326"
FIX_INVALID_GEOMETRY = True
POLY_TYPES = {"Polygon", "MultiPolygon"}

MASK_SIMPLIFY_TOL = 0.0001

CLIMATE_ZONES = [
    {"fullname": "华北湿润半湿润暖温带", "keyword": "climate_hb", "xx": "hb", "out_folder": "nega_hb"},
    {"fullname": "西北荒漠地区",         "keyword": "climate_xb", "xx": "xb", "out_folder": "nega_xb"},
    {"fullname": "华中华南湿润亚热带",   "keyword": "climate_hz", "xx": "hz", "out_folder": "nega_hz"},
    {"fullname": "东北湿润半湿润温带",   "keyword": "climate_db", "xx": "db", "out_folder": "nega_db"},
    {"fullname": "内蒙草原地区",         "keyword": "climate_nm", "xx": "nm", "out_folder": "nega_nm"},
    {"fullname": "青藏高原",             "keyword": "climate_qz", "xx": "qz", "out_folder": "nega_qz"},
    {"fullname": "华南湿润热带地区",     "keyword": "climate_hn", "xx": "hn", "out_folder": "nega_hn"},
]


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
        invalid = ~gdf.is_valid
        if invalid.any():
            try:
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
                gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            except Exception:
                print(f"[WARN] [{name}] 几何修复失败，继续执行")

    return gdf


def dissolve_mask(mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = (
        mask.geometry.union_all()
        if hasattr(mask.geometry, "union_all")
        else mask.geometry.unary_union
    )
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=mask.crs)


def find_climate_shp(keyword: str) -> str:
    files = glob.glob(os.path.join(CLIMATE_DIR, f"*{keyword}*.shp"))
    return files[0] if files else ""


# =========================
# 3) 主流程（Shapely 2.0 + rtree）
# =========================
def main():
    print("========== 1) 读取全国 nega 数据 ==========")
    nega = safe_read(NEGA_PATH)
    nega.columns = [c.lower() for c in nega.columns]

    for col in ("code", "fclass"):
        if col not in nega.columns:
            raise KeyError(f"全国 nega 数据缺少字段：{col}")

    nega = ensure_crs(nega, "NEGA")
    nega = clean_polygons(nega, "NEGA")

    print(f"[INFO] 全国 nega 要素数：{len(nega)}")

    # 显式构建空间索引（rtree）
    sindex = nega.sindex
    print("[INFO] nega 空间索引已构建（rtree）")

    print("\n========== 2) 按气候区裁剪（Shapely 2.0） ==========")
    for z in CLIMATE_ZONES:
        xx = z["xx"]
        fullname = z["fullname"]
        keyword = z["keyword"]

        print(f"\n---- {fullname} ({xx}) ----")

        climate_path = find_climate_shp(keyword)
        if not climate_path:
            print(f"[WARN] 未找到气候区文件：{keyword}")
            continue

        try:
            climate = safe_read(climate_path)
            climate = ensure_crs(climate, f"Climate-{xx}")
            climate = clean_polygons(climate, f"Climate-{xx}")

            if len(climate) == 0:
                print(f"[WARN] {xx} 气候区为空")
                continue

            # 合并气候区 mask
            mask = dissolve_mask(climate)
            mask_geom = mask.geometry.values[0]
            if MASK_SIMPLIFY_TOL and MASK_SIMPLIFY_TOL > 0:
                mask_geom = mask_geom.simplify(MASK_SIMPLIFY_TOL, preserve_topology=True)

            # ===============================
            # 1) bbox + rtree 预筛选
            # ===============================
            minx, miny, maxx, maxy = mask.total_bounds
            candidate_idx = list(sindex.intersection((minx, miny, maxx, maxy)))
            nega_sub = nega.iloc[candidate_idx]

            print(f"[INFO] bbox 预筛选：{len(nega_sub)} / {len(nega)}")

            if len(nega_sub) == 0:
                print(f"[WARN] {xx} 区 bbox 内无 nega")
                continue

            # ===============================
            # 2) Shapely 2.0 向量化裁剪
            # ===============================
            inter = intersection(nega_sub.geometry.values, mask_geom)
            valid = ~(shapely.is_empty(inter) | shapely.is_missing(inter))

            if not valid.any():
                print(f"[WARN] {xx} 区无相交要素")
                continue

            nega_clip = nega_sub.loc[valid].copy()
            nega_clip["geometry"] = inter[valid]
            nega_clip = clean_polygons(nega_clip, f"{xx}-clip")

            if len(nega_clip) == 0:
                print(f"[WARN] {xx} 区结果为空，跳过输出")
                continue

            # ===============================
            # 3) 输出
            # ===============================
            out_dir = os.path.join(OUT_ROOT, z["out_folder"])
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{xx}_nega.shp")

            nega_clip.to_file(
                out_path,
                driver="ESRI Shapefile",
                encoding="UTF-8",
                index=False
            )

            print(f"[OK] 输出完成：{out_path} | 要素数={len(nega_clip)}")

        except Exception as e:
            print(f"[ERROR] {xx} 区处理失败：{e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
