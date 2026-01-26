import os
import glob
import traceback
import geopandas as gpd
import shapely
from shapely import intersection

# 输入与配置
INPUT_PATH = r"D:\pv\data\solarpanel\2023\2023.shp"
CLIMATE_DIR = r"D:\pv\data\Chinese_climate"
OUT_ROOT = r"D:\pv\data\solarpanel"
TARGET_CRS = "EPSG:4326"
MASK_SIMPLIFY_TOL = 0.0001
POLY_TYPES = {"Polygon", "MultiPolygon"}

CLIMATE_ZONES = [
    {"fullname": "华北湿润半湿润温带", "keyword": "climate_hb", "xx": "hb"},
    {"fullname": "西北荒漠地区", "keyword": "climate_xb", "xx": "xb"},
    {"fullname": "华中华南湿润亚热带", "keyword": "climate_hz", "xx": "hz"},
    {"fullname": "东北湿润半湿润温带", "keyword": "climate_db", "xx": "db"},
    {"fullname": "内蒙古草原地区", "keyword": "climate_nm", "xx": "nm"},
    {"fullname": "青藏高原", "keyword": "climate_qz", "xx": "qz"},
    {"fullname": "华南湿润热带地区", "keyword": "climate_hn", "xx": "hn"},
]


def safe_read(path: str, **kwargs) -> gpd.GeoDataFrame:
    for enc in (None, "utf-8", "gbk", "gb18030"):
        try:
            return gpd.read_file(path, **kwargs) if enc is None else gpd.read_file(path, encoding=enc, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"无法读取文件: {path}")


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
    invalid = ~gdf.is_valid
    if invalid.any():
        try:
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        except Exception:
            print(f"[WARN] [{name}] 几何修复失败，继续处理")
    return gdf


def dissolve_mask(mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = mask.geometry.union_all() if hasattr(mask.geometry, "union_all") else mask.geometry.unary_union
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=mask.crs)


def find_climate_shp(keyword: str) -> str:
    files = glob.glob(os.path.join(CLIMATE_DIR, f"*{keyword}*.shp"))
    return files[0] if files else ""


def main():
    print("=== 读取待裁剪数据 ===")
    pv = safe_read(INPUT_PATH)
    pv = ensure_crs(pv, "PV2023")
    pv = clean_polygons(pv, "PV2023")
    if pv is None or len(pv) == 0:
        print("[ERROR] PV 数据为空")
        return
    print(f"[INFO] PV 面数量: {len(pv)}")

    sindex = pv.sindex

    for z in CLIMATE_ZONES:
        xx = z["xx"]
        fullname = z["fullname"]
        keyword = z["keyword"]
        print(f"\n--- {fullname} ({xx}) ---")

        climate_path = find_climate_shp(keyword)
        if not climate_path:
            print(f"[WARN] 未找到气候区文件: {keyword}")
            continue

        try:
            climate = safe_read(climate_path)
            climate = ensure_crs(climate, f"Climate-{xx}")
            climate = clean_polygons(climate, f"Climate-{xx}")
            if len(climate) == 0:
                print(f"[WARN] {xx} 气候区为空")
                continue

            mask = dissolve_mask(climate)
            mask_geom = mask.geometry.values[0]
            if MASK_SIMPLIFY_TOL and MASK_SIMPLIFY_TOL > 0:
                mask_geom = mask_geom.simplify(MASK_SIMPLIFY_TOL, preserve_topology=True)

            minx, miny, maxx, maxy = mask.total_bounds
            candidate_idx = list(sindex.intersection((minx, miny, maxx, maxy)))
            pv_sub = pv.iloc[candidate_idx]
            print(f"[INFO] bbox 预筛选: {len(pv_sub)} / {len(pv)}")
            if len(pv_sub) == 0:
                print(f"[WARN] {xx} 区无候选面")
                continue

            inter = intersection(pv_sub.geometry.values, mask_geom)
            valid = ~(shapely.is_empty(inter) | shapely.is_missing(inter))
            if not valid.any():
                print(f"[WARN] {xx} 区无相交要素")
                continue

            pv_clip = pv_sub.loc[valid].copy()
            pv_clip["geometry"] = inter[valid]
            pv_clip = clean_polygons(pv_clip, f"PV-{xx}-clip")
            if len(pv_clip) == 0:
                print(f"[WARN] {xx} 区裁剪后为空")
                continue

            os.makedirs(OUT_ROOT, exist_ok=True)
            out_path = os.path.join(OUT_ROOT, f"pv2023_{xx}.shp")
            pv_clip.to_file(out_path, driver="ESRI Shapefile", encoding="UTF-8", index=False)
            print(f"[OK] 输出: {out_path} | 面数量: {len(pv_clip)}")

        except Exception as e:
            print(f"[ERROR] {xx} 处理失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
