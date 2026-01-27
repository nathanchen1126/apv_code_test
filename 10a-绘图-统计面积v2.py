"""
根据面积统计提示词：合并 APV，统计 APV 与 GRW 面积到省/市/县，输出 CSV。
"""

from pathlib import Path
import warnings

import fiona
import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

# ================= 配置 =================
CONFIG = {
    "apv_dir": Path(r"D:\pv\result\result_postprocess"),
    "apv_merge_name": "apv_2023_all_merge.shp",
    "province_path": Path(r"D:\矢量地图\2023行政区划\省shp"),
    "city_path": Path(r"D:\矢量地图\2023行政区划\市shp"),
    "county_path": Path(r"D:\矢量地图\2023行政区划\县shp"),
    "calc_crs": "EPSG:32648",
    "gpkg_path": Path(r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg"),
    "output_dir": Path(r"D:\pv\result"),
}


# ================= 工具函数 =================
def ensure_output_dir() -> None:
    """确保输出目录存在"""
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def merge_apv_shps() -> gpd.GeoDataFrame:
    """合并后处理 APV SHP 到 apv_2023_all_merge.shp"""
    apv_dir = CONFIG["apv_dir"]
    merge_path = apv_dir / CONFIG["apv_merge_name"]

    # 找到所有 SHP（排除已合并文件以防重复）
    shp_files = [p for p in apv_dir.glob("*.shp") if p.name != merge_path.name]
    if not shp_files:
        raise FileNotFoundError(f"未在 {apv_dir} 找到后处理 SHP")

    gdfs: list[gpd.GeoDataFrame] = []
    for shp in shp_files:
        try:
            gdf = gpd.read_file(shp)
        except Exception as exc:
            print(f"提示：跳过文件 {shp.name} ({exc})")
            continue
        if gdf.empty:
            continue
        if gdf.crs is None:
            print(f"提示：{shp.name} 没有 CRS 定义，跳过")
            continue
        if gdf.crs != CONFIG["calc_crs"]:
            gdf = gdf.to_crs(CONFIG["calc_crs"])
        # 仅保留几何信息，避免属性不一致
        gdfs.append(gdf[["geometry"]])

    if not gdfs:
        raise RuntimeError("没有有效的 APV SHP 可供合并")

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=CONFIG["calc_crs"])
    merged["area_m2"] = merged.geometry.area
    merged.to_file(merge_path)
    print(f"已合并 APV 到: {merge_path}")
    return merged


def load_grw_filtered() -> gpd.GeoDataFrame:
    """读取 GPKG，筛选 landcover_10/20 且 construction_year<=2023"""
    gpkg_path = CONFIG["gpkg_path"]
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GPKG 不存在: {gpkg_path}")

    layers = fiona.listlayers(gpkg_path)
    if not layers:
        raise RuntimeError("GPKG 中没有图层")
    layer = layers[0]

    grw = gpd.read_file(gpkg_path, layer=layer)
    if grw.crs != CONFIG["calc_crs"]:
        grw = grw.to_crs(CONFIG["calc_crs"])

    # 提取 landcover 数字编码
    grw["landcover_code"] = grw["landcover_in_2018"].astype(str).str.extract(r"(\d+)").astype(float)

    # 解析年份
    grw["year_parsed"] = pd.to_numeric(grw["construction_year"], errors="coerce")
    if grw["year_parsed"].isna().sum() > len(grw) * 0.5:
        grw["year_parsed"] = pd.to_datetime(grw["construction_year"], errors="coerce").dt.year

    mask_lc = grw["landcover_code"].isin([10, 20])
    mask_year = (grw["year_parsed"] <= 2023) & grw["year_parsed"].notna()
    grw_filtered = grw[mask_lc & mask_year].copy()
    grw_filtered["area_m2"] = grw_filtered.geometry.area

    print(f"GRW 筛选后要素数：{len(grw_filtered)}")
    return grw_filtered[["geometry", "area_m2"]]


def _aggregate_area_by_boundary(
    data: gpd.GeoDataFrame, boundary_path: Path, area_label: str
) -> pd.DataFrame:
    """按 boundary 聚合 data 面积，优先相交，失败则用代表点"""
    boundary_raw = gpd.read_file(boundary_path).to_crs(CONFIG["calc_crs"])
    id_col = "_feature_id"
    boundary = boundary_raw.reset_index().rename(columns={"index": id_col})
    boundary_geom = boundary[[id_col, "geometry"]]
    boundary_attrs = boundary.drop(columns=["geometry"])

    area_by_feature = None
    try:
        inter = gpd.overlay(data[["geometry"]], boundary_geom, how="intersection")
        inter["area_m2_tmp"] = inter.geometry.area
        area_by_feature = inter.groupby(id_col)["area_m2_tmp"].sum()
    except Exception as exc:
        print(f"提示：相交计算失败，使用代表点（原因：{exc}）")
        data_points = data.copy()
        data_points["geometry"] = data_points.geometry.representative_point()
        joined = gpd.sjoin(data_points, boundary_geom, how="left", predicate="within")
        area_by_feature = joined.groupby(id_col)["area_m2"].sum()

    stats = boundary_attrs.merge(area_by_feature.rename(area_label), on=id_col, how="left")
    stats[area_label] = stats[area_label].fillna(0.0) / 1_000_000  # m2 -> km2
    return stats.drop(columns=[id_col])


def aggregate_all_levels(apv_gdf: gpd.GeoDataFrame, grw_gdf: gpd.GeoDataFrame):
    """生成省/市/县的 area_apv 与 area_grw"""
    results = {}
    for level, path_key in [("province", "province_path"), ("city", "city_path"), ("county", "county_path")]:
        print(f"正在统计 {level} ...")
        apv_stats = _aggregate_area_by_boundary(apv_gdf, CONFIG[path_key], "area_apv")
        grw_stats = _aggregate_area_by_boundary(grw_gdf, CONFIG[path_key], "area_grw")
        merge_keys = [c for c in apv_stats.columns if c != "area_apv"]
        merged = apv_stats.merge(grw_stats, on=merge_keys, how="left")
        merged["area_grw"] = merged["area_grw"].fillna(0.0)
        results[level] = merged
    return results


def export_csvs(stats_dict: dict):
    """输出 CSV，UTF-8 编码"""
    ensure_output_dir()
    for level, df in stats_dict.items():
        csv_path = CONFIG["output_dir"] / f"area_stats_{level}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"已输出 {level} CSV: {csv_path}")


# ================= 辅助查看 =================
def print_fields():
    """打印 GPKG 字段名"""
    gpkg_path = CONFIG["gpkg_path"]
    print(f"Reading: {gpkg_path}")
    layers = fiona.listlayers(gpkg_path)
    if not layers:
        print("No layers found.")
        return
    for layer in layers:
        with fiona.open(gpkg_path, layer=layer) as src:
            field_names = list(src.schema["properties"].keys())
            print(f"Layer: {layer}")
            print(f"Fields ({len(field_names)}): {field_names}")
            print("-" * 40)


def print_field_values(field_name: str):
    """打印目标字段的唯一值"""
    gpkg_path = CONFIG["gpkg_path"]
    layers = fiona.listlayers(gpkg_path)
    if not layers:
        print("No layers found.")
        return
    for layer in layers:
        with fiona.open(gpkg_path, layer=layer) as src:
            if field_name not in src.schema["properties"]:
                print(f"Layer: {layer} -> field not found")
                print("-" * 40)
                continue
            values = {feat["properties"].get(field_name) for feat in src}
            print(f"Layer: {layer}")
            print(f"Unique values ({len(values)}): {sorted(values)}")
            print("-" * 40)


# ================= 主流程 =================
def main():
    print("1/4 合并 APV SHP ...")
    apv_gdf = merge_apv_shps()

    print("2/4 筛选 GRW 并计算面积 ...")
    grw_gdf = load_grw_filtered()

    print("3/4 按行政区划统计 ...")
    stats_dict = aggregate_all_levels(apv_gdf, grw_gdf)

    print("4/4 输出 CSV ...")
    export_csvs(stats_dict)
    print("完成！")


if __name__ == "__main__":
    main()
