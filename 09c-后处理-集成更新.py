#将所有导出的结果进行集成更新后处理，删除掉噪点
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import GeometryCollection

# 输入数据
PATH_2023 = r"D:\pv\result\qz\qz_merge\APV_qz_merge.shp"  # 待进行后处理的数据
PATH_2024 = r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg"  # 全国数据源；如有多个图层可在此指定；留空默认第一个
GPKG_LAYER = None
# 2022 历史数据（用于向前继承补全）
PATH_2022 = r"D:\pv\realpv\real_agripv.shp"

# 读 2024 时用 4326 bbox 裁剪，减少加载量
BBOX_CRS = "EPSG:4326"
# 输出目录：与 PATH_2023 同目录
OUTPUT_DIR = os.path.dirname(PATH_2023)
# 输出文件名
OUTPUT_NAME = "APV_qz_merge_2023_postprocess.shp" #需要修改，后处理文件名称


def safe_read(path: str, **kwargs) -> gpd.GeoDataFrame:
    """尝试多种编码读取 Shapefile/GPKG。"""
    for enc in (None, "utf-8", "gbk", "gb18030"):
        try:
            return gpd.read_file(path, encoding=enc, **kwargs) if enc else gpd.read_file(path, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"无法读取文件: {path}")


def clean_geoms(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    # 过滤空/无效几何，必要时 buffer(0) 修复
    mask_valid = (~gdf.geometry.is_empty) & gdf.geometry.notna()
    gdf = gdf[mask_valid].copy()
    if gdf.crs is None:
        raise ValueError(f"[{name}] 缺少 CRS")
    invalid = ~gdf.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf


def read_2024_with_bbox(bounds_4326, layer=None) -> gpd.GeoDataFrame:
    """优先用 bbox 读取全球数据，若为空则回退全量读取。"""
    bbox_tuple = tuple(map(float, bounds_4326)) if bounds_4326 is not None else None
    gdf_24 = gpd.read_file(PATH_2024, layer=layer, bbox=bbox_tuple)
    if len(gdf_24) == 0:
        print("[warn] bbox 读取为空，回退全量读取（可能较慢）")
        gdf_24 = gpd.read_file(PATH_2024, layer=layer)
    return gdf_24


def read_2022_with_bbox(bounds_4326) -> gpd.GeoDataFrame:
    """用 2023 边界的 bbox 读取 2022 历史数据，避免全量加载。"""
    bbox_tuple = tuple(map(float, bounds_4326)) if bounds_4326 is not None else None
    gdf_22 = gpd.read_file(PATH_2022, bbox=bbox_tuple)
    if len(gdf_22) == 0:
        print("[warn] 2022 bbox 读取为空，回退全量读取")
        gdf_22 = gpd.read_file(PATH_2022)
    return gdf_22


def main():
    # 1) 读取 2023
    gdf_23 = safe_read(PATH_2023)
    gdf_23 = clean_geoms(gdf_23, "2023")
    print(f"[info] 2023 面数量 {len(gdf_23)}")

    # 2) 计算 4326 bbox，用于裁剪 2024
    bbox_4326 = gdf_23.to_crs(BBOX_CRS).total_bounds

    # 3) 读取 2024（带 bbox）
    gdf_24 = read_2024_with_bbox(bbox_4326, layer=GPKG_LAYER)
    gdf_24 = clean_geoms(gdf_24, "2024")
    print(f"[info] 2024（bbox 范围内）面数量 {len(gdf_24)}")

    # 4) 读取 2022 历史数据（用于补全）
    gdf_22 = read_2022_with_bbox(bbox_4326)
    gdf_22 = clean_geoms(gdf_22, "2022")
    print(f"[info] 2022（bbox 范围内）面数量 {len(gdf_22)}")

    # 5) CRS 对齐
    if gdf_24.crs != gdf_23.crs:
        gdf_24 = gdf_24.to_crs(gdf_23.crs)
    if gdf_22.crs != gdf_23.crs:
        gdf_22 = gdf_22.to_crs(gdf_23.crs)

    # 6) 空间相交 + 裁剪：取 2024 交集后对 2023 进行裁剪（剔除误判，并把面裁到 2024 范围内）
    before = len(gdf_23)

    # 6.1 先用 sjoin 找到与 2024 相交的 2023 候选，避免对全量做 overlay 造成更大开销
    joined_idx = (
        gdf_23.sjoin(gdf_24[["geometry"]], how="inner", predicate="intersects")
        .index.unique()
        .to_numpy()
    )
    candidates = gdf_23.loc[joined_idx, :].copy()
    after_mask = len(candidates)
    removed = before - after_mask
    print(f"[mask] 处理前 {before}，相交候选 {after_mask}，删除 {removed}")

    # 6.2 用 overlay(intersection) 做“裁剪”：输出几何会被 2024 面边界截断（可能被切分成多个小面）
    # 注意：如果你发现输出为空或类型不对，请检查 2024 数据是否为面要素，以及 CRS 是否正确。
    clipped_23 = gpd.overlay(
        candidates,
        gdf_24[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    # overlay 可能产生空几何或 GeometryCollection，统一清理
    clipped_23 = clipped_23[clipped_23.geometry.notna() & ~clipped_23.geometry.is_empty].copy()
    clipped_23 = clipped_23[~clipped_23.geometry.apply(lambda g: isinstance(g, GeometryCollection))].copy()
    clipped_23 = clean_geoms(clipped_23, "2023_clipped")
    print(f"[clip] 裁剪后面数量 {len(clipped_23)}")

    # 7) 补全：向前继承 2022 确认的光伏板（防止假阴性缩减）
    base_cols = list(gdf_23.columns)
    gdf_22_aligned = gdf_22.copy()
    for col in base_cols:
        if col not in gdf_22_aligned.columns:
            gdf_22_aligned[col] = None
    gdf_22_aligned = gdf_22_aligned[base_cols]

    # 注意：不要用 geometry.to_wkb() 做 isin 去重。
    # 对于“栅格转面”的巨大面，WKB 极大，会触发 pandas/numpy 分配超大内存（例如 77GB）而报错。
    # 这里改用“空间反连接”（anti-join）：把与 joined 相交的 2022 面剔除，剩余的作为补回。
    if len(clipped_23) == 0:
        gdf_22_unique = gdf_22_aligned
    else:
        matched_idx = gpd.sjoin(
            gdf_22_aligned[["geometry"]],
            clipped_23[["geometry"]],
            how="inner",
            predicate="intersects",
        ).index.unique()
        gdf_22_unique = gdf_22_aligned.drop(index=matched_idx)
    added = len(gdf_22_unique)

    final_gdf = gpd.GeoDataFrame(
        pd.concat([clipped_23, gdf_22_unique], ignore_index=True),
        crs=gdf_23.crs,
    )
    final_count = len(final_gdf)

    print(f"[fill] 2022 补回: {added}")
    print(f"[result] 最终数量 {final_count}")

    # 8) 输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    final_gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8", index=False)
    print(f"[ok] 已输出: {out_path}")


if __name__ == "__main__":
    main()
