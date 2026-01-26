# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd

# ------------------------ 可配置参数区（按需修改） ------------------------
ROOT_DIR = Path(r"D:\pv\data\osm")
OUTPUT_SHP = Path(r"D:\pv\data\national_osm_combined.shp")

TARGET_A_NAME = "gis_osm_water_a_free_1.shp"
TARGET_B_NAME = "gis_osm_buildings_a_free_1.shp"
TARGET_C_NAME = "gis_osm_landuse_a_free_1.shp"

CODES_A = {8200, 8221} 
CODES_B = {1500}
CODES_C = {7228, 7229}

AREA_THRESHOLD_M2 = 5000.0
STRATIFIED_TRIM_RATIO = 0.05

# 是否按 code 分层删除面积最大/最小的比例（例如 0.05 = 5%）
APPLY_STRATIFIED_TRIM = True

# 是否进行面简化（参考示例 simplify_polygon.py）
APPLY_SIMPLIFY = True
SIMPLIFY_TOLERANCE = 0.1

# 面积计算方式：
#   - "dynamic_utm" : 按要素所在经度动态分配 UTM 分带（推荐，全国范围更稳健）
#   - "epsg3857"    : 统一投影到 EPSG:3857（简单但面积会有一定畸变）
AREA_METHOD = "dynamic_utm"

# 是否在“单文件级别”先做面积过滤（强烈建议 True：能显著降低内存压力）
FILTER_AREA_PER_FILE = True

# 输出编码：推荐 UTF-8；如后续软件兼容性需要可改成 "GBK"
OUTPUT_ENCODING = "UTF-8"

# 是否尝试修复无效几何（可能会变慢；通常不必开）
FIX_INVALID_GEOMETRY = False
# ---------------------------------------------------------------------


# ------------------------ 尝试启用 pyogrio（更快） ------------------------
HAS_PYOGRIO = False
try:
    import pyogrio
    from pyogrio import read_dataframe as pg_read_dataframe
    from pyogrio import read_info as pg_read_info
    from pyogrio import write_dataframe as pg_write_dataframe
    HAS_PYOGRIO = True
except Exception:
    # pyogrio 不可用时，仍可用 geopandas(fiona) 运行，但会慢一些
    HAS_PYOGRIO = False
    pg_read_dataframe = None
    pg_read_info = None
    pg_write_dataframe = None
# ---------------------------------------------------------------------


def setup_logger(log_path: Path) -> logging.Logger:
    """同时输出到控制台与日志文件（UTF-8）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("osm_merge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def cleanup_shapefile(target_shp: Path, logger: logging.Logger) -> None:
    """
    为避免写入时被旧文件残留影响，删除同名 shp 相关的常见附属文件。
    """
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix", ".sbn", ".sbx"]
    for ext in exts:
        p = target_shp.with_suffix(ext)
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"删除旧文件失败（可能被占用）：{p}；原因：{e}")


def find_field_case_insensitive(fields: list[str], target: str) -> str | None:
    """在字段列表中做大小写不敏感匹配，返回真实字段名。"""
    t = target.lower()
    for f in fields:
        if str(f).lower() == t:
            return str(f)
    return None


def safe_read_shp(
    shp_path: Path,
    logger: logging.Logger,
    where: str | None,
    required_fields_lower: set[str],
) -> gpd.GeoDataFrame | None:
    """
    稳健读取：
    - 优先用 pyogrio.read_info 预检查字段（快，避免读全表后才发现缺字段）
    - 优先用 pyogrio.read_dataframe + columns + where（性能高）
    - 任何错误都不会中断：返回 None 并记录 warning
    """
    try:
        if HAS_PYOGRIO:
            # 1) 先读元信息，拿字段列表（更稳健）
            info = pg_read_info(shp_path)
            fields = [str(x) for x in info.get("fields", [])]

            # 2) 做大小写不敏感字段匹配
            code_field = find_field_case_insensitive(fields, "code")
            fclass_field = find_field_case_insensitive(fields, "fclass")

            if code_field is None or fclass_field is None:
                logger.warning(
                    f"[跳过] 缺少必要字段(code/fclass)：{shp_path}；"
                    f"实际字段：{fields}"
                )
                return None

            # 3) columns 必须包含 where 里引用的字段（否则可能返回空或报错）
            columns = [code_field, fclass_field]

            # 4) 真正读取（只读需要字段 + geometry）
            gdf = pg_read_dataframe(
                shp_path,
                columns=columns,
                where=where,
                read_geometry=True,
            )

            # 5) 标准化列名为 code/fclass
            rename_map = {}
            if code_field != "code":
                rename_map[code_field] = "code"
            if fclass_field != "fclass":
                rename_map[fclass_field] = "fclass"
            if rename_map:
                gdf = gdf.rename(columns=rename_map)

            return gdf

        # pyogrio 不可用：退回 geopandas.read_file（会读全字段，较慢）
        gdf = gpd.read_file(shp_path)

        # 字段检查（大小写不敏感）
        cols = list(gdf.columns)
        code_col = None
        fclass_col = None
        for c in cols:
            if str(c).lower() == "code":
                code_col = c
            if str(c).lower() == "fclass":
                fclass_col = c

        if code_col is None or fclass_col is None:
            logger.warning(f"[跳过] 缺少必要字段(code/fclass)：{shp_path}")
            return None

        # 标准化列名
        if code_col != "code" or fclass_col != "fclass":
            gdf = gdf.rename(columns={code_col: "code", fclass_col: "fclass"})

        return gdf

    except Exception as e:
        logger.warning(f"[跳过] 读取失败（可能文件损坏）：{shp_path}；原因：{e}")
        logger.debug(traceback.format_exc())
        return None


def ensure_crs_4326(gdf: gpd.GeoDataFrame, logger: logging.Logger, shp_path: Path) -> gpd.GeoDataFrame:
    """
    确保 CRS 为 EPSG:4326：
    - 若无 CRS：默认按 EPSG:4326 处理（并记录 warning）
    - 若有 CRS 且非 4326：尝试 to_crs 转换
    """
    try:
        if gdf.crs is None:
            logger.warning(f"[警告] 未声明 CRS，默认按 EPSG:4326：{shp_path}")
            return gdf.set_crs(epsg=4326)

        # 如果已经是 4326，就不转换
        epsg = None
        try:
            epsg = gdf.crs.to_epsg()
        except Exception:
            epsg = None

        if epsg == 4326:
            return gdf

        # 尝试转换到 4326
        return gdf.to_crs(epsg=4326)

    except Exception as e:
        # 转换失败时，最后兜底：强制覆盖为 4326（这可能不准确，但不中断）
        logger.warning(f"[警告] CRS 转换失败，强制视为 EPSG:4326：{shp_path}；原因：{e}")
        return gdf.set_crs(epsg=4326, allow_override=True)


def filter_basic(gdf: gpd.GeoDataFrame, codes: set[int], logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    基础过滤：
    - code 转数值
    - 仅保留 code 在 codes 中
    - 仅保留 Polygon/MultiPolygon
    - 删除空几何
    - 字段裁剪：code / fclass / geometry
    """
    # 删除空几何
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    # 仅保留面要素（Polygon/MultiPolygon）
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]

    # code 转数值并过滤
    gdf["code"] = pd.to_numeric(gdf["code"], errors="coerce")
    gdf = gdf.dropna(subset=["code"])
    gdf["code"] = gdf["code"].astype(np.int32)
    gdf = gdf[gdf["code"].isin(list(codes))]

    # 字段裁剪（确保只保留核心字段）
    keep_cols = ["code", "fclass", "geometry"]
    gdf = gdf[keep_cols].copy()

    return gdf


def fix_invalid_geom_if_needed(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    可选：修复无效几何（buffer(0)），只对无效记录做处理，避免全量很慢。
    """
    if not FIX_INVALID_GEOMETRY:
        return gdf

    try:
        invalid_mask = ~gdf.is_valid
        if invalid_mask.any():
            logger.info(f"发现无效几何 {int(invalid_mask.sum())} 条，尝试 buffer(0) 修复...")
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
            # 修复后再剔除空几何
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        return gdf
    except Exception as e:
        logger.warning(f"几何修复失败（将继续流程）：{e}")
        return gdf


def compute_area_m2_dynamic_utm(gdf_4326: gpd.GeoDataFrame, logger: logging.Logger) -> pd.Series:
    """
    动态 UTM 分带面积计算（单位：m²）
    原理：按每条要素的经度中心点计算 UTM zone，并分组投影计算 area。
    """
    # 用 bounds 中心点计算经纬度（避免 centroid 在经纬度坐标系上的警告）
    b = gdf_4326.geometry.bounds
    lon = (b["minx"] + b["maxx"]) / 2.0
    lat = (b["miny"] + b["maxy"]) / 2.0

    # UTM zone: 1~60
    zone = np.floor((lon + 180.0) / 6.0).astype(int) + 1

    # 北半球用 326xx，南半球用 327xx（中国默认北半球，但仍做通用处理）
    epsg_arr = np.where(lat >= 0, 32600 + zone, 32700 + zone)

    utm_epsg = pd.Series(epsg_arr, index=gdf_4326.index).astype(int)

    areas = pd.Series(0.0, index=gdf_4326.index, dtype="float64")

    # 按 epsg 分组投影计算面积
    groups = gdf_4326.groupby(utm_epsg)
    for epsg_code, idx in groups.groups.items():
        try:
            g_proj = gdf_4326.loc[idx].to_crs(epsg=epsg_code)
            areas.loc[idx] = g_proj.geometry.area.values
        except Exception as e:
            # 投影失败兜底：改用 EPSG:3857
            logger.warning(
                f"UTM EPSG:{epsg_code} 投影/面积计算失败，改用 EPSG:3857 兜底。原因：{e}"
            )
            try:
                g_proj = gdf_4326.loc[idx].to_crs(epsg=3857)
                areas.loc[idx] = g_proj.geometry.area.values
            except Exception as e2:
                logger.error(
                    f"EPSG:3857 兜底也失败，该分组面积视为 0，将被过滤。原因：{e2}"
                )
                areas.loc[idx] = 0.0

    return areas


def compute_area_m2_epsg3857(gdf_4326: gpd.GeoDataFrame, logger: logging.Logger) -> pd.Series:
    """统一投影 EPSG:3857 计算面积（m²），简单但全国范围会有面积畸变。"""
    try:
        g_proj = gdf_4326.to_crs(epsg=3857)
        return pd.Series(g_proj.geometry.area.values, index=gdf_4326.index, dtype="float64")
    except Exception as e:
        logger.error(f"EPSG:3857 面积计算失败，全部面积视为 0：{e}")
        return pd.Series(0.0, index=gdf_4326.index, dtype="float64")


def compute_area_series(gdf_4326: gpd.GeoDataFrame, logger: logging.Logger) -> pd.Series:
    """基于全局 AREA_METHOD 计算面积（m²）。"""
    if AREA_METHOD.lower() == "dynamic_utm":
        return compute_area_m2_dynamic_utm(gdf_4326, logger)
    if AREA_METHOD.lower() == "epsg3857":
        return compute_area_m2_epsg3857(gdf_4326, logger)
    logger.warning(f"未知 AREA_METHOD={AREA_METHOD}，默认使用 dynamic_utm")
    return compute_area_m2_dynamic_utm(gdf_4326, logger)


def area_filter(gdf_4326: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    """计算面积并按阈值过滤（< AREA_THRESHOLD_M2 删除）。"""
    if gdf_4326.empty:
        return gdf_4326

    areas = compute_area_series(gdf_4326, logger)
    keep_mask = areas.fillna(0.0) >= float(AREA_THRESHOLD_M2)
    filtered = gdf_4326.loc[keep_mask].copy()

    return filtered


def stratified_area_trim(gdf_4326: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    """分层面积筛选：每个 code 删除最大/最小比例。"""
    if gdf_4326.empty:
        return gdf_4326

    areas = compute_area_series(gdf_4326, logger).fillna(0.0)
    temp = gdf_4326.copy()
    temp["_area_m2"] = areas

    keep_indices = []
    total_removed = 0

    for code_val, sub in temp.groupby("code"):
        sub_area = sub["_area_m2"]
        lower_q = sub_area.quantile(STRATIFIED_TRIM_RATIO)
        upper_q = sub_area.quantile(1.0 - STRATIFIED_TRIM_RATIO)

        if lower_q >= upper_q:
            keep_indices.append(sub.index)
            continue

        mask = (sub_area > lower_q) & (sub_area < upper_q)
        keep_indices.append(sub.index[mask])
        total_removed += int((~mask).sum())

    if keep_indices:
        kept_idx = pd.Index(np.concatenate([idx.values for idx in keep_indices]))
        trimmed = temp.loc[kept_idx].drop(columns=["_area_m2"])
    else:
        trimmed = temp.iloc[0:0].drop(columns=["_area_m2"])

    logger.info(f"分层面积筛选完成：删除 {total_removed} 条，剩余 {len(trimmed)} 条")
    return trimmed


def simplify_polygon_gdf(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    """简化面几何，保留原属性。"""
    if gdf.empty:
        return gdf

    simplified = gdf.copy()
    simplified["geometry"] = simplified.geometry.simplify(SIMPLIFY_TOLERANCE)
    logger.info(f"简化面完成：tolerance={SIMPLIFY_TOLERANCE}")
    return simplified


def process_one_file(shp_path: Path, file_type: str, logger: logging.Logger) -> gpd.GeoDataFrame | None:
    """
    处理单个 shp 文件（读取 -> code过滤 -> 字段裁剪 -> 面要素过滤 -> CRS统一 -> 面积过滤）
    file_type: "A" or "B"
    """
    codes = CODES_A if file_type == "A" else (CODES_B if file_type == "B" else CODES_C)

    # where 子句（尽量在读取阶段就过滤，提高性能）
    # 注意：这里假设字段名为 code（常见 OSM shp 是 code）
    # 若字段实际大小写不同，pyogrio read_info 里会做重命名为 code 后再二次过滤。
    if file_type == "A":
        where = "code IN (8200, 8221)"
    elif file_type == "B":
        where = "code = 1500"
    else:
        where = "code IN (7228, 7229)"

    logger.info(f"读取并处理：{shp_path}")

    gdf = safe_read_shp(
        shp_path=shp_path,
        logger=logger,
        where=where if HAS_PYOGRIO else None,  # fiona 不支持 where，避免传参报错
        required_fields_lower={"code", "fclass"},
    )
    if gdf is None:
        return None

    # CRS 统一到 4326
    gdf = ensure_crs_4326(gdf, logger, shp_path)

    # 基础过滤（code/字段/几何类型）
    before = len(gdf)
    gdf = filter_basic(gdf, codes=codes, logger=logger)
    after_basic = len(gdf)
    logger.info(f"  - 基础过滤：{before} -> {after_basic} 条")

    if gdf.empty:
        logger.info("  - 过滤后为空，跳过该文件")
        return None

    # 可选：修复无效几何
    gdf = fix_invalid_geom_if_needed(gdf, logger)
    if gdf.empty:
        logger.info("  - 几何修复后为空，跳过该文件")
        return None

    # 面积过滤（按需在单文件阶段过滤）
    if FILTER_AREA_PER_FILE:
        before_area = len(gdf)
        gdf = area_filter(gdf, logger)
        after_area = len(gdf)
        logger.info(f"  - 面积过滤(>= {AREA_THRESHOLD_M2} m²)：{before_area} -> {after_area} 条")

        if gdf.empty:
            logger.info("  - 面积过滤后为空，跳过该文件")
            return None

    # 确保最终字段仅 code/fclass/geometry
    gdf = gdf[["code", "fclass", "geometry"]].copy()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    return gdf


def main() -> None:
    log_file = OUTPUT_SHP.with_suffix(".log")
    logger = setup_logger(log_file)

    logger.info("========== OSM 多源提取/合并/清洗 开始 ==========")
    logger.info(f"输入目录：{ROOT_DIR}")
    logger.info(f"输出文件：{OUTPUT_SHP}")
    logger.info(f"pyogrio 可用：{HAS_PYOGRIO}")
    logger.info(f"面积阈值：{AREA_THRESHOLD_M2} m²；面积方法：{AREA_METHOD}")
    logger.info(f"单文件先面积过滤：{FILTER_AREA_PER_FILE}")
    logger.info(f"输出编码：{OUTPUT_ENCODING}")
    logger.info(f"分层面积筛选：{APPLY_STRATIFIED_TRIM}（比例={STRATIFIED_TRIM_RATIO}）")
    logger.info(f"面简化：{APPLY_SIMPLIFY}（tolerance={SIMPLIFY_TOLERANCE}）")

    if not ROOT_DIR.exists():
        logger.error(f"输入目录不存在：{ROOT_DIR}")
        return

    # 递归查找目标文件
    a_files = sorted(ROOT_DIR.rglob(TARGET_A_NAME))
    b_files = sorted(ROOT_DIR.rglob(TARGET_B_NAME))
    c_files = sorted(ROOT_DIR.rglob(TARGET_C_NAME))

    logger.info(f"扫描完成：发现 A 文件 {len(a_files)} 个，B 文件 {len(b_files)} 个，C 文件 {len(c_files)} 个")

    if len(a_files) == 0 and len(b_files) == 0 and len(c_files) == 0:
        logger.warning("未发现任何目标 shp 文件，任务结束。")
        return

    results: list[gpd.GeoDataFrame] = []

    # 处理 A 文件
    for p in a_files:
        try:
            gdf = process_one_file(p, file_type="A", logger=logger)
            if gdf is not None and not gdf.empty:
                results.append(gdf)
        except Exception as e:
            logger.warning(f"[跳过] 处理失败：{p}；原因：{e}")
            logger.debug(traceback.format_exc())

    # 处理 B 文件
    for p in b_files:
        try:
            gdf = process_one_file(p, file_type="B", logger=logger)
            if gdf is not None and not gdf.empty:
                results.append(gdf)
        except Exception as e:
            logger.warning(f"[跳过] 处理失败：{p}；原因：{e}")
            logger.debug(traceback.format_exc())

    # 处理 C 文件
    for p in c_files:
        try:
            gdf = process_one_file(p, file_type="C", logger=logger)
            if gdf is not None and not gdf.empty:
                results.append(gdf)
        except Exception as e:
            logger.warning(f"[跳过] 处理失败：{p}；原因：{e}")
            logger.debug(traceback.format_exc())

    if not results:
        logger.warning("所有文件处理后均无有效要素，任务结束。")
        return

    # 跨省/跨图层合并（一次性 concat，避免循环 append）
    logger.info("开始合并所有省份结果（pd.concat 一次性合并）...")
    combined = gpd.GeoDataFrame(
        pd.concat(results, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    logger.info(f"合并完成：总要素数 = {len(combined)} 条")

    # 若未在单文件阶段做面积过滤，则在全局阶段做一次
    if not FILTER_AREA_PER_FILE:
        logger.info("开始全局面积过滤...")
        before = len(combined)
        combined = area_filter(combined, logger)
        after = len(combined)
        logger.info(f"全局面积过滤完成：{before} -> {after} 条")
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # 再次确保 CRS=4326（坐标还原）
    combined = ensure_crs_4326(combined, logger, OUTPUT_SHP)

    # 按 code 分层删除面积最大/最小比例
    if APPLY_STRATIFIED_TRIM:
        combined = stratified_area_trim(combined, logger)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # 简化面
    if APPLY_SIMPLIFY:
        combined = simplify_polygon_gdf(combined, logger)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # 导出
    logger.info("开始导出 Shapefile...")
    OUTPUT_SHP.parent.mkdir(parents=True, exist_ok=True)
    cleanup_shapefile(OUTPUT_SHP, logger)

    try:
        if HAS_PYOGRIO:
            # promote_to_multi=True：避免 Polygon / MultiPolygon 混写导致部分驱动不兼容
            pg_write_dataframe(
                combined,
                OUTPUT_SHP,
                driver="ESRI Shapefile",
                encoding=OUTPUT_ENCODING,
                promote_to_multi=True,
            )
        else:
            # fiona 写出
            combined.to_file(
                OUTPUT_SHP,
                driver="ESRI Shapefile",
                encoding=OUTPUT_ENCODING,
                index=False,
            )

        logger.info(f"保存成功：{OUTPUT_SHP}")
        logger.info("========== 全部流程完成 ==========")

    except Exception as e:
        logger.error(f"导出失败：{OUTPUT_SHP}；原因：{e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
