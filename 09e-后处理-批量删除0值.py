# -*- coding: utf-8 -*-
"""
批量删除 Shapefile 中字段值为 0 的面（原逻辑参考：09d-后处理-栅格转面后修改.py）。

适用场景：
- 后处理结果里存在 `class` 或 `value` 字段，且部分面值为 0 需要剔除。

默认行为：
- 原地覆盖写回（为安全起见，采用临时文件 + 组件替换）。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd


CONFIG = {
    # 需要批处理的目录
    "input_dir": Path(r"D:\pv\result\result_merge_postprocess"),
    # 匹配规则：按需改为 "APV_*_postprocess.shp" 等
    "pattern": "*.shp",
    # True: 原地覆盖写回；False: 另存为 *_no0.shp
    "overwrite": True,
    # 当筛选后为空：True 直接报错中断；False 跳过该文件不写入
    "fail_on_empty": True,
}


def replace_shapefile(tmp_base: str, target_base: str) -> None:
    """将临时 shapefile 组件覆盖到目标路径。"""
    exts = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".shp.xml")
    for ext in exts:
        tmp_file = tmp_base + ext
        if not os.path.exists(tmp_file):
            continue
        target_file = target_base + ext
        if os.path.exists(target_file):
            os.remove(target_file)
        os.replace(tmp_file, target_file)
    for extra in glob.glob(tmp_base + ".*"):
        if os.path.exists(extra):
            os.remove(extra)


def pick_field(gdf: gpd.GeoDataFrame) -> str:
    """
    优先使用 class 字段，没有则回退 value 字段。
    如果报错，请检查 shp 文件的属性表字段名是否为 class/value（或自行在此处扩展）。
    """
    if "class" in gdf.columns:
        return "class"
    if "value" in gdf.columns:
        return "value"
    raise RuntimeError("缺少 class/value 字段，无法过滤 0 值")


def iter_shapefiles(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def filter_zero(gdf: gpd.GeoDataFrame, field: str) -> gpd.GeoDataFrame:
    vals = pd.to_numeric(gdf[field], errors="coerce")
    # 仅删除明确为 0 的记录；NaN 保留（避免类型/空值导致误删）
    keep = vals.isna() | (vals != 0)
    return gdf.loc[keep].copy()


def process_one(path: Path, overwrite: bool, fail_on_empty: bool) -> Optional[Path]:
    gdf = gpd.read_file(path)
    field = pick_field(gdf)

    before = len(gdf)
    filtered = filter_zero(gdf, field)
    removed = before - len(filtered)

    if removed == 0:
        print(f"[skip] {path.name}: 字段 {field} 无 0 值，跳过")
        return None

    if filtered.empty:
        msg = f"[fail] {path.name}: 字段 {field} 过滤后无数据"
        if fail_on_empty:
            raise RuntimeError(msg)
        print(msg + "，已跳过写入以避免覆盖原始文件")
        return None

    if overwrite:
        tmp_path = path.with_name(path.stem + "_filtered_tmp.shp")
        tmp_base = os.path.splitext(str(tmp_path))[0]
        target_base = os.path.splitext(str(path))[0]
        for f in glob.glob(tmp_base + ".*"):
            os.remove(f)
        filtered.to_file(tmp_path, driver="ESRI Shapefile", encoding="utf-8", index=False)
        replace_shapefile(tmp_base, target_base)
        print(f"[ok] {path.name}: 字段 {field} 删除 0 值 {removed} 条，剩余 {len(filtered)} 条（已覆盖）")
        return path

    out_path = path.with_name(path.stem + "_no0.shp")
    filtered.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8", index=False)
    print(f"[ok] {path.name}: 字段 {field} 删除 0 值 {removed} 条，输出 {out_path.name}")
    return out_path


def main() -> None:
    input_dir: Path = CONFIG["input_dir"]
    pattern: str = CONFIG["pattern"]
    overwrite: bool = CONFIG["overwrite"]
    fail_on_empty: bool = CONFIG["fail_on_empty"]

    if not input_dir.exists():
        raise FileNotFoundError(f"目录不存在：{input_dir}")

    shp_files = list(iter_shapefiles(input_dir, pattern))
    if not shp_files:
        raise FileNotFoundError(f"未找到匹配的 shp：{input_dir} / {pattern}")

    print(f"[info] 输入目录：{input_dir}")
    print(f"[info] 匹配文件：{len(shp_files)} 个（pattern={pattern}）")
    print(f"[info] overwrite={overwrite}，fail_on_empty={fail_on_empty}")

    ok = 0
    skipped = 0
    for shp in shp_files:
        try:
            out = process_one(shp, overwrite=overwrite, fail_on_empty=fail_on_empty)
            if out is None:
                skipped += 1
            else:
                ok += 1
        except Exception as e:
            print(f"[error] {shp.name}: {e}")
            if fail_on_empty:
                raise

    print(f"[done] 写入 {ok} 个，跳过 {skipped} 个")


if __name__ == "__main__":
    main()
