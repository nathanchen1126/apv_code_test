#处理脚本9c-后处理-集成更新.py的结果，让使得输出的结果中，值为0的面被删去，针对hz和hb两个只能用栅格导出的数据
import glob
import os

import geopandas as gpd

# 源 Shapefile 路径（原地覆盖）
SRC_PATH = r"D:\pv\result\hz\APV_hz_2023_postprocess.shp"  # 需要修改的文件


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
    # 清理多余的临时组件
    for extra in glob.glob(tmp_base + ".*"):
        if os.path.exists(extra):
            os.remove(extra)


def pick_field(gdf: gpd.GeoDataFrame) -> str:
    """优先使用 class 字段，没有则回退 value 字段。"""
    if "class" in gdf.columns:
        return "class"
    if "value" in gdf.columns:
        return "value"
    raise RuntimeError("缺少 class/value 字段，无法过滤 0 值")


def main() -> None:
    gdf = gpd.read_file(SRC_PATH)
    field = pick_field(gdf)

    before = len(gdf)
    filtered = gdf[gdf[field] != 0].copy()
    removed = before - len(filtered)
    if filtered.empty:
        raise RuntimeError("过滤后无数据，已取消写入以避免覆盖原始文件")

    tmp_path = SRC_PATH.replace(".shp", "_filtered_tmp.shp")
    tmp_base = os.path.splitext(tmp_path)[0]
    target_base = os.path.splitext(SRC_PATH)[0]

    # 清理旧的临时文件
    for f in glob.glob(tmp_base + ".*"):
        os.remove(f)

    filtered.to_file(tmp_path, driver="ESRI Shapefile", encoding="utf-8", index=False)
    replace_shapefile(tmp_base, target_base)

    print(
        f"[ok] 字段 {field}: 已删除 {removed} 个值为 0 的面，剩余 {len(filtered)} 个；已覆盖写回 {SRC_PATH}"
    )


if __name__ == "__main__":
    main()
