from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings
import re

# 过滤掉警告
warnings.filterwarnings("ignore")

def calculate_area(gpkg_path: Path, layer: str | None) -> None:
    try:
        import geopandas as gpd
        import pandas as pd
        import fiona
    except ImportError:
        print("Error: 请安装必要库: pip install geopandas fiona", file=sys.stderr)
        return

    # 1. 自动获取图层
    if layer is None:
        try:
            layers = fiona.listlayers(str(gpkg_path))
            if not layers:
                print("Error: GPKG 文件中没有图层", file=sys.stderr)
                return
            layer = layers[0]
        except Exception as e:
            print(f"Error: 无法读取图层列表: {e}", file=sys.stderr)
            return

    # 2. 读取数据 (跳过 geometry 提升速度)
    try:
        gdf = gpd.read_file(str(gpkg_path), layer=layer, ignore_geometry=True)
    except Exception as e:
        print(f"Error: 读取文件失败: {e}", file=sys.stderr)
        return

    required_cols = {"landcover_in_2018", "construction_year", "area"}
    if not required_cols.issubset(gdf.columns):
        print(f"Error: 缺少必要字段，现有字段: {list(gdf.columns)}", file=sys.stderr)
        return

    # --- 关键修复：处理 'landcover_XX' 格式 ---
    # 使用正则表达式提取字符串中的数字部分 (例如从 'landcover_200' 提取 '200')
    # .astype(str) 确保输入是字符串
    # .str.extract(r'(\d+)') 提取第一个连续数字
    # .astype(float) 转为浮点数以便比较
    gdf["landcover_code"] = gdf["landcover_in_2018"].astype(str).str.extract(r'(\d+)').astype(float)

    # --- 处理年份 ---
    # 处理 '2018' 或 '2018-01-01' 等格式
    # 先尝试直接转数字
    gdf["year_parsed"] = pd.to_numeric(gdf["construction_year"], errors="coerce")
    
    # 如果大部分转换失败，尝试日期解析
    if gdf["year_parsed"].isna().sum() > (len(gdf) * 0.5):
        gdf["year_parsed"] = pd.to_datetime(gdf["construction_year"], errors="coerce").dt.year

    # --- 处理面积 ---
    gdf["area_numeric"] = pd.to_numeric(gdf["area"], errors="coerce").fillna(0)

    # --- 筛选条件 ---
    # 目标代码: 10, 11, 12, 20
    target_codes = [10, 11, 12, 20]
    
    # 筛选1: Landcover (比较提取出的数字)
    mask_lc = gdf["landcover_code"].isin(target_codes)
    
    # 筛选2: Year <= 2023 (且不为空)
    mask_year = (gdf["year_parsed"] <= 2023) & (gdf["year_parsed"].notna())

    # --- 应用筛选并计算 ---
    filtered_gdf = gdf[mask_lc & mask_year]
    total_area = filtered_gdf["area_numeric"].sum()

    # --- 仅输出最终数字 ---
    print(f"{total_area:.2f}")

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpkg",
        default=r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg",
        help="GPKG 文件路径",
    )
    parser.add_argument("--layer", default=None)
    
    args = parser.parse_args(argv)
    gpkg_path = Path(args.gpkg)
    
    if not gpkg_path.exists():
        print(f"Error: 文件不存在: {gpkg_path}", file=sys.stderr)
        return 2

    calculate_area(gpkg_path, args.layer)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))