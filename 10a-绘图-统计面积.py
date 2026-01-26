# 把后处理的结果进行合并，输出到原始的文件夹中，并分省，分市统计
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
CONFIG = {
    # 输入/输出文件夹配置
    "apv_dir": Path(r"D:\pv\result\result_postprocess"),
    "province_path": Path(r"D:\矢量地图\2023行政区划\省.shp"),
    "city_path": Path(r"D:\矢量地图\2023行政区划\市.shp"),
    "county_path": Path(r"D:\矢量地图\2023行政区划\县.shp"),
    
    # 输出文件名
    "output_shp_name": "APV_2023_all.shp",
    "output_dir": Path(r"D:\pv\result"),
    
    # 计算用投影 (UTM Zone 48N, 适合中国区域面积统计)
    "calc_crs": "EPSG:32648",

    # 面积精确分配：使用相交切分(更准确但更慢)；False 时用代表点近似
    "use_intersection": True,
}

def _infer_name_field(gdf: gpd.GeoDataFrame) -> str:
    preferred = [
        "name",
        "NAME",
        "Name",
        "ADNAME",
        "行政区",
        "省",
        "市",
        "县",
    ]
    for col in preferred:
        if col in gdf.columns:
            return col

    for col in gdf.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_string_dtype(gdf[col]):
            return col

    raise ValueError("未找到行政区名称字段（例如 name/NAME/ADNAME）")

def _area_stats_by_boundary(
    apv_polygons_utm: gpd.GeoDataFrame,
    boundary_path: Path,
    level_name: str,
) -> pd.DataFrame:
    boundary_raw = gpd.read_file(boundary_path).to_crs(CONFIG["calc_crs"])
    id_col = "_feature_id"
    boundary = boundary_raw.reset_index().rename(columns={"index": id_col})
    boundary_geom = boundary[[id_col, "geometry"]].copy()
    boundary_attrs = boundary.drop(columns=["geometry"]).copy()

    area_by_feature = None
    if CONFIG.get("use_intersection", True):
        try:
            apv_for_overlay = apv_polygons_utm[["geometry"]].copy()
            inter = gpd.overlay(apv_for_overlay, boundary_geom, how="intersection")
            inter["apv_area_m2"] = inter.geometry.area
            area_by_feature = inter.groupby(id_col)["apv_area_m2"].sum()
        except Exception as e:
            print(f"提示：{level_name} 统计相交计算失败，将用代表点近似（原因：{e}）")

    if area_by_feature is None:
        apv_points = apv_polygons_utm[["geometry", "area_m2"]].copy()
        apv_points["geometry"] = apv_points.geometry.representative_point()
        joined = gpd.sjoin(apv_points, boundary_geom, how="left", predicate="within")
        area_by_feature = joined.groupby(id_col)["area_m2"].sum()
        area_by_feature = area_by_feature.rename("apv_area_m2")

    stats = boundary_attrs.merge(area_by_feature.reset_index(), on=id_col, how="left")
    stats["apv_area_m2"] = stats["apv_area_m2"].fillna(0.0)
    stats["apv_area_km2"] = stats["apv_area_m2"] / 1_000_000

    stats["apv_area_m2"] = stats["apv_area_m2"].round(2)
    stats["apv_area_km2"] = stats["apv_area_km2"].round(4)
    stats = stats.drop(columns=[id_col])
    return stats

def process_data():
    print("[1/5] 正在扫描并读取光伏 SHP 文件...")
    
    # 1. 获取所有 SHP 文件
    apv_files = list(CONFIG["apv_dir"].glob("*.shp"))
    
    # 排除掉可能已经存在的合并文件，防止重复读取
    apv_files = [f for f in apv_files if f.name != CONFIG["output_shp_name"]]
    
    if not apv_files:
        print("错误：未找到任何 .shp 文件！")
        return

    apv_dfs = []
    for f in apv_files:
        try:
            # 读取文件
            df = gpd.read_file(f)
            if not df.empty:
                # 仅保留几何列以节省内存 (如果需要保留其他属性，请注释掉下面这行)
                df = df[['geometry']]
                
                # 统一投影到 UTM 进行合并前的准备
                if df.crs != CONFIG["calc_crs"]:
                    df = df.to_crs(CONFIG["calc_crs"])
                apv_dfs.append(df)
        except Exception as e:
            print(f"  - 跳过损坏或无法读取的文件: {f.name} ({e})")

    if not apv_dfs:
        print("错误：没有读取到有效的矢量数据。")
        return

    # 2. 合并数据
    print(f"[2/5] 正在合并 {len(apv_dfs)} 个文件...")
    # 使用 UTM 投影合并
    apv_all = gpd.GeoDataFrame(pd.concat(apv_dfs, ignore_index=True), crs=CONFIG["calc_crs"])
    
    # 计算精确面积 (单位: 平方米)
    print("  - 计算图斑面积...")
    apv_all['area_m2'] = apv_all.geometry.area
    
    # 导出合并后的 SHP (转回 WGS84 经纬度以便通用，但保留面积字段)
    output_shp_path = CONFIG["apv_dir"] / CONFIG["output_shp_name"]
    print(f"  - 正在导出合并文件至: {output_shp_path}")
    try:
        # 导出时转为 EPSG:4326 (经纬度)
        apv_export = apv_all.to_crs("EPSG:4326")
        apv_export.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print("    -> 导出 SHP 完成！")
    except Exception as e:
        print(f"    -> 导出 SHP 失败: {e}")

    # 3. 分级统计（省/市/县）
    print("[3/5] 正在进行省/市/县面积统计...")
    try:
        province_stats = _area_stats_by_boundary(apv_all, CONFIG["province_path"], "省")
        city_stats = _area_stats_by_boundary(apv_all, CONFIG["city_path"], "市")
        county_stats = _area_stats_by_boundary(apv_all, CONFIG["county_path"], "县")

        # 4. 直接导出 CSV（每个行政级别一个文件）
        output_dir = CONFIG["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        province_csv = output_dir / "province_area_stats.csv"
        city_csv = output_dir / "city_area_stats.csv"
        county_csv = output_dir / "county_area_stats.csv"

        print(f"[4/5] 正在导出省/市/县统计 CSV 至: {output_dir}")
        province_stats.to_csv(province_csv, index=False, encoding="utf-8-sig")
        city_stats.to_csv(city_csv, index=False, encoding="utf-8-sig")
        county_stats.to_csv(county_csv, index=False, encoding="utf-8-sig")
        print(f"    -> 省：{province_csv}")
        print(f"    -> 市：{city_csv}")
        print(f"    -> 县：{county_csv}")

        print("\n行数校验：")
        print(f"  - 省：{len(province_stats)} 行")
        print(f"  - 市：{len(city_stats)} 行")
        print(f"  - 县：{len(county_stats)} 行")

    except Exception as e:
        print(f"统计过程出错: {e}")

if __name__ == "__main__":
    process_data()
