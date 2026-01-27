# ============================================================
# 市级面积统计（来源：GRW GPKG）
# 条件：construction_year <= 2023
# 输出：area_pv2023.csv
# ============================================================

from pathlib import Path
import warnings

import fiona
import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

# ================= 配置 =================
CONFIG = {
    "gpkg_path": Path(r"D:\pv\grw_microsoft\grw_2024q2_China_only.gpkg"),
    "city_path": Path(r"D:\矢量地图\2023行政区划\市.shp"),
    "calc_crs": "EPSG:32648",
    "output_dir": Path(r"D:\pv\result"),
}

# ================= 工具函数 =================
def ensure_output_dir():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def load_grw_filtered() -> gpd.GeoDataFrame:
    """
    读取 GPKG，筛选 construction_year <= 2023
    """
    gpkg_path = CONFIG["gpkg_path"]
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GPKG 不存在：{gpkg_path}")

    layers = fiona.listlayers(gpkg_path)
    if not layers:
        raise RuntimeError("GPKG 中未发现图层")

    layer = layers[0]
    print(f"读取 GPKG 图层：{layer}")

    gdf = gpd.read_file(gpkg_path, layer=layer)

    if gdf.crs != CONFIG["calc_crs"]:
        gdf = gdf.to_crs(CONFIG["calc_crs"])

    # 解析 construction_year
    gdf["year_parsed"] = pd.to_numeric(gdf["construction_year"], errors="coerce")

    # 若是字符串/日期格式，二次解析
    if gdf["year_parsed"].isna().mean() > 0.5:
        gdf["year_parsed"] = pd.to_datetime(
            gdf["construction_year"], errors="coerce"
        ).dt.year

    filtered = gdf[
        (gdf["year_parsed"].notna()) & (gdf["year_parsed"] <= 2023)
    ].copy()

    filtered["area_m2"] = filtered.geometry.area

    print(f"筛选后要素数：{len(filtered)}")
    return filtered[["geometry", "area_m2"]]


def aggregate_city_area(data: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    按市级统计面积（km²）
    """
    print("正在进行市级面积统计...")

    city = gpd.read_file(CONFIG["city_path"]).to_crs(CONFIG["calc_crs"])
    city = city.reset_index().rename(columns={"index": "_city_id"})

    try:
        inter = gpd.overlay(
            data[["geometry"]],
            city[["_city_id", "geometry"]],
            how="intersection",
        )
        inter["area_m2"] = inter.geometry.area
        area_city = inter.groupby("_city_id")["area_m2"].sum()

    except Exception as exc:
        print(f"overlay 失败，使用代表点方式（原因：{exc}）")
        pts = data.copy()
        pts["geometry"] = pts.geometry.representative_point()

        joined = gpd.sjoin(
            pts,
            city[["_city_id", "geometry"]],
            how="left",
            predicate="within",
        )
        area_city = joined.groupby("_city_id")["area_m2"].sum()

    result = city.drop(columns=["geometry"]).merge(
        area_city.rename("area_pv2023_m2"),
        on="_city_id",
        how="left",
    )

    result["area_pv2023"] = result["area_pv2023_m2"].fillna(0.0) / 1_000_000
    return result.drop(columns=["_city_id", "area_pv2023_m2"])


# ================= 主流程 =================
def main():
    print("1/3 读取并筛选 GPKG（construction_year <= 2023）...")
    grw_gdf = load_grw_filtered()

    print("2/3 市级面积统计 ...")
    city_stats = aggregate_city_area(grw_gdf)

    print("3/3 输出 CSV ...")
    ensure_output_dir()
    out_csv = CONFIG["output_dir"] / "area_pv2023.csv"
    city_stats.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已输出：{out_csv}")
    print("完成！")


if __name__ == "__main__":
    main()
