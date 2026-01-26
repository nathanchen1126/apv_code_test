import geopandas as gpd

###每次需要修改climate！
CLIMATE_PATH = r"D:\pv\data\Chinese_climate\climate_qz.shp"
TARGET_PRE_PATH = r"D:\pv\result\result_postprocess\APV_2023_all.shp"
##每次需要修改后处理文件！
TARGET_POST_PATH = r"D:\pv\result\qz\qz_merge\APV_qz_merge_2023_postprocess.shp"

# Use a metric CRS for area calculations to avoid degree-based outputs.
def _pick_area_crs(climate_gdf: gpd.GeoDataFrame):
    if climate_gdf.crs:
        try:
            return climate_gdf.estimate_utm_crs()
        except Exception:
            pass
    return "EPSG:3857"


def _load_climate():
    climate = gpd.read_file(CLIMATE_PATH)
    if climate.crs is None:
        raise ValueError("气候区矢量缺少 CRS，无法进行面积计算。")
    climate = climate[climate.geometry.notnull()].copy()
    return climate


def _clip_and_area(target_path: str, climate: gpd.GeoDataFrame, area_crs):
    target = gpd.read_file(target_path)
    if target.crs is None:
        raise ValueError(f"{target_path} 缺少 CRS，无法进行面积计算。")
    target = target[target.geometry.notnull()].copy()
    if target.crs != climate.crs:
        target = target.to_crs(climate.crs)

    # Clip to keep only the part that falls inside the climate zones.
    clipped = gpd.clip(target, climate)
    if clipped.empty:
        return 0.0

    clipped = clipped.to_crs(area_crs)
    return float(clipped.area.sum())


def main():
    climate = _load_climate()
    area_crs = _pick_area_crs(climate)

    pre_area = _clip_and_area(TARGET_PRE_PATH, climate, area_crs)
    post_area = _clip_and_area(TARGET_POST_PATH, climate, area_crs)

    print(f"手动难例前面积: {pre_area / 1_000_000:.3f} 平方公里 ({pre_area:.0f} 平方米)")
    print(f"手动难例后面积: {post_area / 1_000_000:.3f} 平方公里 ({post_area:.0f} 平方米)")


if __name__ == "__main__":
    main()
