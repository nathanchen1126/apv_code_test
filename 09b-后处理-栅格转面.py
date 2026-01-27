#将GEE导出的hz和hb两个较大地图的栅格，转面进行后处理。因为hz和hb不能直接输出shp
import argparse
import os
from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# 用户默认输入栅格路径
DEFAULT_RASTER = r"D:\pv\result\hz\apv_hz_block\apv_hz_raster_block.tif"


def parse_values(raw: Optional[List[float]]) -> Optional[List[float]]:
    if raw is None:
        return None
    return list(raw)


def raster_to_polygons(
    raster_path: str,
    include_values: Optional[Iterable[float]] = None,
    exclude_values: Optional[Iterable[float]] = None,
    min_area: float = 0.0,
) -> gpd.GeoDataFrame:
    """使用 rasterio.features.shapes 将栅格转为面。"""
    with rasterio.open(raster_path) as ds:
        band = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        crs = ds.crs

    mask = np.ones_like(band, dtype=bool)
    if nodata is not None:
        mask &= band != nodata
    if include_values is not None:
        mask &= np.isin(band, include_values)
    if exclude_values is not None:
        mask &= ~np.isin(band, exclude_values)

    records = []
    for geom, value in shapes(band, mask=mask, transform=transform):
        poly = shape(geom)
        if poly.is_empty:
            continue
        if min_area > 0 and poly.area < min_area:
            continue
        records.append({"value": value, "geometry": poly})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="栅格转面（输出 Shapefile），便于后续 9b 集成处理。"
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_RASTER,
        help="输入栅格路径（默认: %(default)s）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 Shapefile 路径。默认与输入同目录，文件名加 _polygon.shp 后缀。",
    )
    parser.add_argument(
        "--include",
        type=float,
        nargs="+",
        default=None,
        help="仅保留这些像元值（空格分隔）。",
    )
    parser.add_argument(
        "--exclude",
        type=float,
        nargs="+",
        default=None,
        help="剔除这些像元值（空格分隔）。",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="忽略面积小于该阈值的面（单位为栅格 CRS 的平面单位）。",
    )
    args = parser.parse_args()

    src = args.src
    out_path = args.out
    if out_path is None:
        base, _ = os.path.splitext(src)
        out_path = f"{base}_polygon.shp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    include_vals = parse_values(args.include)
    exclude_vals = parse_values(args.exclude)

    gdf = raster_to_polygons(
        src,
        include_values=include_vals,
        exclude_values=exclude_vals,
        min_area=args.min_area,
    )

    if gdf.empty:
        raise RuntimeError("未生成任何面，请检查 include/exclude/min-area 过滤条件。")

    gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[ok] 已写出 {len(gdf)} 个面要素 -> {out_path}")


if __name__ == "__main__":
    main()
