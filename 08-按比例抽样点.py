import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.prepared import prep

# --- 1. Zone config (suffix uses xx) ---
CLIMATE_ZONES = [
    {"xx": "hb", "out_folder": "nega_hb"},
    {"xx": "xb", "out_folder": "nega_xb"},
    {"xx": "hz", "out_folder": "nega_hz"},
    {"xx": "db", "out_folder": "nega_db"},
    {"xx": "nm", "out_folder": "nega_nm"},
    {"xx": "qz", "out_folder": "nega_qz"},
    {"xx": "hn", "out_folder": "nega_hn"},
]

# --- 2. Paths and parameters ---
BASE_DIR = r"D:\pv\data"
INPUT_TEMPLATE = "{xx}_nega.shp"  # change if your polygon input name differs
OUTPUT_TEMPLATE = "fake_{xx}_point.shp"

GRID_SIZE_M = 50_000  # 50 km
POINTS_PER_CODE_PER_GRID = 10
TARGET_CRS = "EPSG:3857"  # metric CRS for 50km grid; change if needed
POLYGON_SAMPLE_FRAC = 0.5


def safe_read(path: str) -> gpd.GeoDataFrame:
    for enc in (None, "utf-8", "gbk", "gb18030"):
        try:
            return gpd.read_file(path) if enc is None else gpd.read_file(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"failed to read: {path}")


def ensure_projected(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("missing CRS")
    if not gdf.crs.is_projected:
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def build_grid(bounds, cell_size: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    x_start = np.floor(minx / cell_size) * cell_size
    y_start = np.floor(miny / cell_size) * cell_size
    x_end = np.ceil(maxx / cell_size) * cell_size
    y_end = np.ceil(maxy / cell_size) * cell_size

    xs = np.arange(x_start, x_end, cell_size)
    ys = np.arange(y_start, y_end, cell_size)
    cells = []
    grid_id = 0
    for x in xs:
        for y in ys:
            cells.append({"grid_id": grid_id, "geometry": box(x, y, x + cell_size, y + cell_size)})
            grid_id += 1
    return gpd.GeoDataFrame(cells, geometry="geometry")


def sample_points_in_geom(geom, n: int, rng: np.random.Generator):
    if geom.is_empty or n <= 0:
        return []

    minx, miny, maxx, maxy = geom.bounds
    prepared = prep(geom)

    points = []
    max_attempts = max(n * 50, 200)
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if prepared.contains(p):
            points.append(p)
        attempts += 1

    return points


def process_single_zone(zone_info, seed=42):
    xx = zone_info["xx"]
    folder = zone_info["out_folder"]
    in_path = os.path.join(BASE_DIR, folder, INPUT_TEMPLATE.format(xx=xx))
    out_path = os.path.join(BASE_DIR, folder, OUTPUT_TEMPLATE.format(xx=xx))

    print(f"\n--- zone {xx} ---")
    if not os.path.exists(in_path):
        print(f"[skip] missing: {in_path}")
        return

    gdf = safe_read(in_path)
    gdf.columns = [c.lower() for c in gdf.columns]
    if "code" not in gdf.columns:
        print("[skip] missing field: code")
        return

    original_crs = gdf.crs
    gdf = ensure_projected(gdf)

    if 0 < POLYGON_SAMPLE_FRAC < 1:
        gdf = gdf.sample(frac=POLYGON_SAMPLE_FRAC, random_state=seed)
        print(f"[info] polygon sample: {len(gdf)}")

    grid = build_grid(gdf.total_bounds, GRID_SIZE_M)
    grid.set_crs(gdf.crs, inplace=True)

    # Clip polygons by grid cells for per-grid sampling
    inter = gpd.overlay(
        gdf[["code", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="intersection",
    )
    if len(inter) == 0:
        print("[skip] no intersections")
        return

    rng = np.random.default_rng(seed)
    records = []
    for (grid_id, code), sub in inter.groupby(["grid_id", "code"]):
        geom = sub.geometry.union_all() if hasattr(sub.geometry, "union_all") else sub.geometry.unary_union
        pts = sample_points_in_geom(geom, POINTS_PER_CODE_PER_GRID, rng)
        for p in pts:
            records.append({"grid_id": int(grid_id), "code": code, "geometry": p})

    if records:
        pts_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
    else:
        pts_gdf = gpd.GeoDataFrame(columns=["grid_id", "code", "geometry"], geometry="geometry", crs=gdf.crs)

    if original_crs is not None and pts_gdf.crs != original_crs:
        pts_gdf = pts_gdf.to_crs(original_crs)

    pts_gdf.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8", index=False)
    print(f"[ok] saved: {out_path} | points: {len(pts_gdf)}")


if __name__ == "__main__":
    for zone in CLIMATE_ZONES:
        process_single_zone(zone, seed=42)
