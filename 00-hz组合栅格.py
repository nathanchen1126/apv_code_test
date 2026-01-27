"""
将 APV hz 分块栅格拼成一张大图。

输入位于 D:\\pv\\result\\hz\\apv_hz_block，命名规则：
APV_hz_merge_Raster_Tile_<x>_<y>.tif
其中 x 来自 coveringGrid 的第 2 列（索引从 0 开始），y 来自第 5 行，
逗号已用下划线替换。

合并结果写入同目录下的 apv_hz_raster_block.tif。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple

import rasterio
from rasterio.merge import merge

# ---- 配置 ----
CONFIG = {
    "input_dir": Path(r"D:\pv\result\hz\apv_hz_block"),
    "pattern": "APV_hz_merge_Raster_Tile_*_*.tif",
    "output": Path(r"D:\pv\result\hz\apv_hz_block\apv_hz_raster_block.tif"),
    # 如需覆盖已存在的合并结果，设为 True
    "overwrite": True,
    # 输出 GeoTIFF 的可选压缩与大文件控制
    "compress": "LZW",
    "big_tiff": "IF_SAFER",  # 其他可选值：YES/NO/IF_NEEDED
}


# ---- 工具函数 ----
def parse_xy(path: Path) -> Tuple[int, int]:
    """
    从文件名提取 (x, y) 索引，例如 APV_hz_merge_Raster_Tile_10_4.tif。
    如不符合规则则抛出 ValueError。
    """
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected tile name: {path.name}")
    try:
        x = int(parts[-2])
        y = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse tile indices from {path.name}") from exc
    return x, y


def list_tiles(input_dir: Path, pattern: str) -> list[Path]:
    tiles = sorted(input_dir.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No tiles found under {input_dir} matching {pattern}")
    return tiles


def ensure_consistent(datasets: Iterable[rasterio.io.DatasetReader]) -> None:
    """
    确保所有瓦片的 CRS、像元大小、波段数、数据类型一致，
    避免合并时出现静默错位。
    """
    base = None
    for ds in datasets:
        # dtype 使用 profile 中的声明，兼容不同 rasterio 版本
        meta = (ds.crs, ds.transform.a, ds.transform.e, ds.count, ds.profile.get("dtype"))
        if base is None:
            base = meta
            continue
        if meta != base:
            raise RuntimeError(
                "Tile mismatch detected (crs/scale/band/dtype). "
                f"First: {base}, current: {meta} from {ds.name}"
            )


def mosaic_tiles(tile_paths: list[Path], output: Path, overwrite: bool) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists. Set overwrite=True to replace.")

    # 惰性打开瓦片，rasterio.merge 会利用其空间参考
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        ensure_consistent(datasets)
        mosaic, out_transform = merge(datasets)
        first = datasets[0]
        profile = first.profile.copy()
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "driver": "GTiff",
                "compress": CONFIG["compress"],
                "BIGTIFF": CONFIG["big_tiff"],
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    return output


def main() -> None:
    input_dir: Path = CONFIG["input_dir"]
    pattern: str = CONFIG["pattern"]
    output: Path = CONFIG["output"]
    overwrite: bool = CONFIG["overwrite"]

    tile_paths = list_tiles(input_dir, pattern)
    print(f"[info] found {len(tile_paths)} tiles under {input_dir}")
    print(f"[info] example: {tile_paths[0].name}")

    # just to surface obvious naming mistakes early
    try:
        parsed = [parse_xy(p) for p in tile_paths]
        x_vals = sorted({p[0] for p in parsed})
        y_vals = sorted({p[1] for p in parsed})
        print(f"[info] x indices: {x_vals}")
        print(f"[info] y indices: {y_vals}")
    except ValueError as exc:
        print(f"[warn] tile name check skipped: {exc}")

    out_path = mosaic_tiles(tile_paths, output, overwrite=overwrite)
    print(f"[ok] merged raster written to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[error] {err}")
        sys.exit(1)
