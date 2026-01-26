# 修改气候区投影面
import geopandas as gpd
from pathlib import Path
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

def align_projection():
    # ================= 配置路径 =================
    # 1. 参考文件 (提供目标坐标系)
    ref_path = Path(r"D:\pv\result\result_postprocess\APV_2023_all.shp")
    
    # 2. 需要转换的文件 (气候区数据)
    input_path = Path(r"D:\pv\data\Chinese_climate\Chinese_climate.shp")
    
    # 3. 输出文件路径 (建议另存为新文件，避免破坏原始数据)
    output_path = Path(r"D:\pv\data\Chinese_climate\Chinese_climate_reprojected.shp")

    # ================= 执行处理 =================
    print("[1/4] 正在检查文件...")
    if not ref_path.exists():
        print(f"错误：参考文件不存在 -> {ref_path}")
        return
    if not input_path.exists():
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 1. 获取目标坐标系 (只读一行以加快速度)
    print(f"[2/4] 正在读取参考坐标系: {ref_path.name}")
    try:
        ref_gdf = gpd.read_file(ref_path, rows=1)
        target_crs = ref_gdf.crs
        print(f"  -> 目标坐标系为: {target_crs}")
    except Exception as e:
        print(f"  -> 读取参考文件失败: {e}")
        return

    # 2. 读取气候区数据
    print(f"[3/4] 正在读取并转换气候区数据...")
    try:
        climate_gdf = gpd.read_file(input_path)
        print(f"  -> 原始坐标系: {climate_gdf.crs}")
        
        # 执行转换
        if climate_gdf.crs != target_crs:
            climate_transformed = climate_gdf.to_crs(target_crs)
            print("  -> 坐标系转换完成！")
        else:
            climate_transformed = climate_gdf
            print("  -> 坐标系一致，无需转换。")
            
    except Exception as e:
        print(f"  -> 处理数据失败: {e}")
        return

    # 3. 保存结果
    print(f"[4/4] 正在保存结果至: {output_path}")
    try:
        climate_transformed.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        print("完成！所有文件坐标系已统一。")
    except Exception as e:
        print(f"  -> 保存失败: {e}")

if __name__ == "__main__":
    align_projection()