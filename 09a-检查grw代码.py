import geopandas as gpd
import fiona
import os

# 1. 设置文件路径
input_path = r"D:\pv\grw_microsoft\grw_2024q2.gpkg"

# 自动生成输出文件名 (原文件名 + _China_only)
dir_name = os.path.dirname(input_path)
base_name = os.path.basename(input_path)
output_name = base_name.replace(".gpkg", "_China_only.gpkg")
output_path = os.path.join(dir_name, output_name)

print(f"源文件: {input_path}")
print(f"输出文件将保存为: {output_path}")
print("-" * 50)

try:
    # 2. 自动获取所有图层名称
    # fiona.listlayers 可以列出 GPKG 中包含的所有图层
    layers = fiona.listlayers(input_path)
    print(f"检测到文件中包含 {len(layers)} 个图层: {layers}")

    # 如果输出文件已存在，先删除，防止追加数据导致重复或错误
    if os.path.exists(output_path):
        os.remove(output_path)
        print("已清理旧的输出文件。")

    # 3. 遍历每个图层进行处理
    for layer_name in layers:
        print(f"\n正在处理图层: [{layer_name}] ...")
        
        # 读取该图层
        gdf = gpd.read_file(input_path, layer=layer_name)
        
        # 检查是否存在 'COUNTRY' 字段 (忽略大小写判断)
        # 将列名都转为大写来对比，防止 'Country' vs 'COUNTRY' 的差异
        col_map = {c.upper(): c for c in gdf.columns}
        
        if 'COUNTRY' in col_map:
            real_col_name = col_map['COUNTRY'] # 获取真实的列名
            
            # 4. 筛选数据 (兼容 'China', 'china', 'CHINA')
            # 使用 str.lower() == 'china' 来确保大小写不敏感
            gdf_china = gdf[gdf[real_col_name].astype(str).str.lower() == 'china']
            
            count = len(gdf_china)
            print(f"  - 原始数据: {len(gdf)} 条")
            print(f"  - 筛选后 (China): {count} 条")
            
            if count > 0:
                print(f"  - 正在写入新文件...")
                # 写入数据
                # 如果是第一个写入的图层，会自动创建文件
                # 如果文件包含多个图层，Geopandas 会追加
                gdf_china.to_file(output_path, driver='GPKG', layer=layer_name)
                print(f"  - ✅ 图层 [{layer_name}] 处理完毕")
            else:
                print(f"  - ⚠️ 警告: 该图层中没有找到 'China' 的数据，跳过保存。")
        else:
            print(f"  - ⚠️ 跳过: 该图层没有 'COUNTRY' 字段。")

    print("\n" + "=" * 50)
    print(f"所有操作完成！\n请查看文件: {output_path}")

except Exception as e:
    print(f"❌ 发生错误: {e}")