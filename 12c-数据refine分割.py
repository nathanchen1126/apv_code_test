import geopandas as gpd
import os

# 1. 设置路径
input_path = r"D:\pv\result\result_refine\apv_2023_all_merge_block.shp"
output_dir = os.path.dirname(input_path)

# 2. 读取数据
print("正在读取文件...")
gdf = gpd.read_file(input_path)

# 3. 为 refine_id 字段按顺序赋值（从1开始，或者你可以改为0）
# 并确保其类型为短整型 (int32)
gdf['refine_id'] = range(1, len(gdf) + 1)
gdf['refine_id'] = gdf['refine_id'].astype('int32')

# 4. 保存第一个完整的结果文件
main_output = os.path.join(output_dir, "result_refine_apv2023all.shp")
print(f"正在保存完整文件至: {main_output}")
gdf.to_file(main_output, encoding='utf-8')

# 5. 按照 refine_id 拆分为数量相等的两部分
# 计算中点位置
mid_point = len(gdf) // 2

# 拆分数据
gdf_one = gdf.iloc[:mid_point]
gdf_two = gdf.iloc[mid_point:]

# 6. 保存拆分后的两个文件
output_one = os.path.join(output_dir, "result_refine_apv2023all_one.shp")
output_two = os.path.join(output_dir, "result_refine_apv2023all_two.shp")

print(f"正在保存上半部分（{len(gdf_one)} 条记录）...")
gdf_one.to_file(output_one, encoding='utf-8')

print(f"正在保存下半部分（{len(gdf_two)} 条记录）...")
gdf_two.to_file(output_two, encoding='utf-8')

print("所有操作已成功完成！")