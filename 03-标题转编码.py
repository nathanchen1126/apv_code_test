import os
import pandas as pd

# 定义输入和输出路径
input_dir = r"D:\pv\data\txt"
output_dir = r"D:\pv\data\title"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有的 CSV 文件
for i in range(1, 80):  # page_x_titles.csv, x 从 1 到 79
    file_name = f"page_{i}_titles.csv"
    input_file_path = os.path.join(input_dir, file_name)
    
    if os.path.exists(input_file_path):
        # 读取 CSV 文件
        df = pd.read_csv(input_file_path)
        
        # 生成输出文件路径
        output_file_path = os.path.join(output_dir, file_name)
        
        # 将 DataFrame 保存为 ANSI 编码的 CSV 文件，忽略无法编码的 Unicode 字符
        df.to_csv(output_file_path, index=False, encoding='mbcs', errors='ignore')  # 'mbcs' 是 ANSI 编码
        print(f"已保存 {file_name} 为 ANSI 编码，忽略无法编码的字符")
    else:
        print(f"文件 {file_name} 不存在")
