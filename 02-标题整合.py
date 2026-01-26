## 测试上传git
import os
import pandas as pd

# 设置文件夹路径和输出文件名
folder_path = r"D:\pv\data\title"
output_file = os.path.join(folder_path, "all_title.csv")

# 获取所有page_x_titles.csv文件的文件名
file_names = [f"page_{i}_titles.csv" for i in range(1, 80)]

# 初始化一个空的DataFrame来存储所有数据
all_titles = pd.DataFrame()

# 读取每个文件并将其内容追加到all_titles中
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='mbcs')  # 使用ANSI编码读取文件
        all_titles = pd.concat([all_titles, df], ignore_index=True)

# 将合并后的数据保存为新的CSV文件，使用ANSI编码保存，去除前导空格
all_titles.to_csv(output_file, index=False, encoding='mbcs', header=True)

print(f"所有标题已成功合并并保存为 {output_file}")
