import os
import pandas as pd

# 1. 读取 Excel 文件
file_path = r"D:\pv\data\title_nlp.xlsx"
df = pd.read_excel(file_path, sheet_name="清洗")

# 2. 确保“类型”列存在，若不存在则新建
if "类型" not in df.columns:
    df["类型"] = ""

# 3. 定义关键词与对应类型
keywords_map = {
    "农光互补": ["农业", "农光", "茶光", "林光", "林业"],
    "渔光互补": ["渔业", "渔光", "水光"],
    "牧光互补": ["牧光", "牧业", "畜光"]
}

# 4. 根据标题列填充类型列
def classify_title(title):
    if pd.isna(title):
        return ""
    title = str(title)
    for type_name, keys in keywords_map.items():
        if any(k in title for k in keys):
            return type_name
    return ""

df["类型"] = df["标题"].apply(classify_title)

# 5. 保存为新文件
save_dir = os.path.dirname(file_path)
new_file = os.path.join(save_dir, "title_类型.xlsx")
df.to_excel(new_file, index=False, engine="openpyxl")

print("已保存为：", new_file)