import os
import pandas as pd
import requests
import json
import time

# 设置 DeepSeek API 密钥
deepseek_api_key = "sk-0a5add4f37324c4cb2d1e4e37b40e2a7"  # 在此处替换为您的 DeepSeek API 密钥
api_url = "https://api.deepseek.com/v1/chat/completions"

# 读取 CSV 文件
file_path = r"D:\pv\data\all_title.csv"
df = pd.read_csv(file_path, encoding='ansi')

# 准备存储提取结果
extracted_data = []

# 定义函数用于提取信息
def extract_information(title):
    prompt = f"""
    请阅读以下标题并提取其中的信息：
    1. 省、城市、县
    2. 装机量（如果有）
    3. 是否涉及农光互补项目（只需回答"是"或"否"）

    判断农光互补的标准：如果标题中包含以下任何关键词，则为"是"：
    "农业"、"农光"、"灌溉"、"大棚"、"渔业"、"渔光"、"牧光"、"治沙"、"牧业"、"畜光"、"林业"、"林光"、"旅游"、"盐光"、"茶光"、"助贫"

    请以JSON格式返回以下字段：
    {{
        "省": "",
        "市": "",
        "县": "",
        "装机量": "",
        "农光互补": ""  # 只需填写"是"或"否"
    }}
    
    标题内容："{title}"
    """
    
    # 调用 DeepSeek API 获取响应
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个擅长从标题中提取信息的专家。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 获取模型返回的内容
        content = result["choices"][0]["message"]["content"]
        return content
        
    except Exception as e:
        print(f"API调用错误: {e}")
        return json.dumps({
            "省": "",
            "市": "", 
            "县": "",
            "装机量": "",
            "农光互补": "否"
        })

# 逐个处理标题
for index, row in df.iterrows():
    title = row[1]  # 假设标题在第二列
    print(f"处理第 {index + 1} 条标题: {title}")
    
    extracted_info = extract_information(title)
    
    # 延迟以避免API过度请求
    time.sleep(1)  # 如果API有调用频率限制，可以适当调整延迟时间

    # 添加提取结果到数据中
    extracted_data.append([row[0], title, extracted_info])

# 将提取结果保存到新的 CSV 文件
output_file_path = r"D:\pv\data\title_mean.csv"
result_df = pd.DataFrame(extracted_data, columns=["序号", "标题", "提取结果"])
result_df.to_csv(output_file_path, index=False, encoding='ansi')

print(f"标题提取结果已保存到 {output_file_path}")