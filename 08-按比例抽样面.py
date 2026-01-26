import geopandas as gpd
import pandas as pd
import os

# --- 1. 配置区域列表 ---
CLIMATE_ZONES = [
    {"fullname": "华北湿润半湿润暖温带", "keyword": "climate_hb", "xx": "hb", "out_folder": "nega_hb"},
    {"fullname": "西北荒漠地区",         "keyword": "climate_xb", "xx": "xb", "out_folder": "nega_xb"},
    {"fullname": "华中华南湿润亚热带",   "keyword": "climate_hz", "xx": "hz", "out_folder": "nega_hz"},
    {"fullname": "东北湿润半湿润温带",   "keyword": "climate_db", "xx": "db", "out_folder": "nega_db"},
    {"fullname": "内蒙草原地区",         "keyword": "climate_nm", "xx": "nm", "out_folder": "nega_nm"},
    {"fullname": "青藏高原",             "keyword": "climate_qz", "xx": "qz", "out_folder": "nega_qz"},
    {"fullname": "华南湿润热带地区",     "keyword": "climate_hn", "xx": "hn", "out_folder": "nega_hn"},
]

# --- 2. 基础路径设置 ---
BASE_DIR = r"D:\pv\data"

# --- 3. 定义核心处理函数 ---
def process_single_zone(zone_info, target_total=8000, seed=42, priority_code=7229, priority_ratio=0.5):
    """
    priority_ratio: 重点类别（农田）的目标占比，默认为 0.5 (50%)
    """
    xx = zone_info['xx']
    folder = zone_info['out_folder']
    fullname = zone_info['fullname']
    
    input_filename = f"{xx}_nega.shp"
    input_path = os.path.join(BASE_DIR, folder, input_filename)
    output_filename = f"nega_{xx}_sim.shp"
    output_path = os.path.join(BASE_DIR, folder, output_filename)
    
    print(f"\n--- 开始处理: {fullname} ({folder}) ---")
    
    if not os.path.exists(input_path):
        print(f"⚠️ 跳过: 找不到文件 {input_path}")
        return

    try:
        print(f"读取: {input_filename}")
        gdf = gpd.read_file(input_path, encoding='utf-8')
        
        if 'code' not in gdf.columns:
            print(f"❌ 错误: 无 'code' 字段")
            return
            
        # ==========================================
        # 逻辑核心：按比例配额
        # ==========================================
        
        # 1. 拆分数据
        priority_gdf = gdf[gdf['code'] == priority_code]
        others_gdf = gdf[gdf['code'] != priority_code]
        
        n_priority = len(priority_gdf)
        n_others = len(others_gdf)
        
        # 2. 计算配额
        # 农田的目标数量 = 总目标 * 50% (例如 2500)
        target_priority_num = int(target_total * priority_ratio)
        
        sampled_parts = []
        
        # --- 处理重点类别 (7229) ---
        if n_priority > target_priority_num:
            print(f"  [7229] 数量充足 ({n_priority})，降采样至目标配额 {target_priority_num} (占比 {priority_ratio*100}%)")
            # 只要 50%，不多拿
            s_p = priority_gdf.sample(n=target_priority_num, random_state=seed)
            sampled_parts.append(s_p)
        else:
            print(f"  [7229] 数量不足 ({n_priority} < {target_priority_num})，全部保留。")
            sampled_parts.append(priority_gdf)
        
        # --- 处理其他类别 ---
        # 剩下的名额全部给其他类别
        current_count = len(sampled_parts[0]) if sampled_parts else 0
        slots_left = target_total - current_count
        
        if slots_left > 0 and n_others > 0:
            if n_others > slots_left:
                frac = slots_left / n_others
                print(f"  [其他] 填充剩余 {slots_left} 个名额 (抽样比例: {frac:.4f})")
                
                # 分组抽样保证多样性
                s_o = others_gdf.groupby('code', group_keys=False).apply(
                    lambda x: x.sample(frac=frac, random_state=seed)
                )
                # 再次修正总数
                if len(s_o) > slots_left:
                    s_o = s_o.sample(n=slots_left, random_state=seed)
                sampled_parts.append(s_o)
            else:
                print(f"  [其他] 数量不足以填满剩余名额，保留全部 ({n_others})。")
                sampled_parts.append(others_gdf)
        
        # 3. 合并与导出
        if sampled_parts:
            final_gdf = pd.concat(sampled_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            final_gdf = gdf.iloc[0:0]

        final_count = len(final_gdf)
        p_count_final = len(final_gdf[final_gdf['code'] == priority_code])
        ratio = (p_count_final / final_count * 100) if final_count > 0 else 0
        
        print(f"保存: {output_filename}")
        print(f"最终: {final_count} (其中 {priority_code} 数量: {p_count_final}, 占比: {ratio:.1f}%)")
        
        final_gdf.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()

# --- 4. 执行 ---
if __name__ == "__main__":
    print(f"目标：总数8000，农田(7229)目标占比 50% ，其余为其他地物。\n")
    
    for zone in CLIMATE_ZONES:
        process_single_zone(
            zone, 
            target_total=8000, 
            seed=42, 
            priority_code=7229, 
            priority_ratio=0.5  # <--- 在这里控制比例，0.5即50%
        )