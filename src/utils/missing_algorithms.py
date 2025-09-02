"""
补全缺失的关键MATLAB算法
完整实现DetectMutaInBin, Cut_OverLine, InsInHLine3等
"""
import numpy as np
from .projection_utils import bin_projection
from .histogram_utils import drow_z_barh, fill_bin
from .math_utils import rotate_with_axle, rotz, roty

def detect_muta_in_bin(bin_array):
    """
    检测二值化数组中的变化模式
    完全1:1翻译自MATLAB DetectMutaInBin.m
    
    Args:
        bin_array: 1D数组，通常是直方图差分
    Returns:
        检测到的变化模式数组
    """
    if len(bin_array) == 0:
        return np.array([])
    
    muta_bin = bin_array.copy().astype(float)
    i = 0
    
    while i < len(muta_bin):
        if bin_array[i] >= 0:
            # 处理正值序列
            inc_sum = 0
            start_i = i
            
            while i < len(muta_bin) and bin_array[i] >= 0:
                inc_sum += bin_array[i]
                i += 1
            
            # 将累计值放在序列起始位置
            muta_bin[start_i] = inc_sum
            # 其余位置清零
            for j in range(start_i + 1, i):
                muta_bin[j] = 0
                
        elif bin_array[i] < 0:
            # 处理负值序列
            dec_sum = 0
            start_i = i
            
            while i < len(muta_bin) and bin_array[i] < 0:
                dec_sum += bin_array[i]
                i += 1
                
            # 将累计值放在序列结束位置
            if i > 0:
                muta_bin[i - 1] = dec_sum
            # 其余位置清零
            for j in range(start_i, i - 1):
                muta_bin[j] = 0
        else:
            i += 1
    
    return muta_bin

def cut_recursive_line(line_points, grid_width):
    """
    递归线路切割算法
    翻译自MATLAB Cut_OverLine.m和相关cut1, cut2函数
    
    Args:
        line_points: Nx3线路点云
        grid_width: 网格宽度
    Returns:
        切割后的线路点云
    """
    if len(line_points) == 0:
        return line_points
    
    # Step 1: 第一级切割 (cut1)
    line_after_cut1, _ = cut1_algorithm(line_points, grid_width)
    
    # Step 2: 第二级切割 (cut2)
    final_line, _ = cut2_algorithm(line_after_cut1, grid_width)
    
    return final_line

def cut1_algorithm(line_points, grid_width):
    """
    第一级切割算法
    翻译自SplitOverline4Mid1.m中的cut1函数
    
    Args:
        line_points: Nx3点云
        grid_width: 网格宽度
    Returns:
        tuple: (保留的线路点, 切除的点)
    """
    if len(line_points) == 0:
        return line_points, np.empty((0, 3))
    
    # XZ平面投影
    bin_xz, _ = bin_projection(line_points, grid_width, axis_x=0, axis_y=2)
    
    # 计算宽度直方图
    z_wid = drow_z_barh(bin_xz, direction=1, return_type='wid')
    
    if len(z_wid) < 2:
        return line_points, np.empty((0, 3))
    
    # 计算差分
    d_z_wid = np.diff(z_wid)
    
    # 检测变化模式
    nd_z_wid = detect_muta_in_bin(d_z_wid)
    
    # 查找切割位置（阈值35来自MATLAB）
    cut_indices = np.where(np.abs(nd_z_wid) > 35)[0]
    
    if len(cut_indices) > 0:
        cut_pos = cut_indices[0]  # 第一个显著变化位置
        
        # 计算实际Z坐标阈值
        z_min = np.min(line_points[:, 2])
        z_threshold = z_min + cut_pos * grid_width
        
        # 分割点云
        keep_mask = line_points[:, 2] > z_threshold
        line_kept = line_points[keep_mask]
        line_cut = line_points[~keep_mask]
        
        return line_kept, line_cut
    else:
        return line_points, np.empty((0, 3))

def cut2_algorithm(line_points, grid_width):
    """
    第二级切割算法
    翻译自SplitOverline4Mid1.m中的cut2函数
    
    Args:
        line_points: Nx3点云
        grid_width: 网格宽度
    Returns:
        tuple: (保留的线路点, 切除的点)
    """
    if len(line_points) == 0:
        return line_points, np.empty((0, 3))
    
    # XY平面投影
    bin_xy, _ = bin_projection(line_points, grid_width, axis_x=0, axis_y=1)
    
    # 计算宽度直方图
    z_wid = drow_z_barh(bin_xy, direction=1, return_type='wid')
    
    if len(z_wid) < 2:
        return line_points, np.empty((0, 3))
    
    # 计算差分
    d_z_wid = np.diff(z_wid)
    
    # 检测变化模式
    nd_z_wid = detect_muta_in_bin(d_z_wid)
    
    # 查找最大变化位置
    if len(nd_z_wid) > 0 and np.max(np.abs(nd_z_wid)) > 0:
        cut_pos = np.argmax(np.abs(nd_z_wid))
        
        if cut_pos > 0:  # 避免在边界切割
            # 计算实际Y坐标阈值
            y_min = np.min(line_points[:, 1])
            y_threshold = y_min + cut_pos * grid_width
            
            # 分割点云
            keep_mask = line_points[:, 1] > y_threshold
            line_kept = line_points[keep_mask]
            line_cut = line_points[~keep_mask]
            
            return line_kept, line_cut
    
    # 如果没有找到合适的切割位置，返回原始点云
    return line_points, np.empty((0, 3))

def ins_in_h_line3_complete(tower_points, line_points, grid_width):
    """
    水平线路中的绝缘子提取算法
    完全1:1翻译自MATLAB InsInHLine3.m
    
    Args:
        tower_points: Nx3塔身点云
        line_points: Nx3线路点云
        grid_width: 网格宽度
    Returns:
        tuple: (绝缘子点云, theta1, theta2)
    """
    insulator_points = np.empty((0, 3))
    
    if len(tower_points) == 0 or len(line_points) == 0:
        return insulator_points, 0.0, 0.0
    
    # 双轴旋转对齐（完全按照MATLAB步骤）
    # 第一次旋转：Z轴对齐
    line_r3, theta1 = rotate_with_axle(line_points, axis=3)
    
    # 第二次旋转：Y轴对齐
    line_r32, theta2 = rotate_with_axle(line_r3, axis=2)
    
    # 对塔身点云应用相同的旋转
    tower_r32 = tower_points @ rotz(np.degrees(theta1)).T @ roty(np.degrees(theta2)).T
    
    # 计算塔身X轴中点
    tower_x_mid = np.min(tower_r32[:, 0]) + (np.max(tower_r32[:, 0]) - np.min(tower_r32[:, 0])) / 2
    
    # XY平面绝缘子提取
    bin_xy, _ = bin_projection(line_r32, grid_width, axis_x=0, axis_y=1)
    
    # 计算宽度直方图（方向-2对应从上到下）
    xy_wid = drow_z_barh(bin_xy, direction=-2, return_type='wid')
    
    if len(xy_wid) == 0:
        return insulator_points, theta1, theta2
    
    # 填充空洞
    f_xy_wid = fill_bin(xy_wid)
    
    # 找到最大宽度位置
    max_ind = np.argmax(xy_wid)
    mid_pos = int(np.ceil(len(f_xy_wid) / 2))
    
    # 判断绝缘子位置（上方还是下方）
    line_x_min, line_x_max = np.min(line_r32[:, 0]), np.max(line_r32[:, 0])
    
    if abs(tower_x_mid - line_x_min) > abs(tower_x_mid - line_x_max):
        # 绝缘子在上方
        # 去除噪声
        max_v = np.max(f_xy_wid[:mid_pos]) if mid_pos > 0 else 0
        noise_count = np.sum(f_xy_wid[:mid_pos] == max_v) if mid_pos > 0 else 0
        
        if noise_count <= 3:
            f_xy_wid[f_xy_wid[:mid_pos] == max_v] = max_v - 1
        
        # 计算阈值
        threshold = np.max(f_xy_wid[:mid_pos]) + 1 if mid_pos > 0 else 1
        
        # 查找切割位置
        if mid_pos < len(f_xy_wid):
            above_threshold = np.where(f_xy_wid[mid_pos:] > threshold)[0]
            if len(above_threshold) > 0:
                cut_pos = mid_pos + above_threshold[0]
                
                # 提取绝缘子点云
                x_threshold = np.min(line_r32[:, 0]) + cut_pos * grid_width
                insulator_mask = line_r32[:, 0] > x_threshold
                insulator_points_rotated = line_r32[insulator_mask]
                
                # 逆向旋转回原始坐标系
                if len(insulator_points_rotated) > 0:
                    insulator_points = (insulator_points_rotated @ 
                                     roty(-np.degrees(theta2)).T @ 
                                     rotz(-np.degrees(theta1)).T)
    else:
        # 绝缘子在下方
        # 去除噪声
        max_v = np.max(f_xy_wid[mid_pos:]) if mid_pos < len(f_xy_wid) else 0
        noise_count = np.sum(f_xy_wid[mid_pos:] == max_v) if mid_pos < len(f_xy_wid) else 0
        
        if noise_count <= 3:
            f_xy_wid[f_xy_wid[mid_pos:] == max_v] = max_v - 1
        
        # 计算阈值
        threshold = np.max(f_xy_wid[mid_pos:]) + 1 if mid_pos < len(f_xy_wid) else 1
        
        # 查找切割位置
        above_threshold = np.where(f_xy_wid[:mid_pos] > threshold)[0]
        if len(above_threshold) > 0:
            cut_pos = above_threshold[-1]  # 最后一个位置
            
            # 提取绝缘子点云
            x_threshold = np.min(line_r32[:, 0]) + cut_pos * grid_width
            insulator_mask = line_r32[:, 0] < x_threshold
            insulator_points_rotated = line_r32[insulator_mask]
            
            # 逆向旋转回原始坐标系
            if len(insulator_points_rotated) > 0:
                insulator_points = (insulator_points_rotated @ 
                                 roty(-np.degrees(theta2)).T @ 
                                 rotz(-np.degrees(theta1)).T)
    
    return insulator_points, theta1, theta2

def split_overline_4_mid1_complete(cross_points, grid_width):
    """
    完整的中线分割算法
    完全1:1翻译自MATLAB SplitOverline4Mid1.m
    
    Args:
        cross_points: Nx3交叉点云
        grid_width: 网格宽度
    Returns:
        分割后的线路点云
    """
    if len(cross_points) == 0:
        return np.empty((0, 3))
    
    # 计算中点和半长度
    y_min, y_max = np.min(cross_points[:, 1]), np.max(cross_points[:, 1])
    half_len = (y_max - y_min) / 2
    mid_y = y_min + half_len
    
    all_line_pts = []
    
    # 处理两半部分
    for i in range(2):
        # 提取一半的点云
        y_start = y_min + half_len * i
        y_end = y_min + half_len * (i + 1)
        
        half_cross = cross_points[
            (cross_points[:, 1] >= y_start) & 
            (cross_points[:, 1] < y_end)
        ]
        
        if len(half_cross) < 10:  # 点数太少跳过
            continue
        
        # 确定边缘线路部分（远离塔身的1/4部分）
        d_min = abs(mid_y - np.min(half_cross[:, 1]))
        d_max = abs(mid_y - np.max(half_cross[:, 1]))
        part_len = (np.max(half_cross[:, 1]) - np.min(half_cross[:, 1])) / 3
        
        if d_min < d_max:  # Y值较小的一端更接近塔身
            edge_line = half_cross[half_cross[:, 1] > np.max(half_cross[:, 1]) - part_len]
        else:  # Y值较大的一端更接近塔身
            edge_line = half_cross[half_cross[:, 1] < np.min(half_cross[:, 1]) + part_len]
        
        if len(edge_line) < 3:
            continue
        
        # 双轴旋转对齐
        edge_r3, theta1 = rotate_with_axle(edge_line, axis=3)
        edge_r32, theta2 = rotate_with_axle(edge_r3, axis=2)
        
        # 对整个半部分应用相同旋转
        half_cross_r32 = (half_cross @ 
                         rotz(np.degrees(theta1)).T @ 
                         roty(np.degrees(theta2)).T)
        
        # 变换坐标轴顺序 [X,Z,Y]
        half_cross_r32_reordered = half_cross_r32[:, [0, 2, 1]]
        
        # 创建XZ平面投影
        bin_xz, _ = bin_projection(half_cross_r32_reordered, grid_width, axis_x=0, axis_y=2)
        
        # 找到最大宽度位置
        wid_hist = drow_z_barh(bin_xz, direction=1, return_type='wid')
        if len(wid_hist) == 0:
            continue
            
        max_wid_ind = np.argmax(wid_hist)
        
        # 根据位置决定切割方向
        if max_wid_ind > len(bin_xz) / 2:
            # 上半部分，正常切割
            line_r32 = cut_recursive_line(half_cross_r32_reordered, grid_width)
        else:
            # 下半部分，翻转Z轴后切割
            half_cross_r32_reordered[:, 2] = -half_cross_r32_reordered[:, 2]
            line_r32 = cut_recursive_line(half_cross_r32_reordered, grid_width)
            line_r32[:, 2] = -line_r32[:, 2]
        
        # 恢复原始坐标系
        if len(line_r32) > 0:
            # 恢复轴顺序 [X,Y,Z]
            line_original_order = line_r32[:, [0, 2, 1]]
            
            # 逆向旋转
            line_final = (line_original_order @ 
                         roty(-np.degrees(theta2)).T @ 
                         rotz(-np.degrees(theta1)).T)
            
            all_line_pts.append(line_final)
    
    # 合并所有分割结果
    if all_line_pts:
        return np.vstack(all_line_pts)
    else:
        return np.empty((0, 3))