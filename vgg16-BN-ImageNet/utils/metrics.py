"""
混淆效果评估指标
负责:
1. 块间相似度计算(选中块之间、未选中块之间、跨组)
2. 层内同质化程度评估
"""

import torch
import numpy as np
from typing import List, Optional, Dict


class ConfuseMetrics:
    """
    混淆效果评估指标类
    """
    
    @staticmethod
    def compute_intra_similarity(phi_list: List[torch.Tensor], 
                                 S: Optional[List[int]] = None) -> Dict[str, float]:
        """
        计算块间的余弦相似度
        
        Args:
            phi_list: 特征向量列表 [phi_dim]
            S: 选中的块索引列表(可选)
        
        Returns:
            {
                'overall_similarity': 所有块之间的平均相似度,
                'selected_similarity': 选中块之间的平均相似度(如果提供S),
                'unselected_similarity': 未选中块之间的平均相似度,
                'cross_similarity': 选中块与未选中块之间的平均相似度
            }
        
        相似度越高说明块越同质化
        """
        if not phi_list:
            return {
                'overall_similarity': 0.0,
                'selected_similarity': 0.0,
                'unselected_similarity': 0.0,
                'cross_similarity': 0.0
            }
        
        # 堆叠成矩阵 [g, phi_dim]
        phi_matrix = torch.stack(phi_list)
        g = len(phi_list)
        
        # 归一化(计算余弦相似度)
        phi_norm = phi_matrix / (phi_matrix.norm(dim=1, keepdim=True) + 1e-8)
        
        # 计算余弦相似度矩阵 [g, g]
        sim_matrix = phi_norm @ phi_norm.T
        
        # 创建掩码(去掉对角线,不计算自己和自己的相似度)
        mask = ~torch.eye(g, dtype=bool, device=sim_matrix.device)
        
        results = {}
        
        # 1. 总体相似度(所有块对)
        results['overall_similarity'] = sim_matrix[mask].mean().item()
        
        # 如果没有提供S,只返回总体相似度
        if S is None:
            results['selected_similarity'] = results['overall_similarity']
            results['unselected_similarity'] = results['overall_similarity']
            results['cross_similarity'] = results['overall_similarity']
            return results
        
        # 2. 分组计算
        S_set = set(S)
        S_indices = list(S_set)
        U_indices = [i for i in range(g) if i not in S_set]
        
        # 2.1 选中块之间(S-S)
        if len(S_indices) > 1:
            selected_pairs = []
            for i in range(len(S_indices)):
                for j in range(i+1, len(S_indices)):
                    selected_pairs.append(sim_matrix[S_indices[i], S_indices[j]])
            results['selected_similarity'] = torch.stack(selected_pairs).mean().item()
        else:
            results['selected_similarity'] = 1.0  # 只有一个块,定义为完全相似
        
        # 2.2 未选中块之间(U-U)
        if len(U_indices) > 1:
            unselected_pairs = []
            for i in range(len(U_indices)):
                for j in range(i+1, len(U_indices)):
                    unselected_pairs.append(sim_matrix[U_indices[i], U_indices[j]])
            results['unselected_similarity'] = torch.stack(unselected_pairs).mean().item()
        else:
            results['unselected_similarity'] = 1.0
        
        # 2.3 跨组相似度(S-U)
        if S_indices and U_indices:
            cross_pairs = []
            for s_idx in S_indices:
                for u_idx in U_indices:
                    cross_pairs.append(sim_matrix[s_idx, u_idx])
            results['cross_similarity'] = torch.stack(cross_pairs).mean().item()
        else:
            results['cross_similarity'] = 0.0
        
        return results
    
    @staticmethod
    def compute_layer_homogeneity(phi_list: List[torch.Tensor]) -> float:
        """
        计算层内同质化程度
        使用特征向量的标准差作为指标
        
        Args:
            phi_list: 特征向量列表
        
        Returns:
            各维度标准差的平均值(越小越同质化)
        
        原理:
            如果所有块特征都接近,则跨块的标准差应该很小
        """
        if not phi_list:
            return 0.0
        
        # 堆叠成矩阵 [g, phi_dim]
        phi_matrix = torch.stack(phi_list)
        
        # 计算每个维度的标准差
        std_per_dim = phi_matrix.std(dim=0)  # [phi_dim]
        
        # 返回平均标准差
        return std_per_dim.mean().item()


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=== Test ConfuseMetrics ===")
    
    # 1. 创建模拟数据
    # 场景1: 高度同质化的块
    print("\n1. Highly homogeneous blocks:")
    base_phi = torch.tensor([1.0, 2.0, 3.0, 0.5])
    homogeneous_list = [base_phi + 0.01 * torch.randn(4) for _ in range(10)]
    
    similarity_homo = ConfuseMetrics.compute_intra_similarity(homogeneous_list)
    homogeneity_homo = ConfuseMetrics.compute_layer_homogeneity(homogeneous_list)
    
    print(f"   Similarity: {similarity_homo['overall_similarity']:.4f} (should be high)")
    print(f"   Homogeneity: {homogeneity_homo:.4f} (should be low)")
    
    # 场景2: 异质化的块
    print("\n2. Heterogeneous blocks:")
    heterogeneous_list = [torch.randn(4) for _ in range(10)]
    
    similarity_hetero = ConfuseMetrics.compute_intra_similarity(heterogeneous_list)
    homogeneity_hetero = ConfuseMetrics.compute_layer_homogeneity(heterogeneous_list)
    
    print(f"   Similarity: {similarity_hetero['overall_similarity']:.4f} (should be low)")
    print(f"   Homogeneity: {homogeneity_hetero:.4f} (should be high)")
    
    # 3. 测试分组相似度
    print("\n3. Grouped similarity:")
    S = [0, 1, 2, 3, 4]  # 前5个是选中块
    grouped_sim = ConfuseMetrics.compute_intra_similarity(homogeneous_list, S)
    
    print(f"   Selected-Selected: {grouped_sim['selected_similarity']:.4f}")
    print(f"   Unselected-Unselected: {grouped_sim['unselected_similarity']:.4f}")
    print(f"   Selected-Unselected: {grouped_sim['cross_similarity']:.4f}")
    
    print("\n" + "="*60)