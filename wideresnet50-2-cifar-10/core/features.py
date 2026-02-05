"""
特征提取模块
负责:
1. 从Q变换后的权重计算块特征 φ
2. 支持 φ1~φ4 (基础特征) + φ5~φ8 (BN特征)
3. 按列z-score归一化，消除尺度差异对损失和评估的影响
"""

import torch
from typing import Optional, Dict

# ============================================================
# 全局常量
# ============================================================
EPS = 1e-8  # 数值稳定性常数

# ============================================================
# 特征提取函数
# ============================================================
def compute_phi(W_tilde: torch.Tensor, 
                bn_params: Optional[Dict[str, torch.Tensor]] = None,
                eps: float = EPS,
                include_phi4: bool = True) -> torch.Tensor:
    """
    计算块特征向量 φ
    
    在Q空间计算特征,避免攻击者通过原始权重统计识别块
    
    Args:
        W_tilde: Q变换后的权重矩阵 [c, d], 其中:
                 - c: 块大小
                 - d: Cin * k * k (展平后的滤波器维度)
        bn_params: 可选的BN参数字典,包含:
                   - 'weight': gamma [c]
                   - 'bias': beta [c]
        eps: 数值稳定性常数
    
    Returns:
        特征向量:
        - 无BN: [4] = [φ1, φ2, φ3, φ4]
        - 有BN: [8] = [φ1, φ2, φ3, φ4, φ5, φ6, φ7, φ8]
    
    特征含义:
        φ1: log(整体能量) - 抹平块级尺度差异
        φ2: 通道能量均值 - 抹平平均通道强度
        φ3: 通道能量方差 - 抹平通道强弱不均
        φ4: 谱峰值占比 - 抹平主导方向结构
        φ5: BN scale均值
        φ6: BN scale方差
        φ7: BN shift均值
        φ8: BN shift方差
    """
    c, d = W_tilde.shape
    
    # ========================================
    # 基础特征 (无需BN)
    # ========================================
    
    # φ1: 整体能量 (log Frobenius范数)
    # Frobenius范数 = sqrt(sum of all squared elements)
    fro_norm = torch.norm(W_tilde, p='fro')  # 标量
    phi1 = torch.log(fro_norm + eps)
    
    # φ2, φ3: 通道能量统计
    # 先计算每个通道(行)的L2范数
    r = torch.sqrt((W_tilde ** 2).sum(dim=1) + eps)  # [c]
    
    # φ2: 通道能量均值
    phi2 = r.mean()
    
    # φ3: 通道能量方差
    phi3 = r.var()
    
    # φ4: 谱峰值占比 (dominant-direction ratio)
    # Gram矩阵 G = W_tilde @ W_tilde^T, 其特征值反映块内方向结构
    # 最大特征值占总和的比例越高, 说明越存在主导方向
    if include_phi4:
        G = W_tilde @ W_tilde.T  # [c, c]
        eigenvalues = torch.linalg.eigvalsh(G)  # 升序, [c]
        phi4 = eigenvalues[-1] / (eigenvalues.sum() + eps)  # 最大特征值占比
    
    # 组装基础特征
    phi = [phi1, phi2, phi3]
    if include_phi4:
        phi.append(phi4)
    
    # ========================================
    # BN特征 (如果提供了BN参数)
    # ========================================
    if bn_params is not None:
        gamma = bn_params['weight']  # [c]
        beta = bn_params['bias']     # [c]
        
        # φ5: BN scale (gamma) 均值
        phi5 = gamma.mean()
        
        # φ6: BN scale (gamma) 方差
        phi6 = gamma.var()
        
        # φ7: BN shift (beta) 均值
        phi7 = beta.mean()
        
        # φ8: BN shift (beta) 方差
        phi8 = beta.var()
        
        # 添加到特征列表
        phi.extend([phi5, phi6, phi7, phi8])
    
    # 转换为张量并返回
    return torch.stack(phi)  # [3] or [7]


def compute_layer_phi_list(block_manager, key_manager, layer_name: str, 
                          device='cpu', include_phi4: bool = True) -> torch.Tensor:
    """
    计算某层所有块的原始特征矩阵
    
    归一化逻辑不在这里，由调用方按需使用 normalize_phi()
    
    Args:
        block_manager: BlockManager实例
        key_manager: KeyManager实例
        layer_name: 层名称
        device: 设备
        include_phi4: 是否包含φ4谱峰值占比特征
    
    Returns:
        原始特征矩阵 [g, phi_dim]
        phi_dim: 无BN时为3(不含φ4)或4(含φ4), 有BN时为7或8
    """
    info = block_manager.get_layer_info(layer_name)
    c = info['c']
    g = info['g']
    
    phi_list = []
    
    for i in range(g):
        # 1. 生成Q矩阵
        Q = key_manager.generate_Q(layer_name, i, c, device=device)
        
        # 2. 获取权重块
        W_blk = block_manager.get_block_weight(layer_name, i)  # [c, Cin, k, k]
        W_blk = W_blk.to(device)
        
        # 3. 展平
        W_flat = W_blk.reshape(c, -1)  # [c, d]
        
        # 4. Q变换
        W_tilde = Q @ W_flat  # [c, d]
        
        # 5. 获取BN参数
        bn_params = block_manager.get_block_bn(layer_name, i)
        # 将BN参数移到device
        bn_params = {k: v.to(device) for k, v in bn_params.items()}
        
        # 6. 计算特征
        phi = compute_phi(W_tilde, bn_params, include_phi4=include_phi4)
        phi_list.append(phi)
    
    # 堆叠成矩阵并返回原始特征（不做归一化）
    return torch.stack(phi_list)  # [g, phi_dim]


def normalize_phi(phi_matrix: torch.Tensor) -> torch.Tensor:
    """
    对原始特征矩阵按列做 z-score 归一化
    
    仅供损失函数使用。评估指标如果需要归一化版本，也调用此函数，
    但不应将此函数嵌入 compute_layer_phi_list 本身。
    
    Args:
        phi_matrix: 原始特征矩阵 [g, phi_dim]
    
    Returns:
        归一化后的特征矩阵 [g, phi_dim], 每列均值0 std=1
    
    注意:
        sigma 使用 .detach(): 当所有块收敛到相同值时 sigma->0,
        若不 detach 则 1/sigma 会导致梯度爆炸。
        损失函数调用时 phi_matrix 带梯度，detach 只作用于 sigma；
        评估函数调用时 phi_matrix 本身已 detach，detach 无副作用。
    """
    mu = phi_matrix.mean(dim=0, keepdim=True)              # [1, phi_dim]
    sigma = phi_matrix.std(dim=0, keepdim=True).detach()   # [1, phi_dim]
    sigma = torch.clamp(sigma, min=EPS)
    return (phi_matrix - mu) / sigma


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=== Test Feature Extraction ===")
    
    # 模拟Q变换后的权重
    c = 16
    d = 128 * 3 * 3  # 例如: 128个输入通道, 3×3卷积核
    W_tilde = torch.randn(c, d)
    
    # 1. 测试无BN的特征
    print("\n1. Features without BN:")
    phi_no_bn = compute_phi(W_tilde)
    print(f"   Shape: {phi_no_bn.shape}")
    print(f"   Values: {phi_no_bn}")
    
    # 2. 测试有BN的特征
    print("\n2. Features with BN:")
    bn_params = {
        'weight': torch.randn(c),  # gamma
        'bias': torch.randn(c),     # beta
    }
    phi_with_bn = compute_phi(W_tilde, bn_params)
    print(f"   Shape: {phi_with_bn.shape}")
    print(f"   Values: {phi_with_bn}")
    
    # 3. 测试数值稳定性
    print("\n3. Numerical stability test:")
    W_zero = torch.zeros(c, d)
    phi_zero = compute_phi(W_zero)
    print(f"   Zero weights φ1 (should be finite): {phi_zero[0]}")
    print(f"   Is finite: {torch.isfinite(phi_zero).all()}")
    
    # 4. 测试同质化效果
    print("\n4. Homogenization test:")
    # 创建两个相似的块
    W_tilde_1 = torch.randn(c, d)
    W_tilde_2 = W_tilde_1 + 0.1 * torch.randn(c, d)  # 添加小扰动
    
    phi_1 = compute_phi(W_tilde_1)
    phi_2 = compute_phi(W_tilde_2)
    
    distance = torch.norm(phi_1 - phi_2)
    print(f"   Feature distance: {distance:.4f}")
    print(f"   Relative difference: {(distance / torch.norm(phi_1)).item():.2%}")