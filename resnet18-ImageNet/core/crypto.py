"""
密码学工具模块
负责:
1. 密钥管理
2. 从密钥派生确定性随机数(HMAC-SHA256)
3. 生成正交矩阵Q (QR分解)
4. 生成选块集合S (确定性洗牌)
5. 生成块置换π (确定性置换)
"""

import hashlib
import hmac
import numpy as np
import torch
from typing import List, Tuple

# ============================================================
# 全局常量
# ============================================================
HASH_ALGORITHM = 'sha256'           # HMAC使用的哈希算法
DOMAIN_Q = b'Q'                     # Q矩阵生成的域分离标签
DOMAIN_S = b'S'                     # 选块集合的域分离标签  
DOMAIN_P = b'P'                     # 置换的域分离标签

# ============================================================
# KeyManager 类
# ============================================================
class KeyManager:
    """
    密钥管理器
    从主密钥K派生各种确定性随机数
    """
    
    def __init__(self, key: str):
        """
        Args:
            key: 十六进制字符串密钥 (e.g., '4a7d1ed414474e4033ac29ccb8653d9b')
                 长度建议>=32字符(16字节)
        """
        # 将hex字符串转为bytes
        self.key = bytes.fromhex(key)
        
        # 检查密钥长度
        if len(self.key) < 16:
            raise ValueError(f"Key too short: {len(self.key)} bytes. Minimum 16 bytes required.")
        
        print(f"[KeyManager] Initialized with {len(self.key)}-byte key")
    
    def _hmac(self, domain: bytes, message: bytes) -> bytes:
        """
        HMAC-SHA256 伪随机函数
        
        Args:
            domain: 域分离标签 (防止不同用途的种子冲突)
            message: 输入消息 (通常是layer_id + block_idx)
        
        Returns:
            32字节的HMAC输出
        """
        # 组合域标签和消息
        full_message = domain + b'|' + message
        
        # 计算HMAC
        h = hmac.new(self.key, full_message, hashlib.sha256)
        return h.digest()
    
    def _seed_from_hmac(self, hmac_output: bytes) -> int:
        """
        从HMAC输出派生numpy随机种子
        
        Args:
            hmac_output: 32字节HMAC输出
        
        Returns:
            uint32范围内的整数种子
        """
        # 取前4字节转为整数
        seed = int.from_bytes(hmac_output[:4], byteorder='big')
        return seed % (2**32)  # 确保在uint32范围内
    
    def generate_Q(self, layer_id: str, block_idx: int, block_size: int, 
                   device='cpu') -> torch.Tensor:
        """
        生成块的正交变换矩阵 Q_{t,i}
        
        流程:
        1. seed = HMAC(K, "Q" | layer_id | block_idx)
        2. 用seed初始化PRNG生成随机矩阵A [c,c]
        3. QR分解: A = QR
        4. 调整符号保证唯一性
        
        Args:
            layer_id: 层标识字符串 (e.g., 'layer3.0.conv2')
            block_idx: 块索引 (0-based)
            block_size: 块大小 c
            device: 'cpu' or 'cuda'
        
        Returns:
            Q矩阵 [c, c], 满足 Q^T Q = I
        """
        # 1. 构造消息
        message = f"{layer_id}|{block_idx}".encode('utf-8')
        
        # 2. 计算HMAC并派生种子
        hmac_output = self._hmac(DOMAIN_Q, message)
        seed = self._seed_from_hmac(hmac_output)
        
        # 3. 用种子生成随机矩阵A
        rng = np.random.default_rng(seed)  # 确定性PRNG
        A = rng.standard_normal((block_size, block_size))  # 标准正态分布
        A = torch.from_numpy(A).float().to(device)
        
        # 4. QR分解
        Q, R = torch.linalg.qr(A)
        
        # 5. 调整符号保证唯一性 (让R对角线元素为正)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1  # 避免0的情况
        Q = Q * signs.unsqueeze(0)  # 广播到每一列
        
        return Q
    
    def generate_selection(self, layer_id: str, num_blocks: int, 
                          rho: float) -> List[int]:
        """
        生成选块集合 S_t
        
        流程:
        1. seed = HMAC(K, "S" | layer_id)
        2. 用seed洗牌 [0, 1, ..., g-1]
        3. 取前 m = floor(rho * g) 个作为S
        
        Args:
            layer_id: 层标识字符串
            num_blocks: 总块数 g
            rho: 选块比例 (0~1)
        
        Returns:
            选中的块索引列表(已排序,方便实现)
        """
        # 1. 计算HMAC并派生种子
        message = layer_id.encode('utf-8')
        hmac_output = self._hmac(DOMAIN_S, message)
        seed = self._seed_from_hmac(hmac_output)
        
        # 2. 洗牌
        rng = np.random.default_rng(seed)
        indices = np.arange(num_blocks)
        rng.shuffle(indices)
        
        # 3. 取前m个
        m = int(np.floor(rho * num_blocks))
        S = sorted(indices[:m].tolist())  # 排序方便后续处理
        
        return S
    
    def generate_permutation(self, layer_id: str, S: List[int], 
                           num_blocks: int) -> np.ndarray:
        """
        生成块置换 π_t
        只在选中的块集合S内进行置换,其他块保持不变
        
        流程:
        1. seed = HMAC(K, "P" | layer_id)
        2. 对S内的索引洗牌得到S_perm
        3. 构造置换数组: pi[S[k]] = S_perm[k]
        
        Args:
            layer_id: 层标识字符串
            S: 选中的块索引列表
            num_blocks: 总块数 g
        
        Returns:
            长度为g的置换数组,pi[i]表示块i应该移动到的位置
            未选中的块: pi[i] = i (恒等映射)
        """
        # 1. 计算HMAC并派生种子
        message = layer_id.encode('utf-8')
        hmac_output = self._hmac(DOMAIN_P, message)
        seed = self._seed_from_hmac(hmac_output)
        
        # 2. 洗牌S得到S_perm
        rng = np.random.default_rng(seed)
        S_array = np.array(S)
        S_perm = S_array.copy()
        rng.shuffle(S_perm)
        
        # 3. 构造完整的置换数组(初始化为恒等映射)
        pi = np.arange(num_blocks)
        
        # 4. 将S内的块映射到S_perm
        for k, s_idx in enumerate(S):
            pi[s_idx] = S_perm[k]
        
        return pi
    
    def invert_permutation(self, pi: np.ndarray) -> np.ndarray:
        """
        计算置换的逆
        
        Args:
            pi: 置换数组 [g]
        
        Returns:
            逆置换 pi_inv,满足 pi[pi_inv[i]] = i
        """
        pi_inv = np.empty_like(pi)
        pi_inv[pi] = np.arange(len(pi))
        return pi_inv


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    # 初始化密钥管理器
    key = '4a7d1ed414474e4033ac29ccb8653d9b'
    km = KeyManager(key)
    
    # 测试Q生成
    print("\n=== Test Q Generation ===")
    layer_id = 'layer3.0.conv2'
    Q1 = km.generate_Q(layer_id, block_idx=0, block_size=16)
    Q2 = km.generate_Q(layer_id, block_idx=0, block_size=16)
    print(f"Q shape: {Q1.shape}")
    print(f"Q^T Q = I: {torch.allclose(Q1.T @ Q1, torch.eye(16), atol=1e-5)}")
    print(f"Deterministic: {torch.allclose(Q1, Q2)}")
    
    # 测试选块集合生成
    print("\n=== Test Selection Generation ===")
    S = km.generate_selection(layer_id, num_blocks=16, rho=0.5)
    print(f"Selected blocks: {S}")
    print(f"Num selected: {len(S)} (expected {int(0.5*16)})")
    
    # 测试置换生成
    print("\n=== Test Permutation Generation ===")
    pi = km.generate_permutation(layer_id, S, num_blocks=16)
    print(f"Permutation: {pi}")
    print(f"Identity for unselected: {all(pi[i]==i for i in range(16) if i not in S)}")
    
    # 测试逆置换
    pi_inv = km.invert_permutation(pi)
    print(f"Inverse correct: {np.array_equal(pi[pi_inv], np.arange(16))}")