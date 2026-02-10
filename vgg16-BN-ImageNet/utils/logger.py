"""
日志工具模块
负责:
1. 训练过程日志记录
2. 最优模型追踪与保存（纯 test_acc）
3. 指标历史记录
4. CSV格式输出（方便论文作图）

输出文件:
  train_base.py  -> training_history(base).csv  /  best_results(base).csv
  train_confuse  -> training_history(confuse).csv / best_results(confuse).csv

best_results(confuse).csv 字段:
  timestamp, best_epoch, best_test_acc,
  lr, epochs, batch_size, weight_decay, lambda, alpha_hi, alpha_lo, rho,
  每层 × 每个指标 × 4 列（均匀结构）:
    <layer>_<metric>_base(origin)       ← base 模型 raw 值
    <layer>_<metric>_base(normalized)   ← base 模型 normalized 值
    <layer>_<metric>(origin)            ← best_epoch 时 current raw 值
    <layer>_<metric>(normalized)        ← best_epoch 时 current normalized 值
  指标顺序: homogeneity, overall_similarity, selected_similarity,
            unselected_similarity, cross_similarity
"""

import os
import json
import csv
import torch
from typing import Dict, Optional
from datetime import datetime


class Logger:
    """
    训练日志记录器
    自动追踪最优模型并保存
    支持CSV格式实时输出
    """

    # 指标维度名，按此顺序排列每层的列
    _METRIC_NAMES = [
        'homogeneity',
        'overall_similarity',
        'selected_similarity',
        'unselected_similarity',
        'cross_similarity',
    ]

    def __init__(self, log_dir: str, checkpoint_dir: str,
                 file_suffix: str = ''):
        """
        Args:
            log_dir: 日志保存目录
            checkpoint_dir: checkpoint保存目录
            file_suffix: 文件后缀, e.g. '(base)' 或 '(confuse)'
        """
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.file_suffix = file_suffix

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 最优模型追踪（纯 test_acc）
        self.best_acc = 0.0
        self.best_epoch = 0

        # 历史记录（内存中保留）
        self.history = {}

        # CSV实时写入状态
        self._csv_initialized = False
        self._csv_fieldnames = None

        self.config = {}

        # --- confuse 专用：base 基准指标 ---
        # 结构: {layer_name: {'raw': {metric: val}, 'normalized': {metric: val}}}
        self.base_metrics = {}

        # --- confuse 专用：每次评估的快照 ---
        # {epoch: {layer_name: {'raw': {...}, 'normalized': {...}}}}
        self.metrics_snapshots = {}

        self.start_time = datetime.now()

    # ----------------------------------------------------------
    # 配置
    # ----------------------------------------------------------
    def set_config(self, config: Dict):
        """设置训练配置"""
        self.config = config

    def store_base_metrics(self, metrics_summary: Dict):
        """
        记录 base 模型的混淆指标基准（训练前调用一次）

        Args:
            metrics_summary: evaluate_confuse_metrics 返回的结构
                {layer_name: {'raw': {...}, 'normalized': {...}}}
        """
        self.base_metrics = metrics_summary

    def store_metrics_snapshot(self, epoch: int, metrics_summary: Dict):
        """
        记录当前 epoch 的混淆指标快照（每次评估时调用）

        Args:
            epoch: 当前 epoch
            metrics_summary: evaluate_confuse_metrics 返回的结构
        """
        self.metrics_snapshots[epoch] = metrics_summary

    # ----------------------------------------------------------
    # 日志
    # ----------------------------------------------------------
    def log(self, message: str):
        """打印并记录日志"""
        print(message)

    # ----------------------------------------------------------
    # 历史记录
    # ----------------------------------------------------------
    def update(self, epoch: int, metrics: Dict[str, float]):
        """更新指标历史（内存中记录）"""
        if 'epoch' not in self.history:
            self.history['epoch'] = []
        self.history['epoch'].append(epoch)

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def append_history_csv(self, epoch: int, metrics: Dict[str, float]):
        """
        实时写入一行到 training_history{suffix}.csv
        首次调用写 header 并清空文件, 后续追加
        """
        csv_path = os.path.join(
            self.log_dir,
            f'training_history{self.file_suffix}.csv'
        )

        row = {'epoch': epoch}
        row.update(metrics)

        if not self._csv_initialized:
            self._csv_fieldnames = ['epoch'] + list(metrics.keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
                writer.writerow(row)

    # ----------------------------------------------------------
    # 最优模型保存（纯 test_acc）
    # ----------------------------------------------------------
    def save_best_model(self, model: torch.nn.Module, epoch: int,
                        test_acc: float, save_name: str) -> bool:
        """
        保存最优模型（按 test_acc 决定）

        Returns:
            bool: 是否保存了新的最佳模型
        """
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.best_epoch = epoch

            save_path = os.path.join(self.checkpoint_dir, save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, save_path)

            self.log(f"[SaveBest] Epoch {epoch}: test_acc={test_acc:.2f}% "
                     f"(new best, saved to {save_name})")
            return True
        return False

    # ----------------------------------------------------------
    # best_results CSV
    # ----------------------------------------------------------
    def _get_best_epoch_snapshot(self) -> Optional[Dict]:
        """
        从 metrics_snapshots 中找到 best_epoch 对应的快照。
        如果没有精确匹配，取 <= best_epoch 的最大 epoch 的快照。
        """
        if not self.metrics_snapshots:
            return None
        candidates = [e for e in self.metrics_snapshots if e <= self.best_epoch]
        if not candidates:
            # fallback: 取最小 epoch
            candidates = sorted(self.metrics_snapshots.keys())
        chosen = max(candidates)
        return self.metrics_snapshots[chosen]

    def save_best_results_csv(self, model_type: str):
        """
        保存最佳结果到 best_results{suffix}.csv

        base 模型: timestamp, best_epoch, best_test_acc, lr, epochs,
                   batch_size, weight_decay
        confuse 模型: 上述超参 + lambda/alpha/rho +
                      每层的 5 个指标 × 3 列(base / origin / normalized)
        """
        csv_path = os.path.join(
            self.log_dir,
            f'best_results{self.file_suffix}.csv'
        )

        file_exists = os.path.exists(csv_path)

        # --- 基础字段 ---
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_epoch': self.best_epoch,
            'best_test_acc': f"{self.best_acc:.2f}",
        }

        # --- 超参字段 ---
        if self.config:
            if model_type == 'base':
                result['lr'] = self.config.get('training', {}).get('lr_base', '')
                result['epochs'] = self.config.get('training', {}).get('epochs_base', '')
            else:
                result['lr'] = self.config.get('training', {}).get('lr_confuse', '')
                result['epochs'] = self.config.get('training', {}).get('epochs_confuse', '')
                result['lambda'] = self.config.get('confuse', {}).get('lambda', '')
                result['alpha_hi'] = self.config.get('confuse', {}).get('alpha_hi', '')
                result['alpha_lo'] = self.config.get('confuse', {}).get('alpha_lo', '')
                result['rho'] = self.config.get('confuse', {}).get('rho', '')

            result['batch_size'] = self.config.get('dataset', {}).get('batch_size', '')
            result['weight_decay'] = self.config.get('training', {}).get('weight_decay', '')

        # --- 字段顺序构建 ---
        fieldnames = [
            'timestamp', 'best_epoch', 'best_test_acc',
            'lr', 'epochs', 'batch_size', 'weight_decay',
        ]
        if model_type == 'confuse':
            fieldnames.extend(['lambda', 'alpha_hi', 'alpha_lo', 'rho'])

            # --- 每层的指标列 ---
            # 取 base_metrics 的层名作为权威层列表（已排序）
            layer_names = sorted(self.base_metrics.keys()) if self.base_metrics else []

            # best_epoch 对应的 current 指标快照
            current_snapshot = self._get_best_epoch_snapshot()

            for layer in layer_names:
                short = layer.replace('.', '_')

                for m in self._METRIC_NAMES:
                    # 四列: _base(origin), _base(normalized), (origin), (normalized)
                    col_base_origin = f'{short}_{m}_base(origin)'
                    col_base_norm   = f'{short}_{m}_base(normalized)'
                    col_origin      = f'{short}_{m}(origin)'
                    col_norm        = f'{short}_{m}(normalized)'

                    fieldnames.extend([col_base_origin, col_base_norm,
                                       col_origin, col_norm])

                    # base 值（raw + normalized）
                    if layer in self.base_metrics:
                        result[col_base_origin] = f"{self.base_metrics[layer]['raw'][m]:.4f}"
                        result[col_base_norm]   = f"{self.base_metrics[layer]['normalized'][m]:.4f}"

                    # current 值（origin = raw, normalized）
                    if current_snapshot and layer in current_snapshot:
                        result[col_origin] = f"{current_snapshot[layer]['raw'][m]:.4f}"
                        result[col_norm]   = f"{current_snapshot[layer]['normalized'][m]:.4f}"

        # --- 写入 ---
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

        self.log(f"Best results saved to: {csv_path}")

    # ----------------------------------------------------------
    # 其他
    # ----------------------------------------------------------


    def summary(self):
        """打印训练总结"""
        duration = datetime.now() - self.start_time

        self.log("\n" + "=" * 60)
        self.log("Training Summary")
        self.log("=" * 60)
        self.log(f"Best test accuracy: {self.best_acc:.2f}%")
        self.log(f"Best epoch: {self.best_epoch}")
        self.log(f"Total epochs: {len(self.history.get('epoch', []))}")
        self.log(f"Training duration: {str(duration).split('.')[0]}")
        self.log("=" * 60 + "\n")