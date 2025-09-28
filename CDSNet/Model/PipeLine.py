import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from methods.zoomnext.shufflenetv2 import shufflenet_v2_x1_0


# 轻量化分类模型
# def train_and_predict(data, n_clusters=10, device='cpu'):
#     """训练聚类模型并返回预测结果"""
#     model = shufflenet_v2_x1_0().to(device) # 1. 初始化模型并移动到指定设备
#     if isinstance(data, torch.Tensor):
#         data = data.to(device)  # 2. 确保数据也在同一设备上
#     else:
#         data = torch.tensor(data).to(device) # 如果data是其他类型(如numpy数组)，先转换为tensor
#     # 3. 提取特征
#     with torch.no_grad():
#         features = model(data).cpu().numpy()  # 计算完成后移回CPU并转换为numpy
#     # 4. 训练K-means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(features)  # 训练
#     # 5. 预测类别
#     cluster_ids = kmeans.predict(features)
#     return model, kmeans, cluster_ids



# def unsupervised_clustering(data, n_clusters=10):
#     # 1. 初始化模型
#     model = LightweightAE()
#     model.to("cuda")
#
#     # 2. 训练自编码器
#     print("Training Autoencoder...")
#     train_ae(model, data)
#
#     # 3. 提取特征
#     with torch.no_grad():
#         data = data.to("cuda")
#         features = model.encoder(data)
#         print("Encoder 输出形状:", features.shape)  # 检查维度
#
#         # 强制展平：确保 (B, C, H, W) → (B, C*H*W)
#         features_flat = features.reshape(features.size(0), -1).cpu().numpy()
#         print("展平后 NumPy 数组形状:", features_flat.shape)  # 应该是 (B, D)
#
#     # 4. K-means聚类
#     print("Performing K-means clustering...")
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(features_flat)  # 输入必须是 (n_samples, n_features)
#
#     return model, kmeans, clusters


import torch.optim as optim
from tqdm import tqdm  # 用于显示训练进度


def train_and_predict(data, n_clusters=10, device='cpu',
                      mode='feature_extraction', epochs=10,
                      lr=0.001, batch_size=32):

    model = shufflenet_v2_x1_0().to(device)

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    if mode == 'feature_extraction':
        with torch.no_grad():
            data = data.to(device)
            features = model(data).cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        cluster_ids = kmeans.predict(features)

        return model, kmeans, cluster_ids

    elif mode == 'end_to_end':
        # 准备数据加载器
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练循环
        model.train()
        for epoch in range(epochs):
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                inputs = batch[0].to(device)

                # 前向传播
                outputs = model(inputs)

                loss = torch.norm(outputs, p=2)  # 示例损失，实际应根据任务修改

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item())

        with torch.no_grad():
            model.eval()
            features = []
            for batch in dataloader:
                inputs = batch[0].to(device)
                features.append(model(inputs).cpu())
            features = torch.cat(features).numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        cluster_ids = kmeans.predict(features)

        return model, kmeans, cluster_ids

    else:
        raise ValueError("模式必须是 'feature_extraction' 或 'end_to_end'")

class LiteDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # [B,16,192,192]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # [B,32,96,96]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # [B,64,48,48]
            nn.ReLU()
        )

        # 检测头
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, (5 + num_classes) * 3, 1)  # 3个anchor
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class SimpleSegModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # 编码器 (下采样)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 解码器 (上采样)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid()  # 输出[0,1]的概率图
        )

    def forward(self, x):
        # 编码
        x1 = self.enc1(x)  # [B,32,H/2,W/2]
        x2 = self.enc2(x1)  # [B,64,H/4,W/4]

        # 解码
        x = self.dec1(x2)  # [B,32,H/2,W/2]
        x = self.dec2(x)  # [B,C,H,W]

        return x

def full_pipeline(input_data, cluster_model, kmeans, det_models, seg_model):

    with torch.no_grad():
        features = cluster_model(input_data)  # 现在输出 [B,32]

    # 转换为numpy并确保是2D
    features_np = features.cpu().numpy()
    if features_np.ndim > 2:
        features_np = features_np.reshape(features_np.shape[0], -1)  # 强制展平

    cluster_ids = kmeans.predict(features_np)  # 输入形状 [B, n_features]

    results = []
    for img, cls_id in zip(input_data, cluster_ids):
        # 第二步：选择对应检测模型
        det_model = det_models[cls_id % len(det_models)]

        # 目标检测（简化版）
        with torch.no_grad():
            pred = det_model(img.unsqueeze(0))
            bbox = non_max_suppression(pred)[0][0][:4].int()  # 假设的检测输出

        # 第三步：裁剪
        x1, y1, x2, y2 = bbox.tolist()
        cropped = img[:, max(0, y1):min(img.shape[1], y2),
                  max(0, x1):min(img.shape[2], x2)]  # 添加边界保护

        # 第四步：分割
        with torch.no_grad():
            seg_input = F.interpolate(cropped.unsqueeze(0), size=256)
            mask = seg_model(seg_input)

        results.append({
            'class': cls_id,
            'bbox': bbox.tolist(),
            'mask': mask.squeeze().cpu().numpy()
        })

    return results


# 辅助函数（简化版NMS）
def non_max_suppression(pred, conf_thres=0.5):
    # 实现简化的NMS
    return [torch.tensor([[0, 0, 100, 100, 0.9]])]  # 示例返回

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime


def visualize_clusters(data, clusters, n_clusters=10, save_dir="cluster_results"):

    os.makedirs(save_dir, exist_ok=True)

    # 专业级输出配置
    rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 7,
        'figure.titlesize': 8,
        'axes.titlepad': 4
    })

    # 数据预处理（支持Tensor和字典输入）
    if isinstance(data, torch.Tensor):
        images = data.permute(0, 2, 3, 1).cpu().numpy()
    elif isinstance(data, dict):
        tensor_data = torch.stack(list(data.values())).cpu()
        images = tensor_data.permute(0, 2, 3, 1).numpy()
    else:
        raise ValueError("输入必须是PyTorch Tensor或包含Tensor的字典")

    # 智能归一化（兼容各种输入范围）
    if images.dtype in (np.float32, np.float64):
        if images.max() > 1.0: images /= 255.0
        images = np.clip(images, 0, 1)
    elif images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    # 基于原始尺寸的动态画布计算
    img_height = images.shape[1] / 80  # 标准化系数
    figsize = (img_height * 2.2, img_height * 2.2)  # 黄金比例

    # 聚类可视化保存
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if not len(cluster_indices): continue

        fig = plt.figure(figsize=figsize, dpi=300, facecolor='white')

        # 绘制子图（优化布局）
        for i, idx in enumerate(cluster_indices[:4]):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(images[idx],
                      interpolation='hanning',  # 高级插值算法
                      aspect='equal')  # 保持像素方形
            ax.axis('off')

        # 专业级标题排版
        plt.suptitle(
            f'Cluster {cluster_id}\n({len(cluster_indices)} samples)',
            y=0.92,
            fontsize=8,
            fontweight='bold'
        )

        # 保存设置
        save_path = os.path.join(save_dir, f"cluster_{cluster_id}.png")
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0.05,
            transparent=False,
            metadata={
                'CreationTime': datetime.now().isoformat(),
                'Software': 'Matplotlib/Seaborn'
            }
        )
        plt.close(fig)

    print(f"🖼️ 专业级可视化结果已保存至:\n{os.path.abspath(save_dir)}")
