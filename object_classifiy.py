import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.cluster import KMeans
import joblib
# train_cluster.py
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
class MultiFolderImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.image_paths = []
        for root in root_dirs:
            self.image_paths.extend(
                sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
            )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.image_paths[idx]


class ImageKMeansCluster:
    def __init__(self, k, feature_extractor: nn.Module, device="cpu"):
        self.k = k
        self.device = device
        self.kmeans = None
        self.feat_net = feature_extractor.to(device).eval() if feature_extractor else nn.Identity()
        self.use_norm = feature_extractor is not None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _extract(self, x: torch.Tensor):
        x = x.to(self.device)
        if self.use_norm:
            x = (x - self.mean) / self.std
            feats = self.feat_net(x)
            if feats.dim() == 4:
                feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            feats = x.flatten(1)
        return feats.cpu()

    def fit_from_dataloader(self, dataloader):
        all_feats, all_paths = [], []
        for imgs, paths in dataloader:
            feats = self._extract(imgs)
            all_feats.append(feats)
            all_paths.extend(paths)
        all_feats = torch.cat(all_feats, dim=0)
        self.kmeans = KMeans(n_clusters=self.k, n_init="auto", random_state=42).fit(all_feats)
        return all_paths, torch.tensor(self.kmeans.labels_, dtype=torch.long)

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        feats = self._extract(imgs)
        return torch.tensor(self.kmeans.predict(feats), dtype=torch.long)

    def save(self, path):
        joblib.dump(self.kmeans, path)

    def load(self, path):
        self.kmeans = joblib.load(path)


def train_cluster(data_roots, k=10, batch_size=64, model_path="kmeans_model.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    dataset = MultiFolderImageDataset(data_roots, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()

    clusterer = ImageKMeansCluster(k=k, feature_extractor=resnet, device=device)
    paths, labels = clusterer.fit_from_dataloader(dataloader)

    clusterer.save(model_path)

    print("Saved KMeans model to", model_path)
    for path, label in zip(paths[:10], labels[:10]):
        print(f"{path} → cluster {label}")

def infer_single_image(img_path, model_path="kmeans_model.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # (1,3,H,W)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()

    clusterer = ImageKMeansCluster(k=1, feature_extractor=resnet, device=device)
    clusterer.load(model_path)

    label = clusterer.predict(img_tensor)

    print(f"{img_path} → cluster {label.item()}")
    return label.item()
@torch.no_grad()
def infer_batch_images(batch_imgs: torch.Tensor, model_path="kmeans_model.pkl", device=None):

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()

    # Init and load KMeans model
    clusterer = ImageKMeansCluster(k=1, feature_extractor=resnet, device="cuda")
    clusterer.load(model_path)

    # Predict
    labels = clusterer.predict(batch_imgs)
    return labels.tolist()
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg → cluster 7
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-3.jpg → cluster 2
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-7.jpg → cluster 3
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-8.jpg → cluster 3
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-9.jpg → cluster 2
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-415.jpg → cluster 2
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-417.jpg → cluster 6
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-418.jpg → cluster 0
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-420.jpg → cluster 2
# ../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-10-LeafySeaDragon-421.jpg → cluster 7

if __name__ == "__main__":
    import os
    import shutil
    from tqdm import tqdm
    #labales:0,1,2,3
    folders = ["../data/train/COD-TrainDataset/Imgs", "../data/TestDataset/COD10K/Imgs","../data/TestDataset/CAMO/Imgs","../data/TestDataset/NC4K/Imgs"]
    folders_gt = ["../data/train/COD-TrainDataset/GT", "../data/TestDataset/COD10K/GT",
               "../data/TestDataset/CAMO/GT", "../data/TestDataset/NC4K/GT"]
    #
    # lable = infer_single_image("../data/train/COD-TrainDataset/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-7.jpg", model_path="kmeans_model.pkl")
    # print(lable)
    # 遍历每个主目录
    # for folder in folders:
    #     print(f"Processing folder: {folder}")
    #
    #     # 获取所有图像路径
    #     image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder)
    #                    if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]
    #
    #     for image_path in tqdm(image_paths, desc=f"Processing images in {os.path.basename(folder)}"):
    #         # 获取标签
    #         label = infer_single_image(image_path)
    #         label = str(label)  # 转为字符串用于目录名
    #
    #         # 目标子目录
    #         label_folder = os.path.join(folder, label)
    #         os.makedirs(label_folder, exist_ok=True)
    #
    #         # 目标路径
    #         target_path = os.path.join(label_folder, os.path.basename(image_path))
    #
    #         # 拷贝图像
    #         shutil.copy2(image_path, target_path)
    #

    # labels = ['0', '1', '2', '3']

    # # 遍历每组 Img 和 GT 文件夹
    # for folder_img, folder_gt in zip(folders, folders_gt):
    #     print(f"\nProcessing: {folder_img} -> {folder_gt}")
    #
    #     for label in labels:
    #         label_img_dir = os.path.join(folder_img, label)
    #         label_gt_dir = os.path.join(folder_gt, label)
    #         os.makedirs(label_gt_dir, exist_ok=True)
    #
    #         if not os.path.exists(label_img_dir):
    #             continue  # 若该标签文件夹不存在，则跳过
    #
    #         img_files = [f for f in os.listdir(label_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    #
    #         for img_file in tqdm(img_files, desc=f"Label {label}"):
    #             # 获取文件名但改为 .png 后缀
    #             base_name = os.path.splitext(img_file)[0] + '.png'
    #             src_gt_path = os.path.join(folder_gt, base_name)
    #             dst_gt_path = os.path.join(label_gt_dir, base_name)
    #
    #             # 如果 GT 文件存在，则拷贝
    #             if os.path.exists(src_gt_path):
    #                 shutil.copy2(src_gt_path, dst_gt_path)
    #             else:
    #                 print(f"Warning: GT not found for {img_file} at {src_gt_path}")

    train_cluster(folders, k=4, batch_size=512, model_path="kpl/kmeans_model.pkl")
    # B = 8
    # imgs = torch.rand(B, 3, 384, 384)  # example batch input
    # labels = infer_batch_images(imgs, model_path="kmeans_model.pkl")
    # print("Predicted cluster labels:", labels)