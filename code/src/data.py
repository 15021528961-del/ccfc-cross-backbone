"""数据集构建：统一的 (image, label, index) 接口 + 分层划分 + 标签预算采样。

支持的数据集
------------
eurosat      : torchvision 内置，自动下载（遥感，10 类）
isic2019     : ImageFolder 格式，需自行下载整理（皮肤病变，8 类）
plantvillage : ImageFolder 格式，需自行下载整理（植物病害，38 类）

所有数据集的目录约定：
    {data_root}/eurosat/eurosat/2750/ ...      （torchvision 自动处理）
    {data_root}/isic2019/{class_name}/*.jpg    （ImageFolder）
    {data_root}/plantvillage/{class_name}/*.jpg
"""
import json
import os
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets

KNOWN_DATASETS = ("eurosat", "isic2019", "plantvillage")


class IndexedDataset(Dataset):
    """在 (image, label) 之外附带原始索引，便于按索引切分与复现。"""

    def __init__(self, base: Dataset, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, label = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx


def _download_eurosat(root: str) -> None:
    """下载并解压 EuroSAT。

    torchvision 内置的下载源是 HuggingFace，国内经常超时，
    所以这里依次尝试镜像源和官方源，都失败时给出明确的报错。
    """
    base = os.path.join(root, "eurosat")
    target = os.path.join(base, "2750")
    if os.path.isdir(target):
        return
    os.makedirs(base, exist_ok=True)

    rev = "c877bcd43f099cd0196738f714544e355477f3fd"
    urls = [
        f"https://hf-mirror.com/datasets/torchgeo/eurosat/resolve/{rev}/EuroSAT.zip",
        f"https://huggingface.co/datasets/torchgeo/eurosat/resolve/{rev}/EuroSAT.zip",
    ]
    zip_path = os.path.join(base, "EuroSAT.zip")
    last_err = None
    for url in urls:
        try:
            print(f"[data] 下载 EuroSAT: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as resp, open(zip_path, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r[data] {done/1e6:.1f}/{total/1e6:.1f} MB",
                              end="", flush=True)
            print()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"\n[data] 该源失败：{e}")
    else:
        raise RuntimeError(
            f"EuroSAT 全部下载源失败：{last_err}\n"
            f"请手动下载 EuroSAT.zip 解压，确保存在 {target}"
        )

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(base)
    if not os.path.isdir(target):
        for cand in ("EuroSAT/2750", "eurosat/2750"):
            p = os.path.join(base, cand)
            if os.path.isdir(p):
                shutil.move(p, target)
                break
    try:
        os.remove(zip_path)
    except OSError:
        pass
    if not os.path.isdir(target):
        raise RuntimeError(f"解压后未找到 {target}，请检查压缩包结构")


def load_base_dataset(name: str, root: str):
    """返回 (dataset, class_names)。dataset 需提供 .targets 或 .labels。"""
    name = name.lower()
    if name == "eurosat":
        # 注意：torchvision 的 EuroSAT 会在 root 下自动再加一层 "eurosat"，
        # 所以这里传 data_root 而不是 data_root/eurosat，
        # 最终路径是 {data_root}/eurosat/2750/<类别>/*.jpg
        _download_eurosat(root)
        ds = datasets.EuroSAT(root=root, download=False)
        return ds, list(ds.classes)

    # 其余数据集一律按 ImageFolder 读取：{data_root}/{name}/<类别>/*.jpg
    # 这样只要用 prepare_dataset.py 整理出一个新目录，就能直接训练。
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"未找到 {path}。请先运行 python prepare_dataset.py --name {name}，"
            f"或按 ImageFolder 格式手动整理：{path}/<类别名>/*.jpg"
        )
    ds = datasets.ImageFolder(root=path)
    return ds, list(ds.classes)


def _targets(ds: Dataset) -> List[int]:
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "labels"):
        return list(ds.labels)
    if hasattr(ds, "samples"):
        return [s[1] for s in ds.samples]
    raise AttributeError("数据集缺少 targets/labels/samples 字段")


def stratified_split(
    targets: Sequence[int], val_frac: float, test_frac: float, seed: int
) -> Dict[str, List[int]]:
    """按类别分层划分，返回 {"train": [...], "val": [...], "test": [...]}。"""
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(targets):
        by_class[int(y)].append(i)

    splits = {"train": [], "val": [], "test": []}
    for _, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        splits["test"].extend(idxs[:n_test].tolist())
        splits["val"].extend(idxs[n_test : n_test + n_val].tolist())
        splits["train"].extend(idxs[n_test + n_val :].tolist())
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def get_or_make_splits(
    name: str, targets: Sequence[int], cache_dir: str, val_frac: float, test_frac: float, seed: int
) -> Dict[str, List[int]]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_split_s{seed}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    splits = stratified_split(targets, val_frac, test_frac, seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    return splits


def stratified_subsample(
    indices: Sequence[int], targets: Sequence[int], ratio: float, seed: int
) -> List[int]:
    """从 train 索引中按类别等比例抽取 ratio（例如 0.01 表示每类 1%）。"""
    if ratio >= 1.0:
        return sorted(indices)
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i in indices:
        by_class[int(targets[i])].append(i)
    out: List[int] = []
    for _, idxs in sorted(by_class.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * ratio)))
        out.extend(idxs[:k])
    return sorted(out)


def build_datasets(
    name: str,
    root: str,
    transform_train,
    transform_eval,
    cache_dir: str,
    label_ratio: float,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[IndexedDataset, IndexedDataset, IndexedDataset, List[str]]:
    base, classes = load_base_dataset(name, root)
    targets = _targets(base)
    splits = get_or_make_splits(name, targets, cache_dir, val_frac, test_frac, seed)
    train_idx = stratified_subsample(splits["train"], targets, label_ratio, seed)
    train = IndexedDataset(base, train_idx, transform_train)
    val = IndexedDataset(base, splits["val"], transform_eval)
    test = IndexedDataset(base, splits["test"], transform_eval)
    return train, val, test, classes
