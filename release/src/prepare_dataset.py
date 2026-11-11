"""下载一个公开数据集并整理成本项目使用的 ImageFolder 格式。

用法
----
python prepare_dataset.py --name dtd           # 纹理数据集，47 类，约 600MB
python prepare_dataset.py --name flowers102    # 花卉，102 类，约 330MB
python prepare_dataset.py --name caltech101    # 物体，101 类，约 130MB

整理结果：code/data/<name>/<类别名>/*.jpg
之后就可以直接：
    python -m src.train --dataset <name> --method paa ...
"""
import argparse
import os
import shutil
import sys

import torchvision.datasets as tvd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# name -> (torchvision 类名, 是否有官方划分, 大致体积)
SPECS = {
    "dtd": ("DTD", True, "约 600MB，47 类纹理"),
    "flowers102": ("Flowers102", True, "约 330MB，102 类花卉"),
    "caltech101": ("Caltech101", False, "约 130MB，101 类物体"),
    "fgvc-aircraft": ("FGVCAircraft", True, "约 1.7GB，102 类飞机"),
}

OFFICIAL_SPLITS = ("train", "val", "test")


def _paths_of(ds):
    """尽量拿到原始文件路径，避免重新编码导致画质损失。"""
    for attr in ("samples", "imgs", "_image_files", "_data"):
        v = getattr(ds, attr, None)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, (tuple, list)) and len(first) >= 2:
                return [str(p) for p, _ in v]
            if isinstance(first, str):
                return [str(p) for p in v]
    return None


def prepare(name: str, overwrite: bool = False) -> None:
    if name not in SPECS:
        print(f"不支持的数据集 {name}，可选：{', '.join(SPECS)}")
        sys.exit(1)
    cls_name, has_split, size = SPECS[name]
    out_dir = os.path.join(DATA_ROOT, name)
    raw_root = os.path.join(DATA_ROOT, "_raw")

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        n = sum(len(os.listdir(os.path.join(out_dir, c))) for c in os.listdir(out_dir))
        print(f"[跳过] {out_dir} 已存在，共 {n} 张图。要重新整理请加 --overwrite")
        return

    print(f"[1/3] 下载 {name}（{size}），首次运行需要几分钟……")
    splits = OFFICIAL_SPLITS if has_split else (None,)
    idx = 0
    classes = []

    for sp in splits:
        kw = {"split": sp} if sp else {}
        ds = getattr(tvd, cls_name)(root=raw_root, download=True, **kw)
        if not classes:
            classes = list(getattr(ds, "classes", []))
            for c in classes:
                os.makedirs(os.path.join(out_dir, c.replace("/", "_")), exist_ok=True)
            print(f"[2/3] 共 {len(classes)} 类")

        targets = None
        for attr in ("targets", "labels", "_labels", "_targets"):
            v = getattr(ds, attr, None)
            if isinstance(v, list) and len(v) == len(ds):
                targets = list(v)
                break
        if targets is None:
            targets = [int(ds[i][1]) for i in range(len(ds))]

        paths = _paths_of(ds)
        print(f"[3/3] 整理 {sp or 'data'} 划分：{len(ds)} 张……")
        for i in range(len(ds)):
            label = int(targets[i])
            dst_dir = os.path.join(out_dir, classes[label].replace("/", "_"))
            if paths is not None:
                ext = os.path.splitext(paths[i])[1].lower() or ".jpg"
                dst = os.path.join(dst_dir, f"{idx:06d}{ext}")
                if not os.path.exists(dst):
                    shutil.copyfile(paths[i], dst)
            else:
                dst = os.path.join(dst_dir, f"{idx:06d}.jpg")
                if not os.path.exists(dst):
                    img, _ = ds[i]
                    img.convert("RGB").save(dst, quality=95)
            idx += 1
            if idx % 1000 == 0:
                print(f"       {idx} 张")

    print(f"\n完成。目录：{out_dir}，共 {idx} 张，{len(classes)} 类")
    print(f"训练命令示例：python -m src.train --dataset {name} --method paa --ratio 0.01 --epochs 20")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(SPECS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    prepare(args.name, args.overwrite)


if __name__ == "__main__":
    main()
