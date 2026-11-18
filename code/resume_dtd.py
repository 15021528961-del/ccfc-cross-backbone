"""续传 DTD 下载并整理成 ImageFolder 格式（torchvision 不支持断点续传，所以单独写）。

用法
----
python resume_dtd.py             # 续传 + 解压 + 整理
python resume_dtd.py --arrange   # 跳过下载，只解压整理
"""
import argparse
import os
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "_raw", "dtd")
TAR = os.path.join(RAW, "dtd-r1.0.1.tar.gz")
OUT = os.path.join(DATA, "dtd")
URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"


def size_of(p: str) -> int:
    return os.path.getsize(p) if os.path.exists(p) else 0


def resume_download() -> None:
    import requests

    os.makedirs(RAW, exist_ok=True)
    pos = size_of(TAR)
    if pos:
        print(f"[续传] 已有 {pos/1e6:.0f}MB，从断点继续")

    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:
            # 请求的范围超出文件末尾，说明本地文件已经完整
            print(f"[跳过] 本地压缩包已是完整大小（{pos/1e6:.0f}MB），无需续传")
            return
        total = None
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                total = int(cr.split("/")[-1])
            mode = "ab"
        elif r.status_code == 200:
            print("[注意] 服务器不支持断点，只能从头下载")
            pos, mode = 0, "wb"
        else:
            print(f"[错误] HTTP {r.status_code}")
            sys.exit(1)

        r.raise_for_status()
        done = pos
        t0 = time.time()
        with open(TAR, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (20 << 20) < (1 << 20):
                    speed = (done - pos) / max(time.time() - t0, 1) / 1e6
                    print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB  {speed:.2f} MB/s")

    print(f"[完成] 文件大小 {size_of(TAR)/1e6:.0f}MB")


def extract_and_arrange() -> None:
    # 已经整理好就跳过，避免每次都重新复制 5600 多个文件
    if os.path.isdir(OUT) and os.listdir(OUT):
        n = sum(len(fs) for _, _, fs in os.walk(OUT))
        print(f"[跳过] {OUT} 已存在 {n} 张图，不用重新整理")
        return

    if not os.path.exists(TAR):
        print("[错误] 找不到压缩包，先跑下载")
        sys.exit(1)

    # 校验：能否完整打开
    try:
        with tarfile.open(TAR) as tf:
            members = tf.getnames()
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 压缩包不完整（{e}），请重新运行本脚本续传")
        sys.exit(1)

    print(f"[解压] 共 {len(members)} 个条目")
    img_root = None
    with tarfile.open(TAR) as tf:
        tf.extractall(RAW)
    for cand in ("dtd/images", "images"):
        p = os.path.join(RAW, cand)
        if os.path.isdir(p):
            img_root = p
            break
    if img_root is None:
        print("[错误] 解压后找不到 images 目录")
        sys.exit(1)

    classes = sorted(d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d)))
    print(f"[整理] {len(classes)} 类 -> {OUT}")
    n = 0
    for c in classes:
        dst_dir = os.path.join(OUT, c)
        os.makedirs(dst_dir, exist_ok=True)
        src_dir = os.path.join(img_root, c)
        for fn in os.listdir(src_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copyfile(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
                n += 1
    print(f"[完成] 共 {n} 张图，{len(classes)} 类")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrange", action="store_true", help="跳过下载，只解压整理")
    args = ap.parse_args()

    if not args.arrange:
        if os.path.isdir(OUT) and os.listdir(OUT):
            print("[跳过] 数据已就绪，不需要再下载")
        else:
            resume_download()
    extract_and_arrange()


if __name__ == "__main__":
    main()
