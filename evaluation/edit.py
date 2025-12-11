#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评测脚本（不计算 CLAP）
- 仅计算分布类指标：FAD / FD / KL（依赖 audioldm_eval.EvaluationHelper）
- 不读取 caption、不加载 laion-clap、不计算文本-音频相似度

用法示例：
python evaluation/tta_noclap.py \
  --ref_audio_jsonl /mnt/petrelfs/taoye/workspace/editing/data/test.jsonl \
  --gen_audio_jsonl /mnt/petrelfs/taoye/workspace/editing/exp/mmdit2/inference/test.jsonl \
  --output_file /mnt/petrelfs/taoye/workspace/editing/exp/mmdit2/inference/tta.jsonl \
  -c 16

或（生成音频来自目录）：
python evaluation/tta_noclap.py \
  --ref_audio_jsonl /path/to/ref/audio.jsonl \
  --gen_audio_dir   /path/to/gen/audio_dir \
  --output_file     /path/to/out/tta.jsonl \
  -c 16
"""

import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
from copy import deepcopy

import torch

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
from audioldm_eval import EvaluationHelper
import sys
sys.path.append("/mnt/petrelfs/taoye/workspace/mmdit")

# 你的工具函数：把 JSONL/目录 转成 {audio_id: audio_path} 映射
from utils.general import read_jsonl_to_mapping, audio_dir_to_mapping


torch.multiprocessing.set_sharing_strategy('file_system')

def create_subset_symlink_folder(src_folder: str, keep_paths: set[str]) -> str:
    src = Path(src_folder).resolve()
    out = src.parent / f"{src.name}_subset_link"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    keep_names = {Path(p).name for p in keep_paths}
    for f in src.iterdir():
        if f.is_file() and f.name in keep_names:
            (out / f.name).symlink_to(f.resolve())
    return str(out)


# def create_symlink_folder(src_folder: str) -> str:
#     """
#     为 src_folder 创建一个同级的软链接目录（不改名，按原文件名链接），避免后续评测修改原目录结构。
#     返回：软链接目录的绝对路径
#     """
#     src = Path(src_folder).resolve()
#     parent = src.parent
#     link_dir = parent / f"{src.name}_link"

#     # 如果存在旧的链接目录，先删掉
#     if link_dir.exists():
#         shutil.rmtree(link_dir)
#     link_dir.mkdir(parents=True, exist_ok=True)

#     # 仅为普通文件创建软链接；保留原文件名，避免截断导致的重名覆盖风险
#     for f in src.iterdir():
#         if f.is_file():
#             (link_dir / f.name).symlink_to(f.resolve())

#     return str(link_dir)


def get_common_folder_path(audio_dict: dict[str, str]) -> tuple[str | None, bool]:
    """
    从 {audio_id: audio_path} 中提取“共同父目录”，并判断是否所有音频都在同一目录下。
    返回：(common_folder, is_same_folder)
    """
    if not audio_dict:
        return None, False

    paths = list(audio_dict.values())
    parent_folders = [str(Path(p).parent) for p in paths]
    # 只要所有父目录完全一致，就视为在同一目录
    is_same_folder = all(p == parent_folders[0] for p in parent_folders)
    common_prefix = str(Path(parent_folders[0]).resolve()) if is_same_folder else None
    return common_prefix, is_same_folder


def evaluate(args):
    """仅计算 FAD / FD / KL 等分布类指标（不含 CLAP）"""

    # 1) 读取参考与生成音频映射
    print(args.ref_audio_jsonl)
    ref_aid_to_audios = read_jsonl_to_mapping(
        args.ref_audio_jsonl,
        "audio_id",
        "audio",
    )

    if args.gen_audio_jsonl is not None:
        gen_aid_to_audios = read_jsonl_to_mapping(
            args.gen_audio_jsonl, "audio_id", "audio"
        )
    elif args.gen_audio_dir is not None:
        gen_aid_to_audios = audio_dir_to_mapping(args.gen_audio_dir, args.task)
    else:
        raise ValueError("必须提供 --gen_audio_jsonl 或 --gen_audio_dir 其一。")

    # 2) 对齐 id：两侧取交集（更稳妥）
    ref_keys = set(ref_aid_to_audios.keys())
    gen_keys = set(gen_aid_to_audios.keys())
    common_ids = ref_keys & gen_keys

    dropped_ref = len(ref_keys - common_ids)
    dropped_gen = len(gen_keys - common_ids)

    if dropped_ref or dropped_gen:
        print(f"[Info] 对齐 audio_id：保留 {len(common_ids)} 条；"
              f"丢弃参考侧 {dropped_ref} 条、生成侧 {dropped_gen} 条。")

    ref_aid_to_audios = {k: ref_aid_to_audios[k] for k in common_ids}
    gen_aid_to_audios = {k: gen_aid_to_audios[k] for k in common_ids}

    if not ref_aid_to_audios:
        raise RuntimeError("对齐后没有可评测的样本，请检查 id 命名规则。")

    # 3) 初始化评测器（audioldm_eval）
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "cnn14" if args.task == "tta" else "mert"
    evaluator = EvaluationHelper(
        16000, device=args.device, backbone=backbone
    )

    # 4) 要求两侧音频各自位于同一目录；为“生成目录”建立软链接目录
    gen_folder_path, gen_is_same_folder = get_common_folder_path(gen_aid_to_audios)
    ref_folder_path, ref_is_same_folder = get_common_folder_path(ref_aid_to_audios)

    assert gen_is_same_folder, "Generated audio files must be in the same folder."
    assert ref_is_same_folder, "Reference audio files must be in the same folder."

    gen_folder_path_symlink = create_subset_symlink_folder(
        gen_folder_path,
        set(gen_aid_to_audios.values())  # 交集子集对应的路径集合
    )
    ref_folder_path_symlink = create_subset_symlink_folder(
        ref_folder_path,
        set(ref_aid_to_audios.values())
    )

    # 5) 计算分布类指标（FAD / FD / KL 等）
    eval_result = evaluator.main(
        gen_folder_path_symlink,
        ref_folder_path_symlink,
        recalculate=args.recalculate,
        num_workers=args.num_workers,
    )

    # 6) 写出结果
    results = defaultdict(dict)
    results.update(eval_result)  # eval_result 一般类似 {"FAD": 1.23, "FD": 2.34, "KL": 3.45}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as writer:
        for metric, value in results.items():
            # audioldm_eval 返回的 value 通常是 float
            try:
                print_msg = f"{metric}: {float(value):.3f}"
            except Exception:
                # 兜底（如果返回值不是纯 float）
                print_msg = f"{metric}: {value}"
            print(print_msg)
            print(print_msg, file=writer)

    print(f"[Done] 指标已写入：{out_path}")


def build_argparser():
    parser = argparse.ArgumentParser("Text-to-Audio/Music 分布指标评测（不含 CLAP）")
    parser.add_argument(
        "--ref_audio_jsonl", "-r",
        type=str, required=True,
        help="参考音频 JSONL，包含字段：audio_id, audio"
    )
    parser.add_argument(
        "--gen_audio_dir", "-gd",
        type=str, required=False,
        help="生成音频所在目录（与 --gen_audio_jsonl 二选一）"
    )
    parser.add_argument(
        "--gen_audio_jsonl", "-gj",
        type=str, required=False,
        help="生成音频 JSONL（与 --gen_audio_dir 二选一），包含字段：audio_id, audio"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str, required=True,
        help="结果输出文件路径（文本）"
    )
    parser.add_argument(
        "--task", "-t",
        type=str, default="tta", choices=["tta", "ttm"],
        help="任务类型：tta(text-to-audio) 或 ttm(text_to_music)，影响评测 backbone"
    )
    parser.add_argument(
        "--num_workers", "-c",
        type=int, default=4,
        help="并行 worker 数（传给 audioldm_eval）"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="是否强制重新计算嵌入（避免使用缓存）"
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    evaluate(args)
