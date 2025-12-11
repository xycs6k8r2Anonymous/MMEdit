#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
评估脚本：计算单条及平均 LSD，并将结果写入 JSONL 文件  
用法示例：
python eval_sr.py \
  --ref_audio_jsonl /path/ref.jsonl \
  --gen_audio_jsonl /path/gen.jsonl \
  --out_jsonl     /path/result.jsonl \
  --sr 24000
"""
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import librosa                     
import numpy as np                
from ssr_eval.metrics import AudioMetrics


def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["audio_id"]] = item["audio"]
    return data


def evaluate(args):
    ref_dict = load_jsonl(args.ref_audio_jsonl)
    gen_dict = load_jsonl(args.gen_audio_jsonl)

    evaluator = AudioMetrics(rate=args.sr)

    results, total_lsd, count = [], 0.0, 0

    for audio_id in tqdm(gen_dict.keys(), desc="Evaluating"):
        if audio_id not in ref_dict:
            print(f"[WARN] {audio_id} 不在参考集，跳过")
            continue

        ref_path, gen_path = ref_dict[audio_id], gen_dict[audio_id]

        try:
            
            ref_wav, _ = librosa.load(ref_path, sr=args.sr, mono=True)
            gen_wav, _ = librosa.load(gen_path, sr=args.sr, mono=True)

            min_len = min(ref_wav.shape[0], gen_wav.shape[0])
            ref_wav, gen_wav = ref_wav[:min_len], gen_wav[:min_len]
            

            metrics = evaluator.evaluation(gen_wav, ref_wav, gen_path)
            lsd = metrics["lsd"]

            results.append({"audio_id": audio_id, "lsd": lsd})
            total_lsd += lsd
            count += 1
        except Exception as e:
            print(f"[ERROR] {audio_id} 计算失败: {e}")

    if count == 0:
        print("没有成功评估的样本，程序结束")
        return

    avg_lsd = total_lsd / count
    results.append({"audio_id": "AVERAGE", "lsd": avg_lsd, "num_samples": count})

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n共评估 {count} 条音频")
    print(f"平均 LSD: {avg_lsd:.6f}")
    print(f"详细结果已保存至: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref_audio_jsonl", required=True, help="参考音频 jsonl 路径")
    parser.add_argument("-g", "--gen_audio_jsonl", required=True, help="生成音频 jsonl 路径")
    parser.add_argument("-o", "--out_jsonl", required=True, help="输出结果 jsonl 路径")
    parser.add_argument("--sr", type=int, default=24000, help="采样率 (与训练/推理保持一致)")
    args = parser.parse_args()

    evaluate(args)
