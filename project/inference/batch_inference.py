import json
import os
import random
import argparse
import torch
from tqdm import tqdm
from PIL import Image

# 从你已经跑通的 run_baseline.py 中导入配置和模型加载函数
from run_baseline import load_all_models, EMOTION_PROMPTS, NEGATIVE_PROMPT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json", help="验证集 JSON 路径")
    parser.add_argument("--output-dir", default="results/baseline/batch", help="输出文件夹")
    parser.add_argument("--manifest-name", default="manifest.json", help="保存评估元数据的 JSON 文件名")
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter 强度")
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="SD 基础模型路径")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter", help="IP-Adapter 仓库路径")
    # 设置 n=0 表示跑全部数据
    parser.add_argument("--n", type=int, default=500, help="测试数量，设为 0 表示跑完整个 JSON")
    return parser.parse_args()

def batch_generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, args.manifest_name)

    print(f"📖 正在读取数据: {args.pairs_json}")
    with open(args.pairs_json) as f:
        pairs = json.load(f)

    # 如果设置了 n 且大于 0，则只切片前 n 个；否则跑全部
    if args.n > 0:
        pairs = pairs[:args.n]

    # 直接复用 run_baseline 里的完美加载逻辑
    pipe, feature_extractor, device = load_all_models(args)
    
    available_emotions = list(EMOTION_PROMPTS.keys())
    manifest_records = []

    print(f"\n🚀 开始批量推理，共需处理 {len(pairs)} 张图...\n")

    for i, pair in enumerate(tqdm(pairs)):
        ref_path = pair["reference_path"]
        
        # 核心逻辑：如果 JSON 里是 unknown，我们随机指派一个有效情绪
        target_emotion = pair.get("target_emotion", "unknown")
        if target_emotion not in available_emotions:
            target_emotion = random.choice(available_emotions)
        
        prompt = EMOTION_PROMPTS[target_emotion]

        if not os.path.exists(ref_path):
            tqdm.write(f"⚠️ 找不到参考图: {ref_path}")
            continue

        try:
            raw_image = Image.open(ref_path).convert("RGB")
            
            # 使用不断变化的 seed 以增加生成图像的多样性
            generator = torch.Generator(device=device).manual_seed(42 + i)

            # 调用跑通的管道
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[raw_image],  # 重点：列表包装
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            
            # 命名规则：序号_设定图ID_目标情绪.jpg
            sheet_id = pair.get("sheet_id", "none")
            save_name = f"{i:04d}_{sheet_id}_{target_emotion}.jpg"
            output_path = os.path.join(args.output_dir, save_name)
            image.save(output_path)

            manifest_records.append({
                "index": i,
                "sheet_id": sheet_id,
                "reference_path": ref_path,
                "target_path": pair.get("target_path"),
                "generated_path": output_path,
                "requested_label": target_emotion,
                "label_type": "expression",
                "seed": 42 + i,
                "ip_adapter_scale": args.scale,
            })
            
        except Exception as e:
            tqdm.write(f"❌ 生成失败 {ref_path}: {e}")

    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    print(f"\n✨ 批量跑图完成！全部保存在: {args.output_dir}")
    print(f"🧾 评估清单已保存: {manifest_path}")

if __name__ == "__main__":
    args = parse_args()
    batch_generate(args)
