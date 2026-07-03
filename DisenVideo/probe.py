import os
import torch
from diffusers import TextToVideoSDPipeline, DPMSolverMultistepScheduler
from MotionDirector_train import export_to_video

# === [新增] 强制导入我们修改过的本地 U-Net ===
from models.unet_3d_condition import UNet3DConditionModel


def get_alt_prompt_embeds(pipe, alt_prompt, negative_prompt, batch_size):
    # ... (这部分代码保持不变) ...
    text_inputs = pipe.tokenizer(
        [alt_prompt] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]

    uncond_inputs = pipe.tokenizer(
        [negative_prompt] * batch_size if negative_prompt else [""] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeds = pipe.text_encoder(uncond_inputs.input_ids.to(pipe.device))[0]

    return torch.cat([uncond_embeds, prompt_embeds])


@torch.inference_mode()
def probe_layers():
    model_id = "./models/zeroscope_v2_576w"  # 确保路径是你本地的
    device = "cuda"
    output_dir = "./outputs/probe_experiment"
    os.makedirs(output_dir, exist_ok=True)

    # === [修改] 先加载本地定制的 U-Net，再喂给 Pipeline ===
    print("正在加载本地定制的 U-Net (包含拦截逻辑)...")
    local_unet = UNet3DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)

    print(f"正在加载基础模型 {model_id}...")
    # 把 local_unet 强制塞给 pipe
    pipe = TextToVideoSDPipeline.from_pretrained(model_id, unet=local_unet, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)  # 将整个 pipe 移到 GPU
    # =======================================================

    base_prompt = "A dog is sitting on the grass"
    alt_prompt = "A dog is jumping on the grass"
    negative_prompt = "low quality, text, error, extra digits, fewer digits, cropped"

    batch_size = 1
    num_frames = 24

    print("正在提前编码备用提示词特征...")
    alt_prompt_embeds = get_alt_prompt_embeds(pipe, alt_prompt, negative_prompt, batch_size)

    test_layers = [
        "down_blocks.0", "down_blocks.1", "down_blocks.2",
        "mid_block",
        "up_blocks.1", "up_blocks.2", "up_blocks.3"
    ]

    for layer in test_layers:
        print(f"\n========================================")
        print(f"🚀 正在测试注入层: [{layer}]")
        print(f"   基础提示词 -> 所有层: '{base_prompt}'")
        print(f"   备用提示词 -> 仅进入: '{alt_prompt}'")

        generator = torch.Generator(device="cpu").manual_seed(42)

        cross_attention_kwargs = {
            "alt_prompt_embeds": alt_prompt_embeds,
            "inject_blocks": [layer]
        }

        video_frames = pipe(
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            height=320,
            width=576,
            num_frames=num_frames,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs
        ).frames

        video_path = os.path.join(output_dir, f"Probe_Action_{layer}.mp4")
        export_to_video(video_frames, video_path, 8)
        print(f"✅ 视频已保存: {video_path}")


if __name__ == "__main__":
    probe_layers()