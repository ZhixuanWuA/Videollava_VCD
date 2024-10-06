   
    ####均匀采样###
    # 获取视频帧的总数
    # total_frames = video_tensor.size(0)
    # num_samples = total_frames // 4
    # # 生成均匀间隔的采样索引
    # key_frame_indices = torch.linspace(0, total_frames - 1, num_samples).long()
    ####均匀采样###
    # for idx in key_frame_indices:
    #     print(f'idx:{idx}')
    #     noisy_video[idx] = q_x(noisy_video[idx], noise_step)

    # return noisy_video
    
    ### koala-video-llm
    # key_frame_indices = get_key_frame_indices(video_tensor)
    # for idx in key_frame_indices:
    #     print(f'idx:{idx}')
    #     noisy_video[idx] = q_x(noisy_video[idx], noise_delta)
    
    # video_tensor_cd = q_x(noisy_video,noise_delta) 

import torch
import cv2
import numpy as np

def add_diffusion_noise(video_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    betas = torch.linspace(0, 0.01, num_steps)  # Adjusted beta range
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0) * 0.1  # Scale down noise
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_video = video_tensor.clone()

    # 基于内容的关键帧提取
    key_frame_indices = get_key_frame_indices(video_tensor, threshold=400)
    for idx in key_frame_indices:
        print(f'idx:{idx}')
        noisy_video[0, :, idx] = q_x(noisy_video[0, :, idx], noise_step)
    return noisy_video
    
    ### koala-video-llm
    # key_frame_indices = get_key_frame_indices(video_tensor)
    # for idx in key_frame_indices:
    #     print(f'idx:{idx}')
    #     noisy_video[idx] = q_x(noisy_video[idx], noise_delta)
    
    # video_tensor_cd = q_x(noisy_video,noise_delta) 

    # return video_tensor_cd

##calucate absolute difference（motion feature）
def get_key_frame_indices(video_tensor, threshold=10):
    key_frame_indices = []
    # print("Before normalization - Min:", video_tensor.min().item(), "Max:", video_tensor.max().item())
    video_tensor = video_tensor.float() / 255.0  # 将值范围调整到 [0, 1]
    # print("After normalization - Min:", video_tensor.min().item(), "Max:", video_tensor.max().item())
    total_frames = video_tensor.size(2)

    for i in range(1, total_frames):
        prev_frame = video_tensor[0, :, i - 1]  # 获取前一帧
        curr_frame = video_tensor[0, :, i]      # 获取当前帧

        # 计算帧之间的绝对差异
        diff = torch.abs(prev_frame - curr_frame).sum()
        # print(f'Frame {i-1} to {i} difference: {diff.item()}')  # 输出差异值

        # 如果差异超过阈值，记录当前帧为关键帧
        if diff.item() > threshold:
            # print(f'Difference for frames {i-1} to {i}: {diff.item()} exceeds threshold {threshold}')
            key_frame_indices.append(i)
    # print(f'key:{key_frame_indices}')
    return key_frame_indices

## koala-video-llm
# def get_key_frame_indices(video_tensor):
#     num_frames = video_tensor.size(0)  
#     global_clip_indices = np.linspace(0, num_frames-1, num=min(16, num_frames))
#     short_window_indices = np.linspace(0, num_frames-1, num=min(16 * 4, num_frames))

#     return global_clip_indices, short_window_indices
