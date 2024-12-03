import math
import os
import argparse
import json

import torch
import transformers
from transformers import set_seed
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize
import shutil  # 导入 shutil 模块
from PIL import Image
from videoprocess.VCD_sample import evolve_vcd_sampling
evolve_vcd_sampling()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)  #模型地址
    parser.add_argument('--cache_dir', help='', required=True)   #缓存文件
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)  ##测试视频地址
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)  ##gt问题地址
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)   ##gt回答地址
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)  ##输出文件地址
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)  ##输出名称
    parser.add_argument("--num_chunks", type=int, default=1)  ##将数据分成多少块（chunks）
    parser.add_argument("--chunk_idx", type=int, default=0)  #指定当前处理的是数据的哪一块
    parser.add_argument("--device", type=str, required=False, default='cuda:0')  #指定模型运行的设备
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)  #指定模型的基础路径或名称
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)  #指定模型处理的最大长度

    parser.add_argument("--top_p", type=float, default=0.5)  #指定在生成文本时使用的概率分布的累积概率阈值
    parser.add_argument("--top_k", type=int, default=100)  #指定在生成文本时考虑的词汇表中前k个最可能的词

    parser.add_argument("--noise_step", type=int, default=500)  #添加噪声步骤中的步数
    parser.add_argument("--use_mask", action='store_true', default=True)  #是否使用mask
    parser.add_argument("--cd_alpha", type=float, default=1)  #vcd中alpha
    parser.add_argument("--cd_beta", type=float, default=0.5)  #vcd中beta
    parser.add_argument("--seed", type=int, default=42)  #随机数生成器的种子
    
    parser.add_argument('--image_dir', help='Directory containing image files.', required=True)
    parser.add_argument('--video_mask_dir', help='Directory containing mask video files.', required=True)  ##mask测试视频地址
    parser.add_argument('--image_mask_dir', help='Directory containing mask image files.', required=True)  ##mask测试image地址

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, image_processor, image, video_mask, image_mask, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs+ ' Please answer concisely based on both video and image content.'
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs+ ' Please answer concisely based on both video and image content.'

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    image = Image.open(image).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    if args.use_mask:
            print(f'begining mask+cd function...')
            video_tensor_mask = video_processor.preprocess(video_mask, return_tensors='pt')['pixel_values'][0].half().to(args.device)
            image_mask = Image.open(image_mask).convert('RGB')
            image_tensor_mask = image_processor.preprocess(image_mask, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    else:
            video_tensor_mask = None   
            image_tensor_mask = None

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            images_mask=[video_tensor_mask],
            image=image_tensor,
            image_mask=image_tensor_mask,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            temperature=0.5,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            image_temp_path = os.path.join(args.image_dir, f"{video_name}.jpg")
            # 在video_mask_dir目录下找到对应的文件夹，使用id去匹配
            video_mask_folder = os.path.join(args.video_mask_dir, str(id))
            mask_temp_path = os.path.join(video_mask_folder, f"{id}_output.mp4")
            # 在image_mask_dir目录下找到对应的文件夹，使用id去匹配
            image_mask_temp_path = os.path.join(args.image_mask_dir, f"{id}.jpg")
            # print(f'mask_temp_path:{mask_temp_path},image_mask_temp_path:{image_mask_temp_path}')
            if os.path.exists(temp_path):
                video_path = temp_path
                if os.path.exists(mask_temp_path):
                    video_mask_path = mask_temp_path
                else:
                    video_mask_path = temp_path
                if os.path.exists(mask_temp_path):
                    image_mask_path = image_mask_temp_path
                else:
                    image_mask_path = image_temp_path
                print(f'video_path:{video_path}, video_mask_path:{video_mask_path},image_path:{image_temp_path},image_mask_path:{image_mask_path}')
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor['video'], tokenizer, video_path, processor['image'], image_temp_path, video_mask_path, image_mask_path, question, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)
    
    # 将当前脚本文件复制到 output_dir 目录
    current_file_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    shutil.copy(current_file_path, args.output_dir)  # 复制脚本文件到输出目录


if __name__ == "__main__":
    args = parse_args()
    # set_seed(args.seed)
    run_inference(args)