CKPT_NAME="Video-LLaVA-7B"
model_path="/home/zhangshaoxing/wzx/code/Videollava/checkpoints/Video-LLaVA-7B"
# chat_model_path="/home/wuzhixuan/code/meta-llama/Llama-2-7b-chat-hf"
cache_dir="./cache_dir"
# GPT_Zero_Shot_QA="/home/wuzhixuan/code/Video-LLaVA/videollava/eval/GPT_Zero_Shot_QA"
# video_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/videos"
video_dir="/home/zhangshaoxing/wzx/datasets/MSVD_QA/videos"
gt_file_question="/home/zhangshaoxing/wzx/datasets/MSVD_QA/test_q.json"
gt_file_answers="/home/zhangshaoxing/wzx/datasets/MSVD_QA/test_a.json"
output_dir="/home/zhangshaoxing/wzx/code/Videollava/output/MSVD_QA_VCD_test/sam2-mask/tem06/${CKPT_NAME}"
video_mask_dir="/home/zhangshaoxing/wzx/datasets/MSVD_QA/sam2_addmaskblack_withoutlabel_result"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_qa_mask.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --video_mask_dir ${video_mask_dir} \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done