export CUDA_VISIBLE_DEVICES=0
CKPT_NAME="Video-LLaVA-7B"
model_path="/home/zhangshaoxing/cv/code/Videollava/checkpoints/Video-LLaVA-7B"
cache_dir="./cache_dir"
video_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/videos"
gt_file_question="/home/zhangshaoxing/cv/datasets/MSVD_QA/test_q.json"
gt_file_answers="/home/zhangshaoxing/cv/datasets/MSVD_QA/test_a.json"
output_dir="/home/zhangshaoxing/cv/code/Videollava/output/MSVD_QA_VCD_test/image-video-sam2mask/tem05/${CKPT_NAME}"
video_mask_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/sam2_addmaskblack_withoutlabel_result"
image_mask_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/framesall/sam2_image"
image_dir="/home/zhangshaoxing/cv/datasets/MSVD_QA/framesall/image_dir"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_msvd_imageandvideo.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --image_dir ${image_dir} \
      --video_mask_dir ${video_mask_dir} \
      --image_mask_dir ${image_mask_dir} \
      --num_chunks $CHUNKS \
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