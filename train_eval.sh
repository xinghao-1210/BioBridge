# conda activate geneformer

# export WANDB_PROJECT=biokgbind

n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

kg_name='primekg_bulk_excelraopiintegration-lfc_weighted-11112024_ppigort'
data_dir="./data/BindData/$kg_name" # the data directory
split_dir="./data/BindData/$kg_name/train_test_split" # the train/test split directory

hidden_dim=1152 # the hidden dimension of the transformation model, 768
n_layer=12 # the number of transformer layers, 6
batch_size=2048 # the training batch size, 1024
learning_rate=1e-4 # the learning ratesss, 1.6e-3
n_epoch=500 # the number of training epochs, 10
weight_decay=1e-4 # the weight decay, 1e-4
evaluation_strategy='epoch' # evaluation_strategy, steps
eval_steps=200 # the number of steps to evaluate the model, 1000
logging_steps=100 # logging_steps, 1000
dataloader_num_workers=4 # the number of workers for data loading, 4
use_wandb=False # whether to use wandb, False

target_relation=None # the target relation to predict
target_node_type_index=None # the index of the target node type
frequent_threshold=100 # the threshold of the frequent node

save_dir="./checkpoints/${kg_name}/model_${n_layer}layer_${n_epoch}epoch" # the directory to save the model
logbind="log_bind_${kg_name}.txt"


# train with single-gpu
export CUDA_VISIBLE_DEVICES=0
python train_bind.py \
  --batch_size $batch_size \
  --n_epoch $n_epoch \
  --hidden_dim $hidden_dim \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --evaluation_strategy $evaluation_strategy \
  --eval_steps $eval_steps \
  --logging_steps $logging_steps \
  --n_layer $n_layer \
  --data_dir $data_dir \
  --save_dir $save_dir \
  --split_dir $split_dir \
  --dataloader_num_workers $dataloader_num_workers \
  --frequent_threshold $frequent_threshold \
  --use_wandb $use_wandb \
  > $logbind 2>&1 &

# # train with multi-gpu
# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nproc_per_node $n_gpu #--master_port=25678 \
#     train_bind.py \
#     --batch_size $batch_size \
#     --n_epoch $n_epoch \
#     --hidden_dim $hidden_dim \
#     --learning_rate $learning_rate \
#     --weight_decay $weight_decay \
#     --evaluation_strategy $evaluation_strategy \
#     --eval_steps $eval_steps \
#     --logging_steps $logging_steps \
#     --n_layer $n_layer \
#     --data_dir $data_dir \
#     --save_dir $save_dir \
#     --split_dir $split_dir \
#     --dataloader_num_workers $dataloader_num_workers \
#     --frequent_threshold $frequent_threshold \
#     --use_wandb $use_wandb \
#     > $logbind 2>&1 &

# Get the PID of the last background process
TORCHRUN_PID=$!
# Wait for the torchrun process to complete
wait $TORCHRUN_PID

# Run evaluation after training completes
python eval_bind.py \
    --data_dir $data_dir \
    --split_dir $split_dir \
    --hidden_dim $hidden_dim \
    --n_layer $n_layer \
    --checkpoint_dir $save_dir \
    --dataloader_num_workers $dataloader_num_workers \
    --target_relation $target_relation \
    --target_node_type_index $target_node_type_index \
    --frequent_threshold $frequent_threshold \
    > "eval_${logbind}" 2>&1