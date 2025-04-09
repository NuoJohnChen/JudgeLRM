set -x

nvidia-smi -i 0,1,2,3 -c EXCLUSIVE_PROCESS

fuser -k /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3

export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --num-gpus=4

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.reward_ablation=reward_w_length \
    data.train_files=data/judgelrm/instruct/jppl/train.parquet \
    data.val_files=data/judgelrm/instruct/jppl/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl_grpo_example_7b_323' \
    trainer.experiment_name='qwen2_7b_function_rm_323' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/shared/hdd/nuochen/models/GRPO_logic_KK_321/base_reward_7b_323 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 $@
