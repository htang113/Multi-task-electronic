export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export OMP_NUM_THREADS=1

torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_inp.py