
CUDA_NUMBER=""

export PYTHONPATH=.:$PYTHONPATH

for model in 'efficientnet'
#for model in 'densenet169'
do
    for n_round in 0
    do
    seed_number=$((42-2*n_round))
    CUDA_VISIBLE_DEVICES=${CUDA_NUMBER} python test_outside.py --e=1 --b=1 --task_name='Retinal_quality' --model=${model} --round=${n_round} --train_on_dataset='EyePACS_quality' \
    --test_on_dataset='customised_data' --test_csv_dir='../Results/M0/images/' --n_class=3 --seed_num=${seed_number}

    
    done

done



M1_args = {
    "epochs":1,
    "batchsize":1,
    "load": False,
    "test_dir": ???, # Results/M0/images
    "n_class": False,
    "dataset": "customised_data",
    "task": "Retinal_quality",
    "round": 0,
    "model": "efficientnet",
    "seed": 42,
    "local_rank": 0 
}