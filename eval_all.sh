echo `date` >> eval.log

cd /home/smelly/projects/attend2u

CUDA_VISIBLE_DEVICES=3 python -m eval --num_gpus 1 --batch_size 500 >> /home/smelly/projects/attend2u/eval.log 2>&1

# bash /home/smelly/projects/attend2u/eval_nocnn.sh 2 >> /home/smelly/projects/attend2u/eval.log

# bash /home/smelly/projects/attend2u/eval_noword.sh >> /home/smelly/projects/attend2u/eval_noword.log 2>&1


