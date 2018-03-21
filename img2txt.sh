# from image to text
# modified by Jay

image_path='data/Lake_mapourika_NZ.jpg'
# image_path='data/cake.jpg'
#$ image_path='data/food.jpg'

rm data/caption_dataset/test3.txt

CUDA_VISIBLE_DEVICES=0 python xImg.py $image_path

CUDA_VISIBLE_DEVICES=0 python -m eval \
            --num_gpus 1 \
            --batch_size 1 \
            --input_path test3.txt \
            --img_data_dir data
            # --img_data_dir data/resnet_pool5_features

