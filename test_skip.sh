#! /bin/bash
for ((use_num=132;use_num<=320;use_num+=4));do
    CUDA_VISIBLE_DEVICES=1 python test.py --dataset /mnt/data1/jingwengu/kodak --output_path /mnt/data3/jingwengu/ELIC_light/ELIC_sort/lambda5_s/test_skip --cuda -p /mnt/data3/jingwengu/ELIC_light/ELIC_sort/lambda5_s/try_500_50.pth.tar --use_num $use_num
done