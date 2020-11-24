# 1.test the TSN and TSM on Kinetics using 8-frame, you should get top-1 accuracy around:
# TSN: 68.8%
# TSM: 71.2%

# test TSN
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# test TSM
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# Change to --test_crops=10 for 10-crop evaluation. With the above scripts, you should get around 68.8% and 71.2% results respectively.

# 2.To get the Kinetics performance of our dense sampling model under Non-local protocol, run:

# test TSN using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test NL TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res


# 3.For the efficient (center crop and 1 clip) and accurate setting (full resolution and 2 clip) on Something-Something, you can try something like this:

# efficient setting: center crop and 1 clip
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 24 --test_crops=1

# accurate setting: full resolution and 2 clips (--twice sample)
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 24 --test_crops=3  --twice_sample