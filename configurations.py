import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_g = './weights/g.pth'
path_d = './weights/d.pth'
data_dir = 'E:/HKUST/2021F/COMP4471/project/data/img/G2.h5'

generated_dir = './test/'

# photo_sketch_dir = './PhotoSketch/'
# photo_sketch_cmd = f'python {photo_sketch_dir}test_pretrained.py ^ \
#     --name pretrained ^ \
#     --dataset_mode test_dir ^   \
#     --dataroot {generated_dir} ^  \
#     --results_dir ./sketch/ ^  \
#     --checkpoints_dir  {photo_sketch_dir}checkpoints/^  \
#     --model pix2pix ^   \
#     --which_direction AtoB ^    \
#     --norm batch ^  \
#     --input_nc 3 ^  \
#     --output_nc 1 ^ \
#     --which_model_netG resnet_9blocks ^ \
#     --no_dropout ^'


hed_weight_dir = './weights/hed-bsds500'