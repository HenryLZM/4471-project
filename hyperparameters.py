# DATA
batch_size = 32

# MODEL
nz = 100
ngf = 64
ndf = 64

# TRAIN
#   Generator
g_lr = 5e-4
g_beta1, g_beta2 = 0.5, 0.999
#   Fashion Discreminator
d_lr = 4e-6
d_beta1, d_beta2 = 0.5, 0.999
#   Sketch Disciminator
sg_lr = 1e-4
sg_beta1, sg_beta2 = 0.5, 0.999
sd_lr = 1e-6
sd_beta1, sd_beta2 = 0.5, 0.999