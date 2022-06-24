import architectures.deit
def select(arch, opt):
    if 'deit_small_patch16_224' in arch:
        return deit.Network(opt)
#'vit_deit_small_patch16_224'