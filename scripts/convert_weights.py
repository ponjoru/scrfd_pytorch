import torch
from lib.model import init_scrfd_34g_model, init_scrfd_10g_model, init_scrfd_2_5g_model, init_scrfd_500m_model


def convert(model, vanilla_sd, save_path, use_scale):
    # -------- BACKBONE
    from collections import OrderedDict
    b_sd = OrderedDict()
    for k, v in vanilla_sd.items():
        if k.startswith('backbone'):
            b_sd[k[9:]] = v
    model.backbone.load_state_dict(b_sd)

    # -------- NECK
    from collections import OrderedDict
    n_sd = OrderedDict()
    for k, v in vanilla_sd.items():
        if k.startswith('neck'):
            n_sd[k[5:].replace('.conv.', '.')] = v
    model.neck.load_state_dict(n_sd)

    # -------- HEAD
    # (8, 8)
    head_0 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head')}
    head_0_8x8 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head') and '(8, 8)' in k}
    head_1_8x8 = model.bbox_head.stride_heads[0].state_dict()
    if use_scale:
        head_1_8x8.pop('scale.scale')

    keys_0 = head_0_8x8.keys()
    keys_1 = head_1_8x8.keys()

    sd_1 = {}
    for k0, k1 in zip(keys_0, keys_1):
        sd_1[k1] = head_0_8x8[k0]
    if use_scale:
        sd_1['scale.scale'] = head_0['scales.0.scale']
    model.bbox_head.stride_heads[0].load_state_dict(sd_1)

    # (16, 16)
    head_0 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head')}
    head_0_16x16 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head') and '(16, 16)' in k}
    head_1_16x16 = model.bbox_head.stride_heads[0].state_dict()
    if use_scale:
        head_1_16x16.pop('scale.scale')

    keys_0 = head_0_16x16.keys()
    keys_1 = head_1_16x16.keys()

    sd_1 = {}
    for k0, k1 in zip(keys_0, keys_1):
        sd_1[k1] = head_0_16x16[k0]
    if use_scale:
        sd_1['scale.scale'] = head_0['scales.1.scale']
    model.bbox_head.stride_heads[1].load_state_dict(sd_1)

    # (32, 32)
    head_0 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head')}
    head_0_32x32 = {k[10:]: v for k, v in vanilla_sd.items() if k.startswith('bbox_head') and '(32, 32)' in k}
    head_1_32x32 = model.bbox_head.stride_heads[0].state_dict()
    if use_scale:
        head_1_32x32.pop('scale.scale')

    keys_0 = head_0_32x32.keys()
    keys_1 = head_1_32x32.keys()

    sd_1 = {}
    for k0, k1 in zip(keys_0, keys_1):
        sd_1[k1] = head_0_32x32[k0]
    if use_scale:
        sd_1['scale.scale'] = head_0['scales.2.scale']
    model.bbox_head.stride_heads[2].load_state_dict(sd_1)

    new_sd = {'model': model.state_dict()}
    torch.save(new_sd, save_path)
    print(f'Successfully converted the model and saved weights to: {save_path}')


if __name__ == '__main__':
    # scrfd_500m
    load_from = '/app/weights/SCRFD_500M_KPS.pth'
    save_path = '/app/weights/scrfd_500m_kps.pth'
    model = init_scrfd_500m_model(load_from=None, use_kps=True)
    vanilla_sd = torch.load(load_from)['state_dict']
    convert(model, vanilla_sd, save_path, use_scale=False)

    # scrfd_2.5g
    load_from = '/app/weights/SCRFD_2.5G_KPS.pth'
    save_path = '/app/weights/scrfd_2.5g_kps.pth'
    model = init_scrfd_2_5g_model(load_from=None, use_kps=True)
    vanilla_sd = torch.load(load_from)['state_dict']
    convert(model, vanilla_sd, save_path, use_scale=True)

    # scrfd_10g
    load_from = '/app/weights/SCRFD_10G_KPS.pth'
    save_path = '/app/weights/scrfd_10G_kps.pth'
    model = init_scrfd_10g_model(load_from=None, use_kps=True)
    vanilla_sd = torch.load(load_from)['state_dict']
    convert(model, vanilla_sd, save_path, use_scale=True)

    # scrfd_34g
    """ SCRFD34G is not yet supported since it has a shared head between strides"""
    # load_from = '/app/weights/SCRFD_34G.pth'
    # save_path = '/app/weights/scrfd_34g.pth'
    # model = init_scrfd_34g_model(load_from=None, use_kps=False)
    # vanilla_sd = torch.load(load_from)['state_dict']
    # convert(model, vanilla_sd, save_path, use_scale=True)
