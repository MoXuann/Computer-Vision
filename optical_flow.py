import torch
import torch.nn.functional as F
import numpy as np

def json2opticalflow(json_file, arr):
    """ 
    convert the motion vector json into optical flow (h,w,2)
    
    arr have image's height and width
    """
    for dic in json_file:
        width = int(dic["width"] / 2)
        height = int(dic["height"] / 2)
        x = dic['src_x']
        y = dic['src_y']
        arr[y-height:y+height,x-width:x+width,0] = dic["dx"]
        arr[y-height:y+height,x-width:x+width,1] = dic["dy"]
    return arr

# Visualize optical flow
def make_colorwheel():
    """ 
    make colorwheel for visualizing optical flow
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    #return colorwheel #original code
    return np.flip(colorwheel,1) #modified 

def flow_compute_color(flow_u, flow_v):
    """ 
    Compute the color for the optical flow
    Parameters
    ----------
    flow_u: numpy.ndarray, (H, W, 1), float
    flow_v: numpy.ndarray, (H, W, 1), float
        Optical flow.
    Returns
    -------
    dst: numpy.ndarray
        RGB image.
    """
    H, W = flow_u.shape[:2]
    flow_image = np.zeros((H, W, 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(flow_u) + np.square(flow_v))
    a = np.arctan2(- flow_v, - flow_u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        ch_idx = i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow2rgb(flow_uv):
    """Visualize optical flow.
    Parameters
    ----------
    flow_uv: numpy.ndarray, (H, W, 2), float
        Optical flow.
    Returns
    -------
    dst: numpy.ndarray
        RGB image.
    """
    assert flow_uv.ndim == 3, 'flow must be 3 dimensional'
    assert flow_uv.shape[2] == 2, 'flow must have shape (H, W, 2)'
    assert np.issubdtype(flow_uv.dtype, np.floating), \
        'float must be float type'

    flow_u = flow_uv[:, :, 0]
    flow_v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(flow_u) + np.square(flow_v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    flow_u = flow_u / (rad_max + epsilon)
    flow_v = flow_v / (rad_max + epsilon)

    return flow_compute_color(flow_u, flow_v)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid_y, grid_x = grid_y.type_as(x), grid_x.type_as(x)
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

def np2tensor(np_arr):
    return torch.from_numpy(np_arr)

def tensor2np(tensor):
    return tensor.numpy()

