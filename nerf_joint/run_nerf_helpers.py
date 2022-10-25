import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from nerf.load_llff import normalize
from torch.nn.functional import normalize as nn_normalize



# Misc
img2mse = lambda x, y : torch.mean((y-x) ** 2)
l1_loss = lambda x, y : torch.mean(torch.abs(y-x))
#clip_loss = lambda x, y : torch.dot(y, x) / (torch.norm(y) * torch.norm(x))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def dot(x_normalized, y_normalized):
    dot_product =  torch.sum(x_normalized * y_normalized, dim = -1)
    return dot_product

def clip_loss(x_normalized, y_normalized):
    #x_normalized = nn_normalize(torch.tensor([[1.0,1.0],[2.0,2.0]]), p = 2, dim = -1)
    #y_normalized = nn_normalize(torch.tensor([[1.0,1.0],[-2.0,2.0]]), p =2, dim = -1)
    all_losses = 1.0 - dot(x_normalized, y_normalized)
    loss = torch.mean(all_losses)
    return loss
    

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] #d = 3
        out_dim = 0
        if self.kwargs['include_input']: #include_input = True
            embed_fns.append(lambda x : x) #embed_fns = [x=f(x)]
            out_dim += d #out_dim = 3
        max_freq = self.kwargs['max_freq_log2'] #max_freq = 9
        N_freqs = self.kwargs['num_freqs'] #N_freqs = 10
        if self.kwargs['log_sampling']: #log_sampling = True
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d       
        self.embed_fns = embed_fns # len 为 1 + 20 = 21
        self.out_dim = out_dim # 3 + 3 * 20 = 63
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # print("doing get_embedder: multires, i,", multires, i)
    # print("-----------------------------------------")
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x) #torch.cat([fn(x) for fn in embed_fns], -1)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, with_saliency=False, with_CLIP=False, clip_dim=768):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch #63
        self.input_ch_views = input_ch_views #0
        self.skips = skips
        self.use_viewdirs = use_viewdirs #False
        self.with_saliency = with_saliency
        self.with_CLIP = with_CLIP
        self.clip_dim = clip_dim
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]) # input in 0th and 4th layer
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)]) # 256 -> 128
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        # print("using_viewdirs:",use_viewdirs )
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch) #256 -> 4
        if with_saliency:
            self.featureS_linear = nn.Linear(W, W)
            self.alphaS_linear = nn.Linear(W, 1)
            self.saliency_linear = nn.Linear(W//2, 1)
        if with_CLIP:
            #RGB branch
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W//2, 3)
            #CLIP branch
            self.alphaCLIP_linear = nn.Linear(W, 1)
            self.featureCLIP_linear = nn.Linear(W, W)
            self.CLIP_linear = nn.Linear(W//2, self.clip_dim)

            self.alpha_linear.weight.requires_grad=True
            self.feature_linear.weight.requires_grad=True
            self.rgb_linear.weight.requires_grad=True
            #CLIP branch
            self.alphaCLIP_linear.weight.requires_grad=False
            self.featureCLIP_linear.weight.requires_grad=False
            self.CLIP_linear.weight.requires_grad=False
            

    def forward(self, x):
        #Main body
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h_original = input_pts
        for i, l in enumerate(self.pts_linears):
            h_original = self.pts_linears[i](h_original)
            h_original = F.relu(h_original)
            if i in self.skips:
                h_original = torch.cat([input_pts, h_original], -1)
        #RGB branch
        alpha = self.alpha_linear(h_original)
        feature = self.feature_linear(h_original)
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        #CLIP branch
        alphaCLIP = self.alphaCLIP_linear(h_original)
        featureCLIP = self.featureCLIP_linear(h_original)
        hs = torch.cat([featureCLIP, input_views], -1)
        for i, l in enumerate(self.views_linears):
            hs = self.views_linears[i](hs)
            hs = F.relu(hs)
        CLIP_val = self.CLIP_linear(hs)
        #Outputs
        outputs_rgb = torch.cat([rgb, alpha], -1)
        # outputs_clips = torch.cat([CLIP_val, alphaCLIP * alpha], -1) #torch.Size([65536, 769])
        outputs_clips = torch.cat([CLIP_val, alphaCLIP * alpha], -1)#for render test
        return outputs_rgb, outputs_clips

    def switch_to_clip(self):
        if self.with_CLIP:
            #RGB branch
            self.alpha_linear.weight.requires_grad=False
            self.feature_linear.weight.requires_grad=False
            self.rgb_linear.weight.requires_grad=False
            #CLIP branch
            self.alphaCLIP_linear = nn.Linear(self.W, 1)
            self.featureCLIP_linear = nn.Linear(self.W, self.W)
            self.CLIP_linear = nn.Linear(self.W//2, self.clip_dim)

            self.alphaCLIP_linear.weight.requires_grad=True
            self.featureCLIP_linear.weight.requires_grad=True
            self.CLIP_linear.weight.requires_grad=True


    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    K = np.array(K)
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
