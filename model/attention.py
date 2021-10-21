import torch
from torch import nn


class Conv3dAttentionOutput(nn.Conv3d):

    def __init__(self, nf_in, nf_out):
        super().__init__(nf_in, nf_out, kernel_size=1, stride=1, padding=0)

    def reset_parameters(self) -> None:
        nn.init.dirac_(self.weight)
        with torch.no_grad():
            self.weight[:] = self.weight[:] + torch.randn_like(self.weight[:]) * 0.01
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv3dAttentionFeature(nn.Conv3d):

    def __init__(self, nf_in, nf_out):
        super().__init__(nf_in, nf_out, kernel_size=1, stride=1, padding=0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class AttentionFeatureEncoder(nn.Module):

    def __init__(self, n_in, n_out, e):
        super().__init__()
        self.n_in = n_in * (e ** 3)
        self.n_out = n_out
        print("Attention Feature: ", self.n_in, "-->", self.n_out)
        self.encoder = torch.nn.Sequential(nn.Linear(self.n_in, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, self.n_out),)

    def forward(self, x):
        b = x.shape[0]
        return self.encoder(x.reshape((b, self.n_in)))


class AttentionBlock(nn.Module):

    def __init__(self, num_output_channels, patch_extent, K, normalize, use_switching, retrieval_mode, no_output_mapping, blend):
        super().__init__()
        self.cf_op = num_output_channels
        self.cf_feat = 32
        self.theta = AttentionFeatureEncoder(num_output_channels, self.cf_feat, patch_extent)
        self.phi = AttentionFeatureEncoder(num_output_channels, self.cf_feat, patch_extent)
        self.g = Conv3dAttentionOutput(num_output_channels, self.cf_op) if not no_output_mapping else nn.Identity()
        self.o = Conv3dAttentionOutput(self.cf_op, num_output_channels) if not no_output_mapping else nn.Identity()
        self.max = nn.MaxPool1d(kernel_size=K)
        self.init_scale = 35
        self.init_shift = -27
        self.sig_scale = nn.Parameter(torch.ones(1) * self.init_scale)
        self.sig_shift = nn.Parameter(torch.ones(1) * self.init_shift)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.retrieval_mode = retrieval_mode
        self.blend_mode = blend
        self.use_switching = use_switching
        self.normalize = normalize

    def get_features(self, x, p):
        # x: BxC0xExExE, p: BxC0xExExE
        b = x.shape[0]
        x_feat = self.theta(x)
        p_feat = self.phi(p)
        x_feat_flat = x_feat.reshape((b, -1))
        p_feat_flat = p_feat.reshape((b, -1))
        if self.normalize:
            x_feat_flat = nn.functional.normalize(x_feat_flat, dim=1)
            p_feat_flat = nn.functional.normalize(p_feat_flat, dim=1)
        return x_feat_flat, p_feat_flat

    def forward(self, x, p):
        # x: BxC0xExExE, p: BxKxC0xExExE
        b, k, c, e = [p.shape[0], p.shape[1], p.shape[2], p.shape[3]]
        x_feat = self.theta(x)
        x_feat_flat = x_feat.reshape((b, -1))
        p_feat = self.phi(p.reshape(b * k, -1, e, e, e))
        p_feat_flat = p_feat.reshape((b, k, -1))
        if self.normalize:
            x_feat_flat = nn.functional.normalize(x_feat_flat, dim=1)
            p_feat_flat = nn.functional.normalize(p_feat_flat, dim=2)
        g_feat_flat = self.g(p.reshape(b * k, -1, e, e, e)).reshape((b, k, -1, e, e, e)).reshape((b, k, -1))

        scores = torch.einsum('ij,ijk->ik', x_feat_flat, p_feat_flat.permute((0, 2, 1)))
        # switch = self.sigmoid(self.max(scores.view(b, 1, k)).view(b, 1) * self.sig_scale + self.sig_shift) if self.use_switching else 1
        # nihalsid: try simple relu switching instead of learning shift and scale
        switch = self.relu(self.max(scores.view(b, 1, k)).view(b, 1))
        if self.retrieval_mode:
            scaled_similarity_scores = scores * 25
            weights = torch.nn.functional.gumbel_softmax(scaled_similarity_scores, tau=1, hard=True)
            weighted_sum = torch.einsum('ij,ijk->ik', weights, g_feat_flat)
        else:
            sharpness = (self.cf_feat * e * e * e) * 4
            weights = self.softmax(sharpness * scores)
            weighted_sum = torch.einsum('ij,ijk->ik', weights, g_feat_flat).reshape((b, -1, e, e, e))  # BxCFxExExE
        patch_attention = self.o(weighted_sum)
        if self.blend_mode:
            output = (x.view(b, c * e * e * e) * (1 - switch)).view(b, c, e, e, e) + (patch_attention.view(b, c * e * e * e) * switch).view(b, c, e, e, e)
        else:
            output = x + (patch_attention.view(b, c * e * e * e) * switch).view(b, c, e, e, e)
        return output

    def get_regularization_losses(self):
        return ((self.sig_scale - self.init_scale) ** 2 + (self.sig_shift - self.init_shift) ** 2) if self.use_switching else 0


class PatchedAttentionBlock(nn.Module):

    def __init__(self, nf, num_patch_x, patch_extent, num_nearest_neighbors, attention_block):
        super().__init__()
        self.num_patch_x = num_patch_x
        self.patch_extent = patch_extent
        self.num_nearest_neighbors = num_nearest_neighbors
        self.nf = nf
        self.attention_blocks_layer = attention_block
        self.fold_3d = Fold3D(num_patch_x, patch_extent, self.nf)
        self.unfold_3d = Unfold3D(patch_extent, self.nf)
        self.unfold_3d_occ = Unfold3D(patch_extent, 1)

    def get_features(self, x_predicted, x_target, occupancy):
        x_predicted_feat_ = self.unfold_3d(x_predicted)
        x_target_feat_ = self.unfold_3d(x_target)
        occupancy_ = self.unfold_3d_occ(occupancy)
        x_feat_flat, p_feat_flat = self.attention_blocks_layer.get_features(x_predicted_feat_, x_target_feat_)
        occupancy_flat = occupancy_.reshape((x_predicted_feat_.shape[0], -1))
        occupancy_flat = occupancy_flat.any(dim=1)
        return x_feat_flat, p_feat_flat, occupancy_flat

    def forward(self, x_predicted, x_retrieved):
        # x_predicted: BxFxSxSxS
        # x_retrieved: B.KxFxSxSxS
        shape_dim = x_retrieved.shape[-1]
        x_patch = x_retrieved.reshape(-1, self.nf, shape_dim, shape_dim, shape_dim)
        # x_predicted_feat_: (B.R.R.RxFxExExE)
        x_predicted_feat_ = self.unfold_3d(x_predicted)
        # x_patch_feat_: (B.K.R.R.RxFxExExE)
        x_patch_feat_ = self.unfold_3d(x_patch)
        # x_predicted_feat_: (B.R.R.RxKxFxExExE)
        x_patch_feat_ = x_patch_feat_.reshape((-1, self.num_nearest_neighbors, self.num_patch_x, self.num_patch_x, self.num_patch_x, self.nf, self.patch_extent, self.patch_extent, self.patch_extent)).permute(
            (0, 2, 3, 4, 1, 5, 6, 7, 8)).reshape((-1, self.num_nearest_neighbors, self.nf, self.patch_extent, self.patch_extent, self.patch_extent))
        # attention_processed: (B.R.R.RxFxExExE)
        weights = scores = switch = None
        attention_processed = self.attention_blocks_layer(x_predicted_feat_, x_patch_feat_)
        output_feats = self.fold_3d(attention_processed)
        return output_feats


class Fold3D(nn.Module):

    def __init__(self, num_patch_x, patch_extent, nf):
        super().__init__()
        self.nf = nf
        self.num_patch_x = num_patch_x
        self.patch_extent = patch_extent
        self.fold_0 = torch.nn.Fold(output_size=(num_patch_x * patch_extent, num_patch_x * patch_extent), kernel_size=(patch_extent, patch_extent), stride=(patch_extent, patch_extent))
        self.fold_1 = torch.nn.Fold(output_size=(1, num_patch_x * patch_extent), kernel_size=(1, patch_extent), stride=(1, patch_extent))

    def forward(self, x):
        fold_in = x.reshape((-1, self.num_patch_x, self.num_patch_x, self.num_patch_x, self.nf, self.patch_extent, self.patch_extent, self.patch_extent))
        fold_in = fold_in.permute((0, 4, 5, 1, 6, 7, 2, 3)).reshape((-1, self.nf * self.patch_extent * self.num_patch_x * self.patch_extent * self.patch_extent, self.num_patch_x * self.num_patch_x))
        fold_out = self.fold_0(fold_in).reshape((-1, self.nf, self.patch_extent, self.num_patch_x, self.num_patch_x * self.patch_extent, self.num_patch_x * self.patch_extent))
        fold_in = fold_out.permute((0, 1, 4, 5, 2, 3)).reshape((-1, self.nf * self.num_patch_x * self.patch_extent * self.num_patch_x * self.patch_extent * self.patch_extent, self.num_patch_x))
        fold_out = self.fold_1(fold_in).squeeze().reshape((-1, self.nf, self.num_patch_x * self.patch_extent, self.num_patch_x * self.patch_extent, self.num_patch_x * self.patch_extent)).permute((0, 1, 4, 2, 3))
        return fold_out


class Unfold3D(nn.Module):

    def __init__(self, patch_extent, nf):
        super().__init__()
        self.patch_extent = patch_extent
        self.nf = nf

    def forward(self, x):
        unfold_out = x.unfold(2, self.patch_extent, self.patch_extent).unfold(3, self.patch_extent, self.patch_extent).unfold(4, self.patch_extent, self.patch_extent)
        return unfold_out.permute((0, 2, 3, 4, 1, 5, 6, 7)).reshape((-1, self.nf, self.patch_extent, self.patch_extent, self.patch_extent))


class Unfold3DPadStride(nn.Module):

    def __init__(self, patch_extent, pad_size, pad_val, stride):
        super().__init__()
        self.patch_extent = patch_extent
        self.pad_size = pad_size
        self.pad_val = pad_val
        self.stride = stride

    def forward(self, x):
        padded_x = torch.nn.functional.pad(x, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='constant', value=self.pad_val)
        padded_x = padded_x.unfold(2, self.patch_extent, self.stride).unfold(3, self.patch_extent, self.stride).unfold(4, self.patch_extent, self.stride)
        return padded_x.permute((0, 2, 3, 4, 1, 5, 6, 7)).reshape((-1, 1, self.patch_extent, self.patch_extent, self.patch_extent))
