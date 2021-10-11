from torch import nn

from model.unet import UNet3D, DecoderNoJoining


class Superresolution08UNetBackbone(nn.Module):

    def __init__(self, nf, num_levels, layer_order):
        super().__init__()
        self.network = nn.ModuleList([
            UNet3D(in_channels=1, out_channels=2 * nf, final_sigmoid=False, final_conv=False, f_maps=nf, num_groups=nf // 2, layer_order=layer_order, num_levels=num_levels, is_segmentation=False),
            DecoderNoJoining(2 * nf, 2 * nf, conv_layer_order=layer_order, num_groups=nf // 2),
            DecoderNoJoining(2 * nf, nf, conv_layer_order=layer_order, num_groups=nf // 2),
        ])

    def forward(self, x):
        for net in self.network:
            x = net(x)
        return x


class Superresolution16UNetBackbone(nn.Module):

    def __init__(self, nf, num_levels, layer_order):
        super().__init__()
        self.network = nn.ModuleList([
            UNet3D(in_channels=1, out_channels=2 * nf, final_sigmoid=False, final_conv=False, f_maps=nf, num_groups=nf // 2, layer_order=layer_order, num_levels=num_levels, is_segmentation=False),
            DecoderNoJoining(2 * nf, nf, conv_layer_order=layer_order, num_groups=nf // 2),
        ])

    def forward(self, x):
        for net in self.network:
            x = net(x)
        return x


class SurfaceReconstructionUNetBackbone(nn.Module):

    def __init__(self, nf, num_levels, layer_order):
        super().__init__()
        self.network = UNet3D(in_channels=1, out_channels=nf, final_sigmoid=False, final_conv=False, remove_n_final_layers=2, f_maps=nf, layer_order=layer_order, num_groups=nf // 2, num_levels=num_levels, is_segmentation=False)

    def forward(self, x):
        x = self.network(x)
        return x


class Superresolution08FinalDecoder(nn.Module):

    def __init__(self, nf, layer_order):
        super().__init__()
        self.network = nn.ModuleList([
            DecoderNoJoining(nf, nf, conv_layer_order=layer_order, num_groups=nf // 2),
            nn.Conv3d(nf, 1, 1, padding=0),
            nn.Tanh()
        ])

    def forward(self, x):
        for net in self.network:
            x = net(x)
        return x


class RetrievalUNetBackbone(nn.Module):

    def __init__(self, f_maps, nf, num_levels, layer_order):
        super().__init__()
        self.nf = nf
        self.network = UNet3D(in_channels=1, out_channels=nf, num_groups=nf // 2, final_sigmoid=False, final_conv=False, remove_n_final_layers=1, f_maps=f_maps, layer_order=layer_order, num_levels=num_levels, is_segmentation=False)

    def forward(self, x):
        x = self.network(x)
        return x
