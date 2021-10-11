from torch import nn


class Patch32(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class PatchNorm32(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(2 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(4 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(8 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(8 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(8 * nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch04(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4**3, nf * 4),
            nn.ReLU(),
            nn.Linear(nf * 4, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, nf * 16),
            nn.ReLU(),
            nn.Linear(nf * 16, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, z_dim),
        ])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for f in self.layers:
            x = f(x)
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch05(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(5**3, nf * 4),
            nn.ReLU(),
            nn.Linear(nf * 4, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, nf * 16),
            nn.ReLU(),
            nn.Linear(nf * 16, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, z_dim),
        ])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for f in self.layers:
            x = f(x)
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch04V2(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4**3, nf * 4),
            nn.ReLU(),
            nn.Linear(nf * 4, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, nf * 16),
            nn.ReLU(),
            nn.Linear(nf * 16, nf * 16),
            nn.ReLU(),
            nn.Linear(nf * 16, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, z_dim),
        ])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for f in self.layers:
            x = f(x)
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


# noinspection DuplicatedCode
class Patch08(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


# noinspection DuplicatedCode
class PatchNorm08(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(4 * nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm3d(8 * nf),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class PCPatch32(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


# noinspection DuplicatedCode
class PCPatch48(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


# noinspection DuplicatedCode
class PCPatch64(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


# noinspection DuplicatedCode
class Patch16(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch24(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 2 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch24V2(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 2 * nf, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])


class Patch12(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf, 2 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(2 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 4 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(8 * nf, 8 * nf, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        x = self.final_layer(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x.reshape([x.shape[0], x.shape[1], 1, 1, 1])
