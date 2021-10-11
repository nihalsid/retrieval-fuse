import torch


class Patcher(object):

    def __init__(self, patch_size, side, stride, pad_val, base_size):
        super().__init__()
        self.pad = side
        self.stride = stride
        self.pad_val = pad_val
        self.base_size = base_size
        self.kernel = [patch_size[i] + 2 * side[i] for i in range(len(patch_size))]

    def __call__(self, x):
        volume = torch.ones([x.shape[0], x.shape[1], x.shape[2] + self.pad[0] * 2, x.shape[3] + self.pad[1] * 2, x.shape[4] + self.pad[2] * 2], dtype=x.dtype).to(x.device) * self.pad_val
        volume[:, :, self.pad[0]: volume.shape[2] - self.pad[0], self.pad[1]: volume.shape[3] - self.pad[1], self.pad[2]: volume.shape[4] - self.pad[2]] = x
        volume = volume.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1]).unfold(4, self.kernel[2], self.stride[2]).permute((0, 2, 3, 4, 1, 5, 6, 7))
        volume = volume.reshape((-1, x.shape[1], self.kernel[0], self.kernel[1], self.kernel[2]))
        return volume

    def recompose_patches(self, original_shape, patches):
        volume = torch.ones([original_shape[0], original_shape[1], original_shape[2] + self.pad[0] * 2, original_shape[3] + self.pad[1] * 2, original_shape[4] + self.pad[2] * 2], dtype=patches.dtype).to(patches.device) * self.pad_val
        patch_ctr = 0
        for x in range(0, volume.shape[2] - patches.shape[2] + 1, self.stride[0]):
            for y in range(0, volume.shape[3] - patches.shape[3] + 1, self.stride[1]):
                for z in range(0, volume.shape[4] - patches.shape[2] + 1, self.stride[2]):
                    volume[:, :, x: x + self.kernel[0], y: y + self.kernel[1], z: z + self.kernel[2]] = patches[:, patch_ctr: patch_ctr + 1, :, :, :]
                    patch_ctr += 1
        volume = volume[:, :, self.pad[0]: volume.shape[2] - self.pad[0], self.pad[1]: volume.shape[3] - self.pad[1], self.pad[2]: volume.shape[4] - self.pad[2]]
        return volume

    def get_patch_extents(self):
        return [self.kernel[i] - 2 * self.pad[i] for i in range(3)]

    def get_patch_ratio(self):
        return [self.base_size[i] // (self.kernel[i] - 2 * self.pad[i]) for i in range(3)]

    def get_stride_ratio(self):
        return [self.get_patch_extents()[i] // self.stride[i] for i in range(3)]

    def get_patch_counts(self):
        return [(self.base_size[i] + self.pad[i] * 2 - self.kernel[i]) // self.stride[i] + 1 for i in range(3)]
