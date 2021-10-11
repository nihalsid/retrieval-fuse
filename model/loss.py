import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature, use_cosine_similarity, sig_scale=80, sig_shift=-65):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.sig_scale = sig_scale
        self.sig_shift = sig_shift
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _get_correlated_mask(batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, iou_matrix=None):
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        batch_mask = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[batch_mask.cuda(zis.device)].view(2 * batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        if iou_matrix is None:
            logits /= self.temperature
        else:
            negative_ious = iou_matrix[batch_mask.cuda(zis.device)].view(2 * batch_size, -1)
            logits[:, 0] /= self.temperature
            logits[:, 1:] /= (self.temperature + (1 - self.temperature) * self.sigmoid(negative_ious * self.sig_scale + self.sig_shift))

        labels = torch.zeros(2 * batch_size).to(zis.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


def patch_style_loss(zis, zjs):
    gmi = torch.mm(zis, zis.t())
    gmj = torch.mm(zjs, zjs.t()).detach()
    return torch.nn.functional.mse_loss(gmi, gmj)


def get_cosine_similarity(pred_norms, target_norms):
    pred_norms_ = pred_norms.permute((0, 2, 3, 4, 1)).reshape((-1, 3))
    target_norms_ = target_norms.permute((0, 2, 3, 4, 1)).reshape((-1, 3))
    pred_norms_mask = torch.norm(pred_norms_, dim=1) != 0
    target_norms_mask = torch.norm(target_norms_, dim=1) != 0
    valid_norms = pred_norms_mask & target_norms_mask
    loss_norm = torch.cosine_similarity(torch.nn.functional.normalize(pred_norms_[valid_norms], p=2, dim=1), torch.nn.functional.normalize(target_norms_[valid_norms], p=2, dim=1)).mean()
    return loss_norm
