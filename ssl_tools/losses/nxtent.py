import torch
import numpy as np
import lightning as L


class NTXentLoss_poly(L.LightningModule):
    def __init__(
        self,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
    ):
        super(NTXentLoss_poly, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

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

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]
        mask_samples_from_same_repr = self._get_correlated_mask(
            batch_size
        ).type(torch.bool)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(
            representations, representations
        )

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(
            2 * batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = (
            torch.cat(
                (
                    torch.ones(2 * batch_size, 1),
                    torch.zeros(2 * batch_size, negatives.shape[-1]),
                ),
                dim=-1,
            )
            .to(self.device)
            .long()
        )
        # Add poly loss
        pt = torch.mean(
            onehot_label * torch.nn.functional.softmax(logits, dim=-1)
        )

        epsilon = batch_size
        # loss = CE/ (2 * batch_size) + epsilon*(1-pt) # replace 1 by 1/batch_size
        loss = CE / (2 * batch_size) + epsilon * (1 / batch_size - pt)
        # loss = CE / (2 * batch_size)

        return loss
