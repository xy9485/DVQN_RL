import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.cluster import KMeans
from torch import Tensor, nn


class Batch_KMeans(nn.Module):
    def __init__(
        self,
        n_clusters,
        embedding_dim,
        decay=0.99,
        epsilon=1e-5,
        device="cuda",
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.register_buffer("centroids", torch.randn(n_clusters, embedding_dim))
        # self.centroids = torch.randn(self.n_clusters, self.embedding_dim)
        # self.centroids.requires_grad_(False)
        # self.centroids.retain_grad()
        # self.count = 100 * torch.zeros(self.n_clusters)
        self.register_buffer("_ema_cluster_size", torch.zeros(n_clusters))
        self.register_buffer("_ema_w", torch.randn(n_clusters, embedding_dim))
        # self._ema_w = torch.randn(n_clusters, embedding_dim)
        self._decay = decay
        self._epsilon = epsilon

    @torch.no_grad()
    def _compute_distances(self, X):
        # X = torch.norm(X, p=2, dim=1)
        # pass
        X = F.normalize(X, p=2, dim=1)
        distances = (
            torch.sum(X**2, dim=1, keepdim=True)
            + torch.sum(self.centroids**2, dim=1)
            - 2 * torch.matmul(X, self.centroids.t())
        )

        return distances

    @torch.no_grad()
    def init_cluster(self, X: Tensor):
        """Generate initial clusters using sklearn.Kmeans"""
        print("========== Start Initial Clustering ==========")
        self.model = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=0)
        # X = preprocessing.normalize(X)
        self.init_cluster_indices = self.model.fit_predict(preprocessing.normalize(X.cpu().numpy()))
        self.centroids = torch.from_numpy(self.model.cluster_centers_).to(X.device)  # copy clusters
        print("========== End Initial Clustering ==========")

    @torch.no_grad()
    def assign_clusters(self, X):
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def assign_centroid(self, X, update_centroid=True):
        distances = self._compute_distances(X)
        cluster_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        cluster_onehots = torch.zeros(cluster_indices.shape[0], self.n_clusters, device=X.device)
        cluster_onehots.scatter_(1, cluster_indices, 1)
        quantized = torch.matmul(cluster_onehots, self.centroids)

        if update_centroid:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(cluster_onehots, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self.n_clusters * self._epsilon) * n
            )

            dw = torch.matmul(cluster_onehots.t(), X)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self.centroids = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        avg_probs = torch.mean(cluster_onehots, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        cluster_entrophy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        return quantized, cluster_indices.squeeze(), cluster_entrophy.item()

    # def fit(self, inputs):
    #     inputs = inputs.permute(0, 2, 3, 1).contiguous()
    #     inputs = inputs.view(-1, self.embedding_dim)
    #     for i in range(self.num_embeddings):
    #         self.centroids[i] = torch.mean(
    #             inputs[torch.argmax(torch.abs(inputs - self.centroids[i]))]
    #         )
    #     return self.centroids

    # def predict(self, inputs):
    #     inputs = inputs.permute(0, 2, 3, 1).contiguous()
    #     inputs = inputs.view(-1, self.embedding_dim)
    #     distances = (
    #         torch.sum(inputs**2, dim=1, keepdim=True)
    #         + torch.sum(self.centroids**2, dim=1)
    #         - 2 * torch.matmul(inputs, self.centroids.t())
    #     )
    #     encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #     encodings = torch.zeros(
    #         encoding_indices.shape[0], self.num_embeddings, device=inputs.device
    #     )
    #     encodings.scatter_(1, encoding_indices, 1)
    #     return encodings
