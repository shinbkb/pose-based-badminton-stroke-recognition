"""ST-GCN with learnable edge importance and person attention."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_kernel_size,
    ):
        super().__init__()

        self.spatial_kernel_size = (
            spatial_kernel_size
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels
            * spatial_kernel_size,
            kernel_size=1,
        )

    def forward(
        self,
        x,
        adjacency,
    ):
        x = self.conv(x)

        n, kc, t, v = x.shape

        x = x.view(
            n,
            self.spatial_kernel_size,
            kc // self.spatial_kernel_size,
            t,
            v,
        )

        x = torch.einsum(
            "nkctv,kvw->nctw",
            x,
            adjacency,
        )

        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_kernel_size,
        stride=1,
        dropout=0.4,
        residual=True,
    ):
        super().__init__()

        self.gcn = SpatialGraphConv(
            in_channels,
            out_channels,
            spatial_kernel_size,
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),

            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (
            in_channels == out_channels
            and stride == 1
        ):
            self.residual = nn.Identity()

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x,
        adjacency,
    ):
        residual = self.residual(x)

        x = self.gcn(
            x,
            adjacency,
        )

        x = self.tcn(x)
        x = x + residual

        return self.relu(x)


class STGCNV2(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        adjacency,
        dropout=0.4,
    ):
        super().__init__()

        self.register_buffer(
            "adjacency",
            adjacency,
        )

        num_joints = adjacency.shape[-1]

        self.data_bn = nn.BatchNorm1d(
            in_channels * num_joints
        )

        configurations = [
            (in_channels, 32, 1, False),
            (32, 32, 1, True),
            (32, 64, 2, True),
            (64, 64, 1, True),
            (64, 128, 2, True),
            (128, 128, 1, True),
        ]

        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    spatial_kernel_size=(
                        adjacency.shape[0]
                    ),
                    stride=stride,
                    dropout=dropout,
                    residual=residual,
                )
                for (
                    in_c,
                    out_c,
                    stride,
                    residual,
                ) in configurations
            ]
        )

        self.edge_importance = (
            nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones_like(
                            adjacency
                        )
                    )
                    for _ in self.blocks
                ]
            )
        )

        self.person_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        # x: (N, C, T, V, M)
        n, c, t, v, m = x.shape

        x = x.permute(
            0, 4, 3, 1, 2
        ).contiguous()

        x = x.view(
            n * m,
            v * c,
            t,
        )

        x = self.data_bn(x)

        x = x.view(
            n,
            m,
            v,
            c,
            t,
        )

        x = x.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        x = x.view(
            n * m,
            c,
            t,
            v,
        )

        for block, importance in zip(
            self.blocks,
            self.edge_importance,
        ):
            x = block(
                x,
                self.adjacency
                * importance,
            )

        x = F.adaptive_avg_pool2d(
            x,
            output_size=1,
        )

        x = x.view(
            n,
            m,
            128,
        )

        attention_logits = (
            self.person_attention(x)
            .squeeze(-1)
        )

        attention = torch.softmax(
            attention_logits,
            dim=1,
        )

        pooled = (
            x
            * attention.unsqueeze(-1)
        ).sum(dim=1)

        logits = self.classifier(
            pooled
        )

        if return_attention:
            return logits, attention

        return logits
