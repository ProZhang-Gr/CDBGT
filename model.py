import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ==================== 特征投影模块 ====================
class FeatureProjection(nn.Module):
    """特征投影模块：将不同维度的特征投影到指定的隐藏空间，可选是否使用交叉注意力机制"""

    def __init__(self, c_feature_dim, d_feature_dim, projection_dim=128,
                 intermediate_dim=None, dropout=0.2):
        super(FeatureProjection, self).__init__()

        # 如果未指定中间层维度，默认为投影维度的2倍
        if intermediate_dim is None:
            intermediate_dim = projection_dim * 2

        # CircRNA特征投影
        self.c_projection = nn.Sequential(
            nn.Linear(c_feature_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, projection_dim)
        )

        # 药物特征投影
        self.d_projection = nn.Sequential(
            nn.Linear(d_feature_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, projection_dim)
        )

    def forward(self, cf, df):
        # 投影特征到指定维度
        cf_proj = self.c_projection(cf)  # [num_circs, projection_dim]
        df_proj = self.d_projection(df)  # [num_drugs, projection_dim]

        return cf_proj, df_proj


# ==================== 结构感知位置编码 ====================
# 直接修改 StructuralPositionalEncoding 类进行手动对比实验
# 重命名后的 StructuralPositionalEncoding 类进行手动对比实验

class StructuralPositionalEncoding(nn.Module):
    """图结构感知位置编码：通过参数控制不同编码策略的对比实验"""
#self, hidden_dim, max_degree=64, num_eigvecs=8,
                 #use_degree=True, use_degree_rank=True, use_spectral=True):
    def __init__(self, hidden_dim, max_degree=64, num_eigvecs=8,
                 use_degree=True, use_degree_rank=True, use_spectral=True):
        super(StructuralPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_eigvecs = num_eigvecs
        self.use_degree = use_degree
        self.use_degree_rank = use_degree_rank  
        self.use_spectral = use_spectral

        # 计算实际使用的组件数量
        num_components = sum([use_degree, use_degree_rank, use_spectral])
        
        # 如果没有启用任何组件，直接返回原特征
        if num_components == 0:
            self.projection = nn.Identity()
            return
            
        # === 度数绝对值编码模块 ===
        if use_degree:
            self.degree_embedding = nn.Embedding(max_degree + 1, hidden_dim // 4)

        # === 度数相对排名编码模块 ===  
        if use_degree_rank:
            self.degree_rank_embedding = nn.Embedding(10, hidden_dim // 4)

        # === 特征融合模块 ===
        # 计算输入维度
        input_dim = 0
        if use_degree:
            input_dim += hidden_dim // 4
        if use_degree_rank:
            input_dim += hidden_dim // 4
        if use_spectral:
            input_dim += num_eigvecs
            
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: 节点特征矩阵 [batch_size, hidden_dim]
            adjacency_matrix: 图的邻接矩阵 [batch_size, batch_size]

        Returns:
            增强后的节点特征 [batch_size, hidden_dim]
        """
        batch_size = node_features.size(0)
        device = node_features.device

        # 如果没有启用任何编码组件，直接返回原特征
        if not any([self.use_degree, self.use_degree_rank, self.use_spectral]):
            return node_features

        encoding_components = []

        # ==================== 节点度数计算 ====================
        if self.use_degree or self.use_degree_rank:
            node_degrees = torch.sum(adjacency_matrix > 0, dim=1).clamp(0, self.max_degree).long()

        # ==================== 度数绝对值编码 ====================
        if self.use_degree:
            degree_embeddings = self.degree_embedding(node_degrees)
            encoding_components.append(degree_embeddings)

        # ==================== 度数相对排名编码 ====================
        if self.use_degree_rank:
            max_degree_in_graph = node_degrees.float().max()
            if max_degree_in_graph > 0:
                degree_rank_score = (node_degrees.float() / max_degree_in_graph * 9).long()
            else:
                degree_rank_score = torch.zeros_like(node_degrees)
            degree_rank_embeddings = self.degree_rank_embedding(degree_rank_score)
            encoding_components.append(degree_rank_embeddings)

        # ==================== 拉普拉斯谱位置编码计算 ====================
        if self.use_spectral:
            try:
                laplacian_matrix = self._get_laplacian_matrix(adjacency_matrix)
                eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)

                # 选择最小的num_eigvecs个非零特征值对应的特征向量
                laplacian_eigenvector_features = eigenvectors[:, 1:self.num_eigvecs + 1]

                # 处理维度不匹配
                if laplacian_eigenvector_features.size(1) < self.num_eigvecs:
                    padding = torch.zeros(batch_size, self.num_eigvecs - laplacian_eigenvector_features.size(1), device=device)
                    laplacian_eigenvector_features = torch.cat([laplacian_eigenvector_features, padding], dim=1)
                elif laplacian_eigenvector_features.size(1) > self.num_eigvecs:
                    laplacian_eigenvector_features = laplacian_eigenvector_features[:, :self.num_eigvecs]

                encoding_components.append(laplacian_eigenvector_features)
                
            except Exception as e:
                # 如果谱分解失败，使用零向量
                print(f"拉普拉斯谱分解失败: {e}")
                laplacian_eigenvector_features = torch.zeros(batch_size, self.num_eigvecs, device=device)
                encoding_components.append(laplacian_eigenvector_features)

        # ==================== 特征组合与投影 ====================
        combined_features = torch.cat(encoding_components, dim=1)
        positional_encoding = self.projection(combined_features)

        # 残差连接
        return node_features + positional_encoding
    def _get_laplacian_matrix(self, adjacency_matrix):
        """计算图的拉普拉斯矩阵，增强数值稳定性"""
        # 增加更大的正则化项
        reg_term = 1e-4
        
        # 确保邻接矩阵对称
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2
        
        # 计算度矩阵
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        
        # 拉普拉斯矩阵
        laplacian_matrix = degree_matrix - adjacency_matrix
        
        # 添加更强的正则化
        laplacian_matrix = laplacian_matrix + torch.eye(
            adjacency_matrix.size(0), 
            device=adjacency_matrix.device
        ) * reg_term
        
        return laplacian_matrix


# ==================== 图注意力网络(GAT)层 ====================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.2, alpha=0.2, use_bn=True, activation=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_bn = use_bn
        self.activation = activation
        self.head_dim = out_dim // num_heads
        assert out_dim == self.head_dim * num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * self.head_dim)))
        nn.init.xavier_normal_(self.a)
        self.leakyrelu = nn.LeakyReLU(alpha)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):  # adj是邻接矩阵，x是节点特征
        num_nodes = x.size(0)  # 结点的数目
        Wh = self.W(x)  # [num_nodes, out_dim]
        Wh = Wh.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        # 构造注意力机制输入
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [num_nodes, num_heads, num_nodes, head_dim]
        Wh2 = Wh.unsqueeze(0).expand(num_nodes, -1, -1, -1)  # [num_nodes, num_heads, num_nodes, head_dim]
        Wh2 = Wh2.permute(0, 2, 1, 3)  # [num_nodes, num_heads, num_nodes, head_dim]
        concat = torch.cat([Wh1, Wh2], dim=3)  # [num_nodes, num_heads, num_nodes, 2*head_dim]
        # 计算注意力系数
        a_input = self.a.view(1, self.num_heads, 1, 2 * self.head_dim)  # [1, num_heads, 1, 2*head_dim]
        e = torch.matmul(concat, a_input.transpose(2, 3))  # [num_nodes, num_heads, num_nodes, 1]
        e = self.leakyrelu(e.squeeze(3))  # [num_nodes, num_heads, num_nodes]
        # 掩码处理
        adj = adj.unsqueeze(1).expand(-1, self.num_heads, -1)  # [num_nodes, num_heads, num_nodes]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # Softmax
        attention = F.softmax(attention, dim=2)  # [num_nodes, num_heads, num_nodes]
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 加权求和
        h_prime = torch.einsum('bhn,bhd->bhnd', attention, Wh)  # [num_nodes, num_heads, num_nodes, head_dim]
        h_prime = h_prime.sum(dim=2)  # [num_nodes, num_heads, head_dim]
        # 合并多头输出
        h_prime = h_prime.view(num_nodes, -1)  # [num_nodes, out_dim]
        if self.use_bn:
            h_prime = self.bn(h_prime)
        if self.activation:
            h_prime = F.relu(h_prime)
        h_prime = F.dropout(h_prime, self.dropout, training=self.training)
        return h_prime


# ==================== BipartiteGraphTransformer模块 ====================
class BipartiteGraphTransformer(nn.Module):
    """用于二分图的图Transformer"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.2, use_structure_pe=False):
        super(BipartiteGraphTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_structure_pe = use_structure_pe
        assert hidden_dim % num_heads == 0
        # 如果使用结构感知位置编码
        if use_structure_pe:
            self.c_pos_enc = StructuralPositionalEncoding(hidden_dim)
            self.d_pos_enc = StructuralPositionalEncoding(hidden_dim)
        # circRNA到drug的投影
        self.c2d_q = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_k = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_v = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_out = nn.Linear(hidden_dim, hidden_dim)
        # drug到circRNA的投影
        self.d2c_q = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_k = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_v = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_out = nn.Linear(hidden_dim, hidden_dim)
        # 层归一化
        self.c_norm1 = nn.LayerNorm(hidden_dim)
        self.c_norm2 = nn.LayerNorm(hidden_dim)
        self.d_norm1 = nn.LayerNorm(hidden_dim)
        self.d_norm2 = nn.LayerNorm(hidden_dim)
        # 前馈网络
        self.c_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.d_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, c_features, d_features, cd_adj):
        """
        Args:
            c_features: circRNA特征 [num_circs, hidden_dim]
            d_features: 药物特征 [num_drugs, hidden_dim]
            cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        """
        num_circs = c_features.size(0)
        num_drugs = d_features.size(0)
        # circRNA残差连接
        c_residual = c_features
        d_residual = d_features
        # 应用位置编码（如果使用）
        if self.use_structure_pe:
            c_adj = torch.mm(cd_adj, cd_adj.t())  # [num_circs, num_circs]
            d_adj = torch.mm(cd_adj.t(), cd_adj)  # [num_drugs, num_drugs]
            c_features = self.c_pos_enc(c_features, c_adj)  # 128
            d_features = self.d_pos_enc(d_features, d_adj)  # 128

        # circRNA到drug的注意力
        c_q = self.c2d_q(c_features).view(num_circs, self.num_heads, self.head_dim)
        d_k = self.c2d_k(d_features).view(num_drugs, self.num_heads, self.head_dim)
        d_v = self.c2d_v(d_features).view(num_drugs, self.num_heads, self.head_dim)
        c2d_scores = torch.bmm(
            c_q.transpose(0, 1),  # [num_heads, num_circs, head_dim]
            d_k.transpose(0, 1).transpose(1, 2)  # [num_heads, head_dim, num_drugs]
        ) / math.sqrt(self.head_dim)  # [num_heads, num_circs, num_drugs]
        c2d_scores = c2d_scores.permute(1, 2, 0)  # [num_circs, num_drugs, num_heads]
        # 应用邻接矩阵掩码
        mask = (cd_adj == 0).unsqueeze(-1).expand(-1, -1, self.num_heads)
        c2d_scores.masked_fill_(mask, -1e9)
        # 注意力权重
        c2d_weights = F.softmax(c2d_scores, dim=1)  # 对drug维度做softmax
        c2d_weights = self.dropout_layer(c2d_weights)
        # 加权求和得到新的circRNA特征 [num_circs, hidden_dim]
        c_out = torch.zeros(num_circs, self.hidden_dim, device=c_features.device)
        for h in range(self.num_heads):
            c_out_h = torch.mm(c2d_weights[:, :, h], d_v[:, h])  # [num_circs, head_dim]
            c_out[:, h * self.head_dim:(h + 1) * self.head_dim] = c_out_h
        c_out = self.c2d_out(c_out)
        c_out = self.dropout_layer(c_out)
        # 第一个残差连接
        c_out = self.c_norm1(c_residual + c_out)
        # 前馈网络
        c_ff_out = self.c_ffn(c_out)
        # 第二个残差连接
        c_final = self.c_norm2(c_out + c_ff_out)

        # drug到circRNA的注意力
        d_q = self.d2c_q(d_features).view(num_drugs, self.num_heads, self.head_dim)
        c_k = self.d2c_k(c_features).view(num_circs, self.num_heads, self.head_dim)
        c_v = self.d2c_v(c_features).view(num_circs, self.num_heads, self.head_dim)
        d2c_scores = torch.bmm(
            d_q.transpose(0, 1),  # [num_heads, num_drugs, head_dim]
            c_k.transpose(0, 1).transpose(1, 2)  # [num_heads, head_dim, num_circs]
        ) / math.sqrt(self.head_dim)  # [num_heads, num_drugs, num_circs]
        d2c_scores = d2c_scores.permute(1, 2, 0)  # [num_drugs, num_circs, num_heads]
        # 应用邻接矩阵掩码
        mask = (cd_adj.transpose(0, 1) == 0).unsqueeze(-1).expand(-1, -1, self.num_heads)
        d2c_scores.masked_fill_(mask, -1e9)
        # 注意力权重
        d2c_weights = F.softmax(d2c_scores, dim=1)  # 对circRNA维度做softmax
        d2c_weights = self.dropout_layer(d2c_weights)
        # 加权求和得到新的drug特征 [num_drugs, hidden_dim]
        d_out = torch.zeros(num_drugs, self.hidden_dim, device=d_features.device)
        for h in range(self.num_heads):
            d_out_h = torch.mm(d2c_weights[:, :, h], c_v[:, h])  # [num_drugs, head_dim]
            d_out[:, h * self.head_dim:(h + 1) * self.head_dim] = d_out_h
        d_out = self.d2c_out(d_out)
        d_out = self.dropout_layer(d_out)
        # 第一个残差连接
        d_out = self.d_norm1(d_residual + d_out)
        # 前馈网络
        d_ff_out = self.d_ffn(d_out)
        # 第二个残差连接
        d_final = self.d_norm2(d_out + d_ff_out)
        return c_final, d_final


# ==================== 不确定性引导的特征融合模块 ====================
class UncertaintyGuidedFusion(nn.Module):
    """不确定性引导的特征融合：基于不确定性自适应融合不同来源的特征"""

    def __init__(self, hidden_dim, use_homo_features=True, use_hetero_features=True):
        super(UncertaintyGuidedFusion, self).__init__()
        self.use_homo_features = use_homo_features
        self.use_hetero_features = use_hetero_features
        # 估计CS特征的不确定性
        self.cs_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 标量不确定性参数
        )
        # 估计DS特征的不确定性
        self.ds_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 标量不确定性参数
        )
        # 估计CD-C特征的不确定性
        self.cd_c_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 标量不确定性参数
        )
        # 估计CD-D特征的不确定性
        self.cd_d_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 标量不确定性参数
        )
        # 输出投影
        self.c_output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.d_output_projection = nn.Linear(hidden_dim, hidden_dim)
        # 层归一化
        self.c_norm = nn.LayerNorm(hidden_dim)
        self.d_norm = nn.LayerNorm(hidden_dim)

    def forward(self, cs_features, ds_features, cd_c_features, cd_d_features):
        """
        Args:
            cs_features: CS图处理后的circRNA特征 [num_circs, hidden_dim]
            ds_features: DS图处理后的药物特征 [num_drugs, hidden_dim]
            cd_c_features: CD图处理后的circRNA特征 [num_circs, hidden_dim]
            cd_d_features: CD图处理后的药物特征 [num_drugs, hidden_dim]
        """
        # 计算每个特征来源的不确定性
        if self.use_homo_features:
            cs_uncertainty = torch.exp(self.cs_uncertainty(cs_features))
            ds_uncertainty = torch.exp(self.ds_uncertainty(ds_features))
        else:
            # 不使用同质图特征，设置为0或非常小的值
            cs_uncertainty = torch.zeros_like(cs_features[:, 0:1])
            ds_uncertainty = torch.zeros_like(ds_features[:, 0:1])

        if self.use_hetero_features:
            cd_c_uncertainty = torch.exp(self.cd_c_uncertainty(cd_c_features))
            cd_d_uncertainty = torch.exp(self.cd_d_uncertainty(cd_d_features))
        else:
            # 不使用异质图特征，设置为0或非常小的值
            cd_c_uncertainty = torch.zeros_like(cd_c_features[:, 0:1])
            cd_d_uncertainty = torch.zeros_like(cd_d_features[:, 0:1])

        # 对可能出现的全零情况进行处理
        epsilon = 1e-10  # 小值防止除以零
        c_total_precision = cs_uncertainty + cd_c_uncertainty + epsilon
        d_total_precision = ds_uncertainty + cd_d_uncertainty + epsilon

        # 基于不确定性的加权融合
        if self.use_homo_features and not self.use_hetero_features:
            # 只使用同质图特征
            c_fused = cs_features
            d_fused = ds_features
        elif not self.use_homo_features and self.use_hetero_features:
            # 只使用异质图特征
            c_fused = cd_c_features
            d_fused = cd_d_features
        else:
            # 正常融合
            c_fused = (cs_features * cs_uncertainty + cd_c_features * cd_c_uncertainty) / c_total_precision
            d_fused = (ds_features * ds_uncertainty + cd_d_features * cd_d_uncertainty) / d_total_precision
        # 输出投影
        c_output = self.c_output_projection(c_fused)
        d_output = self.d_output_projection(d_fused)
        # 层归一化
        c_output = self.c_norm(c_output)
        d_output = self.d_norm(d_output)
        return c_output, d_output


# ==================== 双线性预测模块 ====================
class BilinearPredictor(nn.Module):
    """双线性预测模块"""

    def __init__(self, hidden_dim, dropout=0.2):  # 这块可以改改
        super(BilinearPredictor, self).__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, c_features, d_features):
        # 应用dropout
        c_features = self.dropout(c_features)
        d_features = self.dropout(d_features)
        # 双线性变换
        output = self.bilinear(c_features, d_features)
        # sigmoid激活
        output = torch.sigmoid(output)
        return output


# ==================== 同质图处理模块工厂 ====================
class HomoGraphEncoder(nn.Module):
    """同质图处理模块：GAT 编码器"""

    def __init__(self, in_dim, hidden_dim, n_layers=2,
                 dropout=0.2, num_heads=8):
        super(HomoGraphEncoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(
                GATLayer(in_dim=input_dim, out_dim=hidden_dim,
                         num_heads=num_heads, dropout=dropout)
            )

    def forward(self, x, adj):
        """
        x: 节点特征 [num_nodes, in_dim]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x


# ==================== 异质图处理模块工厂 ====================
class HeteroGraphEncoder(nn.Module):
    """异质图处理模块：拓扑感知的二分图 Transformer"""

    def __init__(self, hidden_dim, n_layers=2,
                 dropout=0.2, num_heads=8, use_structure_pe=True):
        super(HeteroGraphEncoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                BipartiteGraphTransformer(hidden_dim=hidden_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          use_structure_pe=use_structure_pe)
            )

    def forward(self, c_features, d_features, cd_adj):
        """
        c_features: circRNA特征 [num_circs, hidden_dim]
        d_features: 药物特征 [num_drugs, hidden_dim]
        cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        """
        for layer in self.layers:
            c_features, d_features = layer(c_features, d_features, cd_adj)
        return c_features, d_features

# ==================== CircRNA-Drug 图转换器 ====================
class CDGT(nn.Module):
    """CircRNA-Drug 双路径图转换器 (CDBGT)

    主模型架构：MLP 特征投影 → GAT 同质图编码 + 拓扑感知二分图 Transformer
    （度数/度数排名/谱 三重位置编码）→ 不确定性引导融合 → 双线性预测。
    """

    def __init__(self,
                 c_feature_dim=640,  # circRNA特征维度
                 d_feature_dim=768,  # 药物特征维度
                 projection_dim=512,  # 特征投影模块的输出维度,同时也是异构图的输入输出维度
                 homo_hidden_dim=128,  # 同质图处理模块的隐藏层维度
                 homo_num_layers=2,  # 同质图(GAT)处理器层数
                 hetero_num_layers=2,  # 异质图(BipartiteGraphTransformer)处理器层数
                 homo_num_heads=4,  # 同质图处理器中的注意力头数
                 hetero_num_heads=32,  # 异质图处理器中的注意力头数
                 dropout=0.2,  # Dropout比率
                 use_structure_pe=True  # 拓扑感知位置编码(degree+rank+spectral)
                 ):
        super(CDGT, self).__init__()
        # 1. 特征投影模块
        self.feature_projection = FeatureProjection(
            c_feature_dim=c_feature_dim,
            d_feature_dim=d_feature_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        # 2. 同质图处理模块 (GAT)
        self.cs_encoder = HomoGraphEncoder(
            in_dim=projection_dim,
            hidden_dim=homo_hidden_dim,
            n_layers=homo_num_layers,
            dropout=dropout,
            num_heads=homo_num_heads
        )
        self.ds_encoder = HomoGraphEncoder(
            in_dim=projection_dim,
            hidden_dim=homo_hidden_dim,
            n_layers=homo_num_layers,
            dropout=dropout,
            num_heads=homo_num_heads
        )
        # 3. 异质图处理模块 (拓扑感知二分图 Transformer)
        self.cd_encoder = HeteroGraphEncoder(
            hidden_dim=projection_dim,
            n_layers=hetero_num_layers,
            dropout=dropout,
            num_heads=hetero_num_heads,
            use_structure_pe=use_structure_pe
        )
        #  异质图输出维度到同质图输出维度的投影层
        self.cd_c_projection = nn.Sequential(
            nn.Linear(projection_dim, homo_hidden_dim),
            nn.LayerNorm(homo_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if projection_dim != homo_hidden_dim else nn.Identity()

        self.cd_d_projection = nn.Sequential(
            nn.Linear(projection_dim, homo_hidden_dim),
            nn.LayerNorm(homo_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if projection_dim != homo_hidden_dim else nn.Identity()
        # 4. 不确定性引导的特征融合模块
        self.fusion = UncertaintyGuidedFusion(homo_hidden_dim)
        # 5. 双线性预测模块
        self.predictor = BilinearPredictor(homo_hidden_dim, dropout)

    def forward(self, cf, df, cs_adj, ds_adj, cd_adj, c_idx, d_idx):
        """
        Args:
            cf: circRNA特征 [num_circs, c_feature_dim]
            df: 药物特征 [num_drugs, d_feature_dim]
            cs_adj: CS邻接矩阵 [num_circs, num_circs]
            ds_adj: DS邻接矩阵 [num_drugs, num_drugs]
            cd_adj: CD邻接矩阵 [num_circs, num_drugs]
            c_idx: 待预测的circRNA索引 [batch_size]
            d_idx: 待预测的药物索引 [batch_size]
        """
        # 1. 特征投影
        cf_proj, df_proj = self.feature_projection(cf, df)  # 128维
        # 2. 同质图处理
        cs_features = self.cs_encoder(cf_proj, cs_adj)  # 128
        ds_features = self.ds_encoder(df_proj, ds_adj)  # 128
        # 3. 异质图处理
        cd_c_features, cd_d_features = self.cd_encoder(cf_proj, df_proj, cd_adj)

        cd_c_features = self.cd_c_projection(cd_c_features)  # [num_circs, homo_hidden_dim]
        cd_d_features = self.cd_d_projection(cd_d_features)  # [num_drugs, homo_hidden_dim]

        # 4. 不确定性引导的特征融合
        c_fused, d_fused = self.fusion(cs_features, ds_features, cd_c_features, cd_d_features)
        # 5. 提取待预测节点的特征
        c_selected = c_fused[c_idx]  # [batch_size, hidden_dim]
        d_selected = d_fused[d_idx]  # [batch_size, hidden_dim]
        # 6. 预测
        predictions = self.predictor(c_selected, d_selected)
        return predictions.squeeze()


# ==================== 损失函数 ====================
def loss_function():
    """返回模型的损失函数"""
    return nn.BCELoss()


# ==================== 模型工厂函数 ====================
def create_model(model_config):
    """根据配置创建 CDGT 模型"""
    return CDGT(**model_config)


