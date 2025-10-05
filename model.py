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


# ==================== 图卷积网络(GCN)层 ====================
class GCNLayer(nn.Module):
    """图卷积网络层"""

    def __init__(self, in_dim, out_dim, dropout=0.2, use_bn=True, activation=True):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        """
        x: 节点特征 [num_nodes, in_dim]
        adj: 归一化的邻接矩阵 [num_nodes, num_nodes]
        """
        # 对邻接矩阵进行归一化处理
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum + 1e-6, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        # GCN传播规则
        support = self.linear(x)
        output = torch.mm(adj_normalized, support)
        # 归一化和激活
        if self.use_bn:
            output = self.bn(output)
        if self.activation:
            output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        return output

# ==================== 二分图GAT ====================
class BipartiteGAT(nn.Module):
    """专门用于二分图的GAT层，不使用双向注意力，只是单纯的GAT处理二分图结构"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.2, alpha=0.2, use_bn=True, activation=True):
        super(BipartiteGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = activation
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # CircRNA节点的注意力计算
        self.c_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c_a = nn.Parameter(torch.zeros(size=(num_heads, 2 * self.head_dim)))
        
        # Drug节点的注意力计算
        self.d_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.d_a = nn.Parameter(torch.zeros(size=(num_heads, 2 * self.head_dim)))
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # 批归一化
        if use_bn:
            self.c_bn = nn.BatchNorm1d(hidden_dim)
            self.d_bn = nn.BatchNorm1d(hidden_dim)
        
        # 参数初始化
        nn.init.xavier_normal_(self.c_a)
        nn.init.xavier_normal_(self.d_a)
    
    def forward(self, c_features, d_features, cd_adj):
        """
        Args:
            c_features: circRNA特征 [num_circs, hidden_dim]
            d_features: 药物特征 [num_drugs, hidden_dim]
            cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        
        Returns:
            c_output: 更新后的circRNA特征 [num_circs, hidden_dim]
            d_output: 更新后的药物特征 [num_drugs, hidden_dim]
        """
        num_circs = c_features.size(0)
        num_drugs = d_features.size(0)
        
        # 1. 对CircRNA节点应用GAT，聚合来自Drug节点的信息
        c_output = self._gat_aggregation(
            source_features=d_features,  # 来源：Drug节点
            target_features=c_features,  # 目标：CircRNA节点
            adjacency=cd_adj,  # [num_circs, num_drugs]
            W=self.c_W,
            a=self.c_a,
            is_circRNA=True
        )
        
        # 2. 对Drug节点应用GAT，聚合来自CircRNA节点的信息
        d_output = self._gat_aggregation(
            source_features=c_features,  # 来源：CircRNA节点
            target_features=d_features,  # 目标：Drug节点
            adjacency=cd_adj.t(),  # [num_drugs, num_circs] (转置)
            W=self.d_W,
            a=self.d_a,
            is_circRNA=False
        )
        
        # 3. 批归一化
        if self.use_bn:
            c_output = self.c_bn(c_output)
            d_output = self.d_bn(d_output)
        
        # 4. 激活函数
        if self.activation:
            c_output = F.relu(c_output)
            d_output = F.relu(d_output)
        
        # 5. Dropout
        c_output = F.dropout(c_output, self.dropout, training=self.training)
        d_output = F.dropout(d_output, self.dropout, training=self.training)
        
        return c_output, d_output
    
    def _gat_aggregation(self, source_features, target_features, adjacency, W, a, is_circRNA):
        """
        GAT注意力聚合函数
        
        Args:
            source_features: 源节点特征 [num_source, hidden_dim]
            target_features: 目标节点特征 [num_target, hidden_dim]
            adjacency: 邻接矩阵 [num_target, num_source]
            W: 线性变换矩阵
            a: 注意力参数
            is_circRNA: 是否为circRNA节点（用于区分不同的归一化）
        
        Returns:
            output: 聚合后的目标节点特征 [num_target, hidden_dim]
        """
        num_target = target_features.size(0)
        num_source = source_features.size(0)
        
        # 1. 线性变换
        source_transformed = W(source_features)  # [num_source, hidden_dim]
        target_transformed = W(target_features)  # [num_target, hidden_dim]
        
        # 2. 重塑为多头形式
        source_heads = source_transformed.view(num_source, self.num_heads, self.head_dim)
        target_heads = target_transformed.view(num_target, self.num_heads, self.head_dim)
        
        # 3. 计算注意力分数
        attention_heads = []
        
        for h in range(self.num_heads):
            # 当前头的特征
            source_h = source_heads[:, h, :]  # [num_source, head_dim]
            target_h = target_heads[:, h, :]  # [num_target, head_dim]
            
            # 构建注意力输入：对每个目标节点，与所有源节点计算注意力
            # target_expanded: [num_target, num_source, head_dim]
            target_expanded = target_h.unsqueeze(1).expand(-1, num_source, -1)
            # source_expanded: [num_target, num_source, head_dim]
            source_expanded = source_h.unsqueeze(0).expand(num_target, -1, -1)
            
            # 拼接特征 [num_target, num_source, 2*head_dim]
            concat_features = torch.cat([target_expanded, source_expanded], dim=2)
            
            # 计算注意力分数 [num_target, num_source]
            e = torch.matmul(concat_features, a[h].unsqueeze(0).unsqueeze(0).transpose(1, 2))
            e = e.squeeze(2)  # [num_target, num_source]
            e = self.leakyrelu(e)
            
            # 4. 应用邻接矩阵掩码
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adjacency > 0, e, zero_vec)
            
            # 5. Softmax归一化
            attention = F.softmax(attention, dim=1)  # [num_target, num_source]
            attention = F.dropout(attention, self.dropout, training=self.training)
            
            # 6. 加权聚合
            h_prime = torch.matmul(attention, source_h)  # [num_target, head_dim]
            attention_heads.append(h_prime)
        
        # 7. 合并多头输出
        output = torch.cat(attention_heads, dim=1)  # [num_target, hidden_dim]
        
        return output
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


# ==================== GATv2层 ====================
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.2, alpha=0.2, use_bn=True, activation=True):
        super(GATv2Layer, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_bn = use_bn
        self.activation = activation
        self.head_dim = out_dim // num_heads
        assert out_dim == self.head_dim * num_heads
        # GATv2关键差异：先拼接后变换
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        num_nodes = x.size(0)
        Wh = self.W(x)  # [num_nodes, out_dim]
        Wh = Wh.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        # 为每个注意力头准备特征
        attention_heads = []
        for h in range(self.num_heads):
            head_features = Wh[:, h]  # [num_nodes, head_dim]
            # 准备所有节点对
            src_transformed = head_features.repeat(num_nodes, 1)  # 源节点重复
            dst_transformed = head_features.repeat_interleave(num_nodes, dim=0)  # 目标节点重复
            # 拼接特征
            src_dst_concat = torch.cat([src_transformed, dst_transformed], dim=1)  # [num_nodes*num_nodes, 2*head_dim]
            # GATv2的关键差异：先拼接后做变换
            e = self.leakyrelu(self.a(src_dst_concat)).view(num_nodes, num_nodes)
            # 掩码处理
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            # Softmax
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            # 加权求和
            h_prime = torch.matmul(attention, head_features)  # [num_nodes, head_dim]
            attention_heads.append(h_prime)
        # 合并多头输出
        h_prime = torch.cat(attention_heads, dim=1)  # [num_nodes, out_dim]
        if self.use_bn:
            h_prime = self.bn(h_prime)
        if self.activation:
            h_prime = F.relu(h_prime)
        h_prime = F.dropout(h_prime, self.dropout, training=self.training)
        return h_prime


# ==================== GraphSAGE层 ====================
class GraphSAGELayer(nn.Module):
    """GraphSAGE层"""

    def __init__(self, in_dim, out_dim, dropout=0.2, aggr_method="mean", use_bn=True, activation=True):
        super(GraphSAGELayer, self).__init__()
        self.dropout = dropout
        self.aggr_method = aggr_method
        self.use_bn = use_bn
        self.activation = activation
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        """
        x: 节点特征 [num_nodes, in_dim]
        adj: 邻接矩阵 [num_nodes, num_nodes]
        """
        # 自身特征变换
        self_transformed = self.self_linear(x)
        # 邻居特征聚合
        if self.aggr_method == "mean":
            # 归一化邻接矩阵
            rowsum = adj.sum(1, keepdim=True) + 1e-6
            norm_adj = adj / rowsum
            # 聚合邻居特征
            neigh_feat = torch.mm(norm_adj, x)
        elif self.aggr_method == "max":
            # Max pooling
            dummy = torch.ones_like(adj) * (-9e15)
            adj = torch.where(adj > 0, adj, dummy)
            neigh_feat = torch.max(adj.unsqueeze(-1) * x.unsqueeze(0), dim=1)[0]
        else:  # sum
            neigh_feat = torch.mm(adj, x)
        # 邻居特征变换
        neigh_transformed = self.neigh_linear(neigh_feat)
        # 合并自身和邻居特征
        output = self_transformed + neigh_transformed
        # 归一化和激活
        if self.use_bn:
            output = self.bn(output)
        if self.activation:
            output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        return output


# ==================== 交叉注意力融合模块 ====================
class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.2):
        super(CrossAttentionFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0  # 确保hidden_dim可以被num_heads整除
        # C到D的交叉注意力
        self.c2d_query = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_key = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_value = nn.Linear(hidden_dim, hidden_dim)
        self.c2d_proj = nn.Linear(hidden_dim, hidden_dim)
        # D到C的交叉注意力
        self.d2c_query = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_key = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_value = nn.Linear(hidden_dim, hidden_dim)
        self.d2c_proj = nn.Linear(hidden_dim, hidden_dim)
        # 层归一化
        self.c_norm1 = nn.LayerNorm(hidden_dim)
        self.c_norm2 = nn.LayerNorm(hidden_dim)
        self.d_norm1 = nn.LayerNorm(hidden_dim)
        self.d_norm2 = nn.LayerNorm(hidden_dim)
        # 前馈网络
        self.c_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.d_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, c_features, d_features, cd_adj=None):
        """
        c_features: CircRNA特征 [num_circs, hidden_dim]
        d_features: 药物特征 [num_drugs, hidden_dim]
        cd_adj: CircRNA-Drug邻接矩阵 [num_circs, num_drugs]，可选
        """
        num_circs = c_features.size(0)
        num_drugs = d_features.size(0)
        # 保存原始特征用于残差连接
        c_residual = c_features
        d_residual = d_features
        # 对CircRNA和药物特征进行处理
        c_q = self.c2d_query(c_features).view(num_circs, self.num_heads, self.head_dim)
        d_k = self.c2d_key(d_features).view(num_drugs, self.num_heads, self.head_dim)
        d_v = self.c2d_value(d_features).view(num_drugs, self.num_heads, self.head_dim)
        # 使用einsum计算注意力得分
        attn_scores_c2d = torch.einsum('ihd,jhd->ijh', c_q, d_k) / math.sqrt(self.head_dim)
        # 如果提供了邻接矩阵，使用它进行掩码
        if cd_adj is not None:
            mask = (cd_adj == 0).unsqueeze(2).expand(-1, -1, self.num_heads)
            attn_scores_c2d.masked_fill_(mask, -1e9)
        # 注意力权重 [num_circs, num_drugs, num_heads]
        attn_weights_c2d = F.softmax(attn_scores_c2d, dim=1)
        attn_weights_c2d = self.dropout(attn_weights_c2d)
        # 应用注意力 [num_circs, num_heads, head_dim]
        c_context = torch.einsum('ijh,jhd->ihd', attn_weights_c2d, d_v)
        c_context = c_context.reshape(num_circs, -1)
        c_context = self.c2d_proj(c_context)
        c_context = self.dropout(c_context)
        # 第一个残差连接
        c_features = self.c_norm1(c_residual + c_context)
        # 前馈网络
        c_ffn_out = self.c_ffn(c_features)
        c_features = self.c_norm2(c_features + c_ffn_out)
        # D到C的交叉注意力
        d_q = self.d2c_query(d_features).view(num_drugs, self.num_heads, self.head_dim)
        c_k = self.d2c_key(c_features).view(num_circs, self.num_heads, self.head_dim)
        c_v = self.d2c_value(c_features).view(num_circs, self.num_heads, self.head_dim)
        attn_scores_d2c = torch.einsum('ihd,jhd->ijh', d_q, c_k) / math.sqrt(self.head_dim)
        if cd_adj is not None:
            mask = (cd_adj.t() == 0).unsqueeze(2).expand(-1, -1, self.num_heads)
            attn_scores_d2c.masked_fill_(mask, -1e9)
        attn_weights_d2c = F.softmax(attn_scores_d2c, dim=1)
        attn_weights_d2c = self.dropout(attn_weights_d2c)
        d_context = torch.einsum('ijh,jhd->ihd', attn_weights_d2c, c_v)
        d_context = d_context.reshape(num_drugs, -1)
        d_context = self.d2c_proj(d_context)
        d_context = self.dropout(d_context)
        # 第一个残差连接
        d_features = self.d_norm1(d_residual + d_context)
        # 前馈网络
        d_ffn_out = self.d_ffn(d_features)
        d_features = self.d_norm2(d_features + d_ffn_out)
        return c_features, d_features


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


# ==================== RGCN层 ====================
class RGCNLayer(nn.Module):
    """关系图卷积网络层，专为异质图设计的GCN扩展"""

    def __init__(self, in_dim, out_dim, dropout=0.2, use_bn=True, activation=True):
        super(RGCNLayer, self).__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = activation
        # 对circRNA到drug的变换
        self.c2d_transform = nn.Linear(in_dim, out_dim)
        # 对drug到circRNA的变换
        self.d2c_transform = nn.Linear(in_dim, out_dim)
        # 自环变换
        self.self_transform = nn.Linear(in_dim, out_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, c_features, d_features, cd_adj):
        """
        c_features: circRNA特征 [num_circs, in_dim]
        d_features: 药物特征 [num_drugs, in_dim]
        cd_adj: circRNA-drug邻接矩阵 [num_circs, num_drugs]
        """
        num_circs = c_features.size(0)
        num_drugs = d_features.size(0)
        # 自环变换
        c_self = self.self_transform(c_features)
        d_self = self.self_transform(d_features)
        # circRNA到drug的信息传递
        c_rowsum = cd_adj.sum(1, keepdim=True) + 1e-6
        c_norm_adj = cd_adj / c_rowsum
        c_from_d = torch.mm(c_norm_adj, d_features)
        c_from_d = self.c2d_transform(c_from_d)
        # drug到circRNA的信息传递
        d_rowsum = cd_adj.t().sum(1, keepdim=True) + 1e-6
        d_norm_adj = cd_adj.t() / d_rowsum
        d_from_c = torch.mm(d_norm_adj, c_features)
        d_from_c = self.d2c_transform(d_from_c)
        # 组合信息
        c_output = c_self + c_from_d
        d_output = d_self + d_from_c
        # 归一化和激活
        if self.use_bn:
            c_output = self.bn(c_output)
            d_output = self.bn(d_output)
        if self.activation:
            c_output = F.relu(c_output)
            d_output = F.relu(d_output)
        c_output = F.dropout(c_output, self.dropout, training=self.training)
        d_output = F.dropout(d_output, self.dropout, training=self.training)
        return c_output, d_output


# ==================== SimpleHGN ====================
class SimpleHGN(nn.Module):
    """简化版异质图注意力网络，平衡性能与计算效率"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.2):
        super(SimpleHGN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        assert hidden_dim % num_heads == 0
        # 注意力计算
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = nn.Parameter(torch.Tensor(1, num_heads, 1))
        # 更新函数
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 初始化
        nn.init.xavier_uniform_(self.edge_attn)

    def forward(self, c_features, d_features, cd_adj):
        """
        c_features: circRNA特征 [num_circs, hidden_dim]
        d_features: 药物特征 [num_drugs, hidden_dim]
        cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        """
        num_circs = c_features.size(0)
        num_drugs = d_features.size(0)
        # 拼接特征，便于统一处理
        x = torch.cat([c_features, d_features], dim=0)  # [num_circs+num_drugs, hidden_dim]
        # 构建完整的邻接矩阵，包含c-d和d-c关系
        full_adj = torch.zeros(num_circs + num_drugs, num_circs + num_drugs, device=c_features.device)
        full_adj[:num_circs, num_circs:] = cd_adj
        full_adj[num_circs:, :num_circs] = cd_adj.t()
        # 计算多头注意力
        q = self.query(x).view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = self.key(x).view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        v = self.value(x).view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        # 注意力分数
        attn_scores = torch.einsum('ihd,jhd->ijh', q, k) / math.sqrt(self.head_dim)  # [num_nodes, num_nodes, num_heads]
        # 加入边信息
        edge_importance = self.edge_attn.expand(1, self.num_heads, 1)
        # 掩码无连接的边
        mask = (full_adj == 0).unsqueeze(-1).expand(-1, -1, self.num_heads)
        attn_scores.masked_fill_(mask, -1e9)
        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # [num_nodes, num_nodes, num_heads]
        attn_weights = F.dropout(attn_weights, self.dropout, training=self.training)
        # 聚合信息
        output = torch.zeros_like(x)
        for h in range(self.num_heads):
            h_output = torch.mm(attn_weights[:, :, h], v[:, h, :])  # [num_nodes, head_dim]
            output_idx = h * self.head_dim
            output[:, output_idx:output_idx + self.head_dim] = h_output
        # 更新节点表示，结合原始特征和聚合特征
        output = self.update(torch.cat([x, output], dim=1))
        # 分离回circRNA和drug特征
        c_output = output[:num_circs]
        d_output = output[num_circs:]
        return c_output, d_output


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


# ==================== 加权和融合 ====================
class WeightedSumFusion(nn.Module):
    """加权和融合：使用可学习的权重融合不同来源的特征"""

    def __init__(self, hidden_dim):
        super(WeightedSumFusion, self).__init__()
        # 可学习的融合权重
        self.cs_weight = nn.Parameter(torch.ones(1))
        self.ds_weight = nn.Parameter(torch.ones(1))
        self.cd_c_weight = nn.Parameter(torch.ones(1))
        self.cd_d_weight = nn.Parameter(torch.ones(1))
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
        # 获取归一化的融合权重 - circRNA
        c_weights = F.softmax(torch.stack([self.cs_weight, self.cd_c_weight]), dim=0)
        cs_norm_weight, cd_c_norm_weight = c_weights[0], c_weights[1]
        # 获取归一化的融合权重 - 药物
        d_weights = F.softmax(torch.stack([self.ds_weight, self.cd_d_weight]), dim=0)
        ds_norm_weight, cd_d_norm_weight = d_weights[0], d_weights[1]
        # 加权融合
        c_fused = cs_features * cs_norm_weight + cd_c_features * cd_c_norm_weight
        d_fused = ds_features * ds_norm_weight + cd_d_features * cd_d_norm_weight
        # 输出投影
        c_output = self.c_output_projection(c_fused)
        d_output = self.d_output_projection(d_fused)
        # 层归一化
        c_output = self.c_norm(c_output)
        d_output = self.d_norm(d_output)
        return c_output, d_output


# ==================== 门控机制融合 ====================
class GatedFusion(nn.Module):
    """门控机制融合：使用门控单元动态控制不同特征的重要性"""

    def __init__(self, hidden_dim):
        super(GatedFusion, self).__init__()
        # circRNA特征门控
        self.c_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        # 药物特征门控
        self.d_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
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
        # 计算门控权重 - circRNA
        c_concat = torch.cat([cs_features, cd_c_features], dim=1)
        c_gate_weights = self.c_gate(c_concat)
        # 计算门控权重 - 药物
        d_concat = torch.cat([ds_features, cd_d_features], dim=1)
        d_gate_weights = self.d_gate(d_concat)
        # 门控融合
        c_fused = cs_features * c_gate_weights + cd_c_features * (1 - c_gate_weights)
        d_fused = ds_features * d_gate_weights + cd_d_features * (1 - d_gate_weights)
        # 输出投影
        c_output = self.c_output_projection(c_fused)
        d_output = self.d_output_projection(d_fused)
        # 层归一化
        c_output = self.c_norm(c_output)
        d_output = self.d_norm(d_output)
        return c_output, d_output


# ==================== MLP预测模块 ====================
class MLPPredictor(nn.Module):
    """MLP预测模块"""

    def __init__(self, hidden_dim, dropout=0.2):
        super(MLPPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, c_features, d_features):
        # 拼接特征
        combined = torch.cat([c_features, d_features], dim=1)
        # 预测
        output = self.predictor(combined)
        return output


# ==================== 点积预测模块 ====================
class DotProductPredictor(nn.Module):
    """点积预测模块"""

    def __init__(self, hidden_dim, dropout=0.2):
        super(DotProductPredictor, self).__init__()
        # 特征变换
        self.c_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.d_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, c_features, d_features):
        # 特征变换
        c_transformed = self.c_transform(c_features)
        d_transformed = self.d_transform(d_features)
        # 计算点积
        dot_product = torch.sum(c_transformed * d_transformed, dim=1, keepdim=True)
        # sigmoid激活
        output = torch.sigmoid(dot_product)
        return output


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
    """同质图处理模块工厂，支持多种图神经网络模型"""

    def __init__(self, in_dim, hidden_dim, n_layers=2, model_type="gat",
                 dropout=0.2, num_heads=8, aggr_method="mean"):
        super(HomoGraphEncoder, self).__init__()
        self.model_type = model_type
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            if model_type == "gcn":
                self.layers.append(
                    GCNLayer(in_dim=input_dim, out_dim=hidden_dim,
                             dropout=dropout, use_bn=True)
                )
            elif model_type == "gat":
                self.layers.append(
                    GATLayer(in_dim=input_dim, out_dim=hidden_dim,  # 128——》64
                             num_heads=num_heads, dropout=dropout)
                )
            elif model_type == "gatv2":
                self.layers.append(
                    GATv2Layer(in_dim=input_dim, out_dim=hidden_dim,
                               num_heads=num_heads, dropout=dropout)
                )
            elif model_type == "sage":
                self.layers.append(
                    GraphSAGELayer(in_dim=input_dim, out_dim=hidden_dim,
                                   dropout=dropout, aggr_method=aggr_method)
                )
            else:
                raise ValueError(f"未知的模型类型: {model_type}")

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
    """异质图处理模块工厂，支持多种图神经网络模型"""

    def __init__(self, hidden_dim, n_layers=2, model_type="BipartiteGraphTransformer",
                 dropout=0.2, num_heads=8, use_structure_pe=False):
        super(HeteroGraphEncoder, self).__init__()
        self.model_type = model_type
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if model_type == "BipartiteGraphTransformer":
                self.layers.append(
                    BipartiteGraphTransformer(hidden_dim=hidden_dim,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              use_structure_pe=use_structure_pe)
                )
            elif model_type == "bipartite_gat":
                self.layers.append(
                    BipartiteGAT(hidden_dim=hidden_dim,
                                num_heads=num_heads,
                                dropout=dropout)
                )
            elif model_type == "cross_attention":
                self.layers.append(
                    CrossAttentionFusion(hidden_dim=hidden_dim,
                                         num_heads=num_heads,
                                         dropout=dropout)
                )
            elif model_type == "bipartite_sage":
                self.layers.append(
                    BipartiteGraphSAGE(hidden_dim=hidden_dim,
                                    dropout=dropout,
                                    aggr_method="mean")
                )
            elif model_type == "rgcn":
                self.layers.append(
                    RGCNLayer(in_dim=hidden_dim,
                              out_dim=hidden_dim,
                              dropout=dropout)
                )
            elif model_type == "hgn":
                self.layers.append(
                    SimpleHGN(hidden_dim=hidden_dim,
                              num_heads=num_heads,
                              dropout=dropout)
                )
            else:
                raise ValueError(f"未知的异质图模型类型: {model_type}")

    def forward(self, c_features, d_features, cd_adj):
        """
        c_features: circRNA特征 [num_circs, hidden_dim]
        d_features: 药物特征 [num_drugs, hidden_dim]
        cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        """
        for layer in self.layers:
            c_features, d_features = layer(c_features, d_features, cd_adj)
        return c_features, d_features

# ==================== 二分图GraphSAGE ====================
class BipartiteGraphSAGE(nn.Module):
    """专门用于二分图的GraphSAGE层，不创新，就是标准的SAGE应用到二分图"""
    
    def __init__(self, hidden_dim, dropout=0.2, aggr_method="mean", use_bn=True, activation=True):
        super(BipartiteGraphSAGE, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.aggr_method = aggr_method
        self.use_bn = use_bn
        self.activation = activation
        
        # CircRNA节点的SAGE变换
        # 自身特征变换
        self.c_self_linear = nn.Linear(hidden_dim, hidden_dim)
        # 邻居特征变换（来自Drug节点）
        self.c_neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Drug节点的SAGE变换
        # 自身特征变换
        self.d_self_linear = nn.Linear(hidden_dim, hidden_dim)
        # 邻居特征变换（来自CircRNA节点）
        self.d_neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # 批归一化
        if use_bn:
            self.c_bn = nn.BatchNorm1d(hidden_dim)
            self.d_bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, c_features, d_features, cd_adj):
        """
        Args:
            c_features: circRNA特征 [num_circs, hidden_dim]
            d_features: 药物特征 [num_drugs, hidden_dim]
            cd_adj: CD邻接矩阵 [num_circs, num_drugs]
        
        Returns:
            c_output: 更新后的circRNA特征 [num_circs, hidden_dim]
            d_output: 更新后的药物特征 [num_drugs, hidden_dim]
        """
        # 1. CircRNA节点聚合Drug邻居信息
        c_neigh_feat = self._aggregate_neighbors(
            source_features=d_features,  # 来源：Drug节点
            adjacency=cd_adj,  # [num_circs, num_drugs]
            aggr_method=self.aggr_method
        )
        
        # 2. Drug节点聚合CircRNA邻居信息
        d_neigh_feat = self._aggregate_neighbors(
            source_features=c_features,  # 来源：CircRNA节点
            adjacency=cd_adj.t(),  # [num_drugs, num_circs] (转置)
            aggr_method=self.aggr_method
        )
        
        # 3. SAGE更新：自身特征 + 邻居特征
        # CircRNA更新
        c_self_transformed = self.c_self_linear(c_features)
        c_neigh_transformed = self.c_neigh_linear(c_neigh_feat)
        c_output = c_self_transformed + c_neigh_transformed
        
        # Drug更新
        d_self_transformed = self.d_self_linear(d_features)
        d_neigh_transformed = self.d_neigh_linear(d_neigh_feat)
        d_output = d_self_transformed + d_neigh_transformed
        
        # 4. 批归一化
        if self.use_bn:
            c_output = self.c_bn(c_output)
            d_output = self.d_bn(d_output)
        
        # 5. 激活函数
        if self.activation:
            c_output = F.relu(c_output)
            d_output = F.relu(d_output)
        
        # 6. Dropout
        c_output = F.dropout(c_output, self.dropout, training=self.training)
        d_output = F.dropout(d_output, self.dropout, training=self.training)
        
        return c_output, d_output
    
    def _aggregate_neighbors(self, source_features, adjacency, aggr_method):
        """
        聚合邻居特征
        
        Args:
            source_features: 源节点特征 [num_source, hidden_dim]
            adjacency: 邻接矩阵 [num_target, num_source]
            aggr_method: 聚合方法 ("mean", "sum", "max")
        
        Returns:
            aggregated_features: 聚合后的特征 [num_target, hidden_dim]
        """
        if aggr_method == "mean":
            # 平均聚合
            # 计算每个目标节点的邻居数量，避免除以0
            rowsum = adjacency.sum(1, keepdim=True) + 1e-6
            norm_adj = adjacency / rowsum
            aggregated = torch.mm(norm_adj, source_features)
            
        elif aggr_method == "sum":
            # 求和聚合
            aggregated = torch.mm(adjacency, source_features)
            
        elif aggr_method == "max":
            # 最大值聚合
            num_target, num_source = adjacency.shape
            hidden_dim = source_features.size(1)
            
            # 创建掩码矩阵，无连接处设为很小的值
            mask_value = -1e9
            masked_adj = torch.where(adjacency > 0, adjacency, 
                                   torch.full_like(adjacency, mask_value))
            
            # 扩展维度进行max pooling
            # masked_adj: [num_target, num_source, 1]
            # source_features: [1, num_source, hidden_dim]
            masked_adj_expanded = masked_adj.unsqueeze(-1)
            source_expanded = source_features.unsqueeze(0)
            
            # 广播相乘然后取最大值
            weighted_features = masked_adj_expanded * source_expanded
            aggregated, _ = torch.max(weighted_features, dim=1)  # [num_target, hidden_dim]
            
        else:
            raise ValueError(f"不支持的聚合方法: {aggr_method}")
        
        return aggregated

# ==================== CircRNA-Drug 图转换器 ====================
class CDGT(nn.Module):
    """CircRNA-Drug 图转换器 (CDGT)
    一个灵活的框架，用于预测 circRNA-Drug 关联，支持多种图神经网络模型和融合策略
    """

    def __init__(self,
                 c_feature_dim=640,  # circRNA特征维度
                 d_feature_dim=768,  # 药物特征维度
                 projection_dim=256,  # 特征投影模块的输出维度,同时也是异构图的输入输出维度
                 homo_hidden_dim=64,  # 同质图处理模块的隐藏层维度
                 homo_model="gat",  # 同质图处理模型: gat, gcn, gatv2, sage
                 hetero_model="BipartiteGraphTransformer",
                 # 异质图处理模型: BipartiteGraphTransformer, cross_attention, rgcn, hgn
                 fusion_method="uncertainty",  # 融合方法: uncertainty, cross_attention, WeightedSumFusion, GatedFusion
                 num_layers=2,  # 图处理器层数
                 homo_num_layers=2,  # 同质图处理器层数
                 hetero_num_layers=2,  # 异质图处理器层数
                 num_heads=8,  # 图处理器中的注意力头数
                 homo_num_heads=8,  # 同质图处理器中的注意力头数
                 hetero_num_heads=8,  # 异质图处理器中的注意力头数
                 dropout=0.2,  # Dropout比率
                 prediction_method="mlp",  # 预测方法: mlp, dot_product, bilinear
                 use_structure_pe=True,  # 是否使用结构感知位置编码
                 use_cross_attention_projection=False,  # 是否在特征投影中使用交叉注意力机制
                 cross_attention_heads=4,
                 sage_aggr="mean",  # GraphSAGE聚合方法: mean, sum, max
                 use_homo_features=True,
                 use_hetero_features=True
                 ):
        super(CDGT, self).__init__()
        self.homo_model = homo_model
        self.hetero_model = hetero_model
        self.fusion_method = fusion_method
        self.prediction_method = prediction_method
        # 1. 特征投影模块
        self.feature_projection = FeatureProjection(
            c_feature_dim=c_feature_dim,
            d_feature_dim=d_feature_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        # 2. 同质图处理模块
        self.cs_encoder = HomoGraphEncoder(
            in_dim=projection_dim,
            hidden_dim=homo_hidden_dim,
            n_layers=homo_num_layers,
            model_type=homo_model,
            dropout=dropout,
            num_heads=homo_num_heads,
            aggr_method=sage_aggr
        )
        self.ds_encoder = HomoGraphEncoder(
            in_dim=projection_dim,
            hidden_dim=homo_hidden_dim,
            n_layers=homo_num_layers,
            model_type=homo_model,
            dropout=dropout,
            num_heads=homo_num_heads,
            aggr_method=sage_aggr
        )
        # 3. 异质图处理模块
        self.cd_encoder = HeteroGraphEncoder(  # 128——》64……
            hidden_dim=projection_dim,
            n_layers=hetero_num_layers,
            model_type=hetero_model,
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
        # 4. 特征融合模块
        if fusion_method == "uncertainty":
            self.fusion = UncertaintyGuidedFusion(homo_hidden_dim,
                                                  use_homo_features=use_homo_features,
                                                  use_hetero_features=use_hetero_features)
        elif fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(homo_hidden_dim, num_heads, dropout)
        elif fusion_method == "WeightedSumFusion":
            self.fusion = WeightedSumFusion(homo_hidden_dim)
        elif fusion_method == "GatedFusion":
            self.fusion = GatedFusion(homo_hidden_dim)
        else:
            raise ValueError(f"未知的融合方法: {fusion_method}")
        # 5. 预测模块
        if prediction_method == "mlp":
            self.predictor = MLPPredictor(homo_hidden_dim, dropout)
        elif prediction_method == "dot_product":
            self.predictor = DotProductPredictor(homo_hidden_dim, dropout)
        elif prediction_method == "bilinear":
            self.predictor = BilinearPredictor(homo_hidden_dim, dropout)
        else:
            raise ValueError(f"未知的预测方法: {prediction_method}")

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

        # 4. 特征融合
        if self.fusion_method in ["uncertainty", "WeightedSumFusion", "GatedFusion"]:
            c_fused, d_fused = self.fusion(cs_features, ds_features, cd_c_features, cd_d_features)
        else:  # cross_attention
            c_fused, d_fused = self.fusion(cs_features, ds_features, cd_adj)
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


# ==================== 命令行参数更新 ====================
def update_train_arguments(parser):
    """更新train.py中的命令行参数解析器，支持更全面的调参实验"""
    # 基本参数
    parser.add_argument('--model', type=str, default='CDGT',
                        choices=['CDGT', 'MLP'],
                        help='使用的模型 (默认: CDGT)')
    parser.add_argument('--dataset_type', type=str, default='Resistance',
                        choices=['Resistance', 'Target', 'Wang'],
                        help='数据集类型: Resistance、Target或Wang (默认: Resistance)')
    # 通用训练参数
    parser.add_argument('--n_epochs', type=int, default=200, help='训练轮数 (默认: 200)')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率 (默认: 0.0002)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数 (默认: 5)')
    parser.add_argument('--seed', type=int, default=51, help='随机种子 (默认: 51)')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (默认: 20)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率 (默认: 0.2)')
    # 图层参数
    parser.add_argument('--homo_num_layers', type=int, default=2, help='同质图处理器层数 (默认: 2)')
    parser.add_argument('--hetero_num_layers', type=int, default=2, help='异质图处理器层数 (默认: 2)')
    # 注意力头参数
    parser.add_argument('--homo_num_heads', type=int, default=4, help='同质图处理器中的注意力头数 (默认: 4)')
    parser.add_argument('--hetero_num_heads', type=int, default=32, help='异质图处理器中的注意力头数 (默认: 32)')
    parser.add_argument('--prediction_method', type=str, default='bilinear',
                        choices=['mlp', 'dot_product', 'bilinear'],
                        help='预测方法 (默认: bilinear)')
    parser.add_argument('--circ_threshold', type=float, default=0.4034, help='circRNA相似性阈值 (默认: 0.4034)')
    parser.add_argument('--drug_threshold', type=float, default=0.06, help='药物相似性阈值 (默认: 0.06)')
    parser.add_argument('--use_structure_pe', action='store_false', default=True,
                        help='使用结构感知位置编码 (默认: True)')

    # 添加消融实验参数
    parser.add_argument('--use_homo_features', action='store_false', default=True,
                        help='是否使用同质图特征 (默认: True)')
    parser.add_argument('--use_hetero_features', action='store_false', default=True,
                        help='是否使用异质图特征 (默认: True)')

    # 隐藏层参数
    parser.add_argument('--projection_dim', type=int, default=512,
                        help='特征投影模块输出维度 (默认: 512)')
    parser.add_argument('--homo_hidden_dim', type=int, default=128,
                        help='同质图处理模块隐藏层维度 (默认: 128)')

    # 图模型配置参数
    parser.add_argument('--homo_model', type=str, default='gat',
                        choices=['gcn', 'gat', 'gatv2', 'sage'],
                        help='同质图处理模型 (默认: gat)')
    parser.add_argument('--hetero_model', type=str, default='BipartiteGraphTransformer',
                        choices=['BipartiteGraphTransformer',  'rgcn', 'bipartite_gat', 'bipartite_sage'],
                        help='异质图处理模型')
    parser.add_argument('--fusion_method', type=str, default='uncertainty',
                        choices=['uncertainty', 'cross_attention', 'WeightedSumFusion', 'GatedFusion'])
    parser.add_argument('--sage_aggr', type=str, default='mean',
                        choices=['mean', 'sum', 'max'],
                        help='GraphSAGE聚合方法 (默认: sum)')
    # MLP特定参数
    parser.add_argument('--hidden_dims', type=str, default='256,128,64',
                        help='MLP隐藏层维度, 逗号分隔 (默认: 256,128,64)')
    return parser