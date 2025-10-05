import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.model_selection import KFold
import time
import re
import random
import warnings
from scipy.interpolate import interp1d
import numpy as np
warnings.filterwarnings("ignore")


def visualize_heterogeneous_graph(cd_adj, labels=None, output_file='heterogeneous_graph.jpg'):
    """可视化异构图结构"""
    import networkx as nx

    # 创建二分图
    G = nx.Graph()

    num_circs, num_drugs = cd_adj.shape

    # 添加节点
    for i in range(num_circs):
        G.add_node(f'c{i}', bipartite=0, label=f'circRNA {i}')

    for j in range(num_drugs):
        G.add_node(f'd{j}', bipartite=1, label=f'Drug {j}')

    # 添加边
    for i in range(num_circs):
        for j in range(num_drugs):
            if cd_adj[i, j] > 0:
                weight = 1 if labels is None else (2 if labels.get((i, j), 0) == 1 else 1)
                G.add_edge(f'c{i}', f'd{j}', weight=weight)

    # 提取节点集
    circ_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    drug_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    # 设置节点位置
    pos = {}
    pos.update((node, (1, i)) for i, node in enumerate(circ_nodes))
    pos.update((node, (2, i)) for i, node in enumerate(drug_nodes))

    # 绘制图
    plt.figure(figsize=(10, 8))

    # 画节点
    nx.draw_networkx_nodes(G, pos, nodelist=circ_nodes, node_color='skyblue', node_size=100, label='circRNA')
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='lightgreen', node_size=100, label='Drug')

    # 画边
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]

    if labels is not None:
        pos_edges = [(u, v) for u, v, d in edges if d['weight'] == 2]
        neg_edges = [(u, v) for u, v, d in edges if d['weight'] == 1]
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=1, edge_color='red', alpha=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=0.5, edge_color='gray', alpha=0.5)
    else:
        nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.5)

    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return G


def visualize_model_architecture(model, output_file='model_architecture.txt'):
    """保存模型架构为文本文件"""
    with open(output_file, 'w') as f:
        # 获取模型架构
        model_str = str(model)
        f.write(model_str)

        # 计算总参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"\n\n总参数数量: {total_params}\n")
        f.write(f"可训练参数数量: {trainable_params}\n")

        # 分解各模块参数
        f.write("\n各模块参数分布:\n")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            f.write(f"{name}: {params} ({params / total_params * 100:.2f}%)\n")


def compare_model_configs(result_files, output_dir="./comparison"):
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有配置的性能数据
    all_metrics = []
    model_names = []

    # 解析结果文件
    for result_file in result_files:
        # 解析文件名获取模型配置
        model_name = os.path.basename(os.path.dirname(result_file))
        model_names.append(model_name)

        # 读取平均结果
        with open(result_file, 'r') as f:
            content = f.read()

        # 解析平均指标
        avg_section = content.split("平均结果")[1].split("运行时间")[0]
        metrics = {}

        for line in avg_section.strip().split('\n'):
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    metric_name = parts[0]
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value

        metrics['Model'] = model_name
        all_metrics.append(metrics)

    # 转换为DataFrame
    df = pd.DataFrame(all_metrics)

    # 保存对比结果
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # 绘制对比图表
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROCAUC', 'PRAUC']

    bar_width = 0.8 / len(model_names)
    index = np.arange(len(metrics_to_plot))

    for i, model in enumerate(model_names):
        values = [df[df['Model'] == model][metric].values[0] for metric in metrics_to_plot]
        plt.bar(index + i * bar_width, values, bar_width, label=model)

    plt.xticks(index + bar_width * (len(model_names) - 1) / 2, metrics_to_plot)
    plt.ylabel('Score')
    plt.title('Performance Comparison of Different Model Configurations')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'model_comparison.jpg'))
    plt.close()

    return df


def load_data(data_path='./datasets/'):
    """加载所有必要的数据矩阵"""
    print("正在加载数据...")

    # 加载五大矩阵
    CS = np.loadtxt(os.path.join(data_path, 'CS.csv'), delimiter=',')
    DS = np.loadtxt(os.path.join(data_path, 'DS.csv'), delimiter=',')
    CF = np.loadtxt(os.path.join(data_path, 'CF.csv'), delimiter=',')
    DF = np.loadtxt(os.path.join(data_path, 'DF.csv'), delimiter=',')
    CD = np.loadtxt(os.path.join(data_path, 'CD.csv'), delimiter=',')

    # 加载样本
    edges = pd.read_csv(os.path.join(data_path, 'samples/edge.csv'))
    labels = pd.read_csv(os.path.join(data_path, 'samples/labels.csv'))

    # 打印数据形状
    print(f"CS 形状: {CS.shape}")
    print(f"DS 形状: {DS.shape}")
    print(f"CF 形状: {CF.shape}")
    print(f"DF 形状: {DF.shape}")
    print(f"CD 形状: {CD.shape}")
    print(f"边样本数量: {len(edges)}")
    print(f"标签样本数量: {len(labels)}")

    return CS, DS, CF, DF, CD, edges, labels


def plot_metrics(result_file, output_dir):
    """绘制评价指标图"""
    print("正在绘制评价指标图...")

    # 读取结果文件
    with open(result_file, 'r') as f:
        content = f.read()

    # 匹配评价指标
    pattern = r"Fold\s+(\d+).*?Accuracy\s+([\d\.]+)\s+Precision\s+([\d\.]+)\s+Recall\s+([\d\.]+)\s+F1\s+([\d\.]+)\s+ROC\s*AUC\s+([\d\.]+)\s+PR\s*AUC\s+([\d\.]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    # 备用模式匹配
    if not matches:
        pattern_alt = r"Fold\s+(\d+)[^0-9.]*?([0-9.]+)[^0-9.]*?([0-9.]+)[^0-9.]*?([0-9.]+)[^0-9.]*?([0-9.]+)[^0-9.]*?([0-9.]+)[^0-9.]*?([0-9.]+)"
        matches = re.findall(pattern_alt, content, re.DOTALL)

    # 提取表格数据
    if not matches:
        table_pattern = r"\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+"
        table_rows = re.findall(table_pattern, content)

        if table_rows:
            matches = []
            for i, row in enumerate(table_rows):
                values = re.findall(r"\d+\.\d+", row)
                if len(values) >= 6:
                    matches.append([str(i + 1)] + values[:6])

    if not matches:
        print("未找到评价指标数据")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "无法解析评价指标数据\n请检查结果文件格式",
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'metrics.jpg'))
        plt.close()
        return

    # 提取数据
    folds = [int(m[0]) for m in matches]
    accuracy = [float(m[1]) for m in matches]
    precision = [float(m[2]) for m in matches]
    recall = [float(m[3]) for m in matches]
    f1 = [float(m[4]) for m in matches]
    rocauc = [float(m[5]) for m in matches]
    prauc = [float(m[6]) for m in matches]

    print(f"找到 {len(folds)} 折的评价指标数据")

    # 绘制图像
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(folds, accuracy, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(folds, precision, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(folds, recall, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(folds, f1, 'o-', label='F1')
    plt.title('F1 Score')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(folds, rocauc, 'o-', label='ROCAUC')
    plt.title('ROC AUC')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(folds, prauc, 'o-', label='PRAUC')
    plt.title('PR AUC')
    plt.xlabel('Fold')
    plt.ylim([0, 1.05])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.jpg'))
    plt.close()

    # 创建综合指标图
    plt.figure(figsize=(10, 6))
    width = 0.15
    indices = np.arange(len(folds))

    plt.bar(indices - 2 * width, accuracy, width, label='Accuracy')
    plt.bar(indices - width, precision, width, label='Precision')
    plt.bar(indices, recall, width, label='Recall')
    plt.bar(indices + width, f1, width, label='F1')
    plt.bar(indices + 2 * width, rocauc, width, label='ROC AUC')

    plt.xticks(indices, [f'Fold {i}' for i in folds])
    plt.ylim(0, 1.05)
    plt.title('评价指标比较')
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.jpg'))
    plt.close()

    print(f"评价指标图已保存至 {os.path.join(output_dir, 'metrics.jpg')} 和 {os.path.join(output_dir, 'metrics_comparison.jpg')}")


def normalize_features(features, method='min-max'):
    """归一化特征"""
    # 检查输入类型
    is_torch = isinstance(features, torch.Tensor)

    if not is_torch:
        features = torch.FloatTensor(features)

    if method == 'min-max':
        # 最小-最大归一化
        min_val = features.min(0, keepdim=True)[0]
        max_val = features.max(0, keepdim=True)[0]
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
    elif method == 'z-score':
        # Z-分数归一化
        mean = features.mean(0, keepdim=True)
        std = features.std(0, keepdim=True)
        normalized = (features - mean) / (std + 1e-8)
    else:
        normalized = features

    # 返回与输入相同类型的结果
    if is_torch:
        return normalized
    else:
        return normalized.numpy()


def analyze_graph_statistics(CS, DS, CD, cs_threshold=0.4034, ds_threshold=0.06):
    """分析三个图的统计特性"""
    print("分析图统计特性...")

    # 创建邻接矩阵
    cs_adj = CS > cs_threshold
    ds_adj = DS > ds_threshold
    cd_adj = CD > 0

    # 计算节点数
    num_circs = CS.shape[0]
    num_drugs = DS.shape[0]

    # 计算边数
    cs_edges = np.sum(cs_adj) // 2  # 无向图计算一半
    ds_edges = np.sum(ds_adj) // 2
    cd_edges = np.sum(cd_adj)

    # 计算平均度
    cs_avg_degree = cs_edges * 2 / num_circs
    ds_avg_degree = ds_edges * 2 / num_drugs
    cd_avg_degree_c = cd_edges / num_circs
    cd_avg_degree_d = cd_edges / num_drugs

    # 计算稀疏度
    cs_density = cs_edges / (num_circs * (num_circs - 1) / 2)
    ds_density = ds_edges / (num_drugs * (num_drugs - 1) / 2)
    cd_density = cd_edges / (num_circs * num_drugs)

    # 打印统计信息
    print(f"circRNA节点数: {num_circs}")
    print(f"药物节点数: {num_drugs}")
    print(f"CS图边数: {cs_edges}")
    print(f"DS图边数: {ds_edges}")
    print(f"CD图边数: {cd_edges}")
    print(f"CS图平均度: {cs_avg_degree:.2f}")
    print(f"DS图平均度: {ds_avg_degree:.2f}")
    print(f"CD图中circRNA节点平均度: {cd_avg_degree_c:.2f}")
    print(f"CD图中药物节点平均度: {cd_avg_degree_d:.2f}")
    print(f"CS图密度: {cs_density:.6f}")
    print(f"DS图密度: {ds_density:.6f}")
    print(f"CD图密度: {cd_density:.6f}")

    # 返回统计信息字典
    return {
        'num_circs': num_circs,
        'num_drugs': num_drugs,
        'cs_edges': cs_edges,
        'ds_edges': ds_edges,
        'cd_edges': cd_edges,
        'cs_avg_degree': cs_avg_degree,
        'ds_avg_degree': ds_avg_degree,
        'cd_avg_degree_c': cd_avg_degree_c,
        'cd_avg_degree_d': cd_avg_degree_d,
        'cs_density': cs_density,
        'ds_density': ds_density,
        'cd_density': cd_density
    }


def visualize_adjacency_matrices(CS, DS, CD, cs_threshold=0.4034, ds_threshold=0.06, output_dir="./"):
    """可视化三个邻接矩阵"""
    # 创建二值化邻接矩阵
    cs_adj = (CS > cs_threshold).astype(np.float32)
    ds_adj = (DS > ds_threshold).astype(np.float32)
    cd_adj = CD.astype(np.float32)

    # 设置图像大小
    plt.figure(figsize=(18, 6))

    # 绘制CS矩阵
    plt.subplot(1, 3, 1)
    plt.imshow(cs_adj, cmap='Blues')
    plt.title(f'CS Matrix (threshold={cs_threshold})')
    plt.colorbar()

    # 绘制DS矩阵
    plt.subplot(1, 3, 2)
    plt.imshow(ds_adj, cmap='Blues')
    plt.title(f'DS Matrix (threshold={ds_threshold})')
    plt.colorbar()

    # 绘制CD矩阵
    plt.subplot(1, 3, 3)
    plt.imshow(cd_adj, cmap='Blues')
    plt.title('CD Matrix')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjacency_matrices.jpg'))
    plt.close()

    print(f"邻接矩阵可视化已保存至 {os.path.join(output_dir, 'adjacency_matrices.jpg')}")


def visualize_graph_structure(CS, DS, CD, cs_threshold=0.4034, ds_threshold=0.06, output_dir="./"):
    """可视化三个图的结构特性"""
    # 创建二值化邻接矩阵
    cs_adj = (CS > cs_threshold).astype(np.float32)
    ds_adj = (DS > ds_threshold).astype(np.float32)
    cd_adj = CD.astype(np.float32)

    # 计算CS图的聚类系数
    def calculate_clustering_coefficient(adj):
        n = adj.shape[0]
        clustering_coeffs = np.zeros(n)

        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) < 2:
                continue

            possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
            actual_connections = 0

            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj[neighbors[j], neighbors[k]] > 0:
                        actual_connections += 1

            if possible_connections > 0:
                clustering_coeffs[i] = actual_connections / possible_connections

        return clustering_coeffs

    # 计算聚类系数
    cs_clustering = calculate_clustering_coefficient(cs_adj)
    ds_clustering = calculate_clustering_coefficient(ds_adj)

    # 计算度
    cs_degree = np.sum(cs_adj, axis=1)
    ds_degree = np.sum(ds_adj, axis=1)
    cd_c_degree = np.sum(cd_adj, axis=1)
    cd_d_degree = np.sum(cd_adj, axis=0)

    # 绘制聚类系数与度的关系
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(cs_degree, cs_clustering, alpha=0.6, s=10)
    plt.title('CS图: 聚类系数与度的关系')
    plt.xlabel('节点度')
    plt.ylabel('聚类系数')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(ds_degree, ds_clustering, alpha=0.6, s=10, color='green')
    plt.title('DS图: 聚类系数与度的关系')
    plt.xlabel('节点度')
    plt.ylabel('聚类系数')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_vs_degree.jpg'))
    plt.close()

    # 绘制路径长度分布
    def calculate_approx_path_length(adj):
        """使用Floyd-Warshall算法近似计算平均路径长度"""
        n = adj.shape[0]
        if n > 1000:  # 对于大型图，仅使用子图进行估计
            indices = np.random.choice(n, size=1000, replace=False)
            adj_sub = adj[np.ix_(indices, indices)]
        else:
            adj_sub = adj

        n_sub = adj_sub.shape[0]
        dist = np.zeros((n_sub, n_sub))

        # 初始化距离矩阵
        for i in range(n_sub):
            for j in range(n_sub):
                if i == j:
                    dist[i, j] = 0
                elif adj_sub[i, j] > 0:
                    dist[i, j] = 1
                else:
                    dist[i, j] = float('inf')

        # Floyd-Warshall算法
        for k in range(n_sub):
            for i in range(n_sub):
                for j in range(n_sub):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # 计算平均路径长度，忽略不可达的节点对
        path_lengths = []
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                if dist[i, j] != float('inf'):
                    path_lengths.append(dist[i, j])

        if len(path_lengths) > 0:
            return np.mean(path_lengths), np.std(path_lengths)
        else:
            return 0, 0

    # 计算CS和DS图的平均路径长度
    cs_path_length_mean, cs_path_length_std = calculate_approx_path_length(cs_adj)
    ds_path_length_mean, ds_path_length_std = calculate_approx_path_length(ds_adj)

    # 保存路径长度统计信息
    path_length_stats = {
        'cs_path_length_mean': cs_path_length_mean,
        'cs_path_length_std': cs_path_length_std,
        'ds_path_length_mean': ds_path_length_mean,
        'ds_path_length_std': ds_path_length_std
    }

    # 绘制网络结构指标总结
    plt.figure(figsize=(10, 8))

    # 绘制平均聚类系数
    metrics = ['平均聚类系数', '平均路径长度', '平均度', '网络密度']
    cs_values = [np.mean(cs_clustering), cs_path_length_mean, np.mean(cs_degree),
                 np.sum(cs_adj) / (cs_adj.shape[0] * (cs_adj.shape[0] - 1))]
    ds_values = [np.mean(ds_clustering), ds_path_length_mean, np.mean(ds_degree),
                 np.sum(ds_adj) / (ds_adj.shape[0] * (ds_adj.shape[0] - 1))]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, cs_values, width, label='CS图')
    ax.bar(x + width / 2, ds_values, width, label='DS图')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title('CS图和DS图的网络结构指标比较')

    # 添加数值标签
    for i, v in enumerate(cs_values):
        ax.text(i - width / 2, v + v * 0.05, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(ds_values):
        ax.text(i + width / 2, v + v * 0.05, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_metrics.jpg'))
    plt.close()

    print(f"图结构特性分析已保存至目录: {output_dir}")

    return path_length_stats


def calculate_shortest_paths(cs_adj, ds_adj, cd_adj):
    """计算CS、DS和CD三个图之间的最短路径分布"""
    # 此函数仅返回一个示例实现，由于计算量大，实际使用时可能需要优化
    pass  # 实现省略，可根据需要添加


def plot_loss_curves(train_losses, val_losses, output_dir='./', output_file='loss_curves.jpg'):
    """绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可以为None）
        output_dir: 输出目录
        output_file: 输出文件名
    """
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(train_losses) + 1))

    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='验证损失')

    plt.title('训练过程中的损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加最终损失值标注
    plt.text(epochs[-1], train_losses[-1], f'{train_losses[-1]:.4f}',
             verticalalignment='bottom', horizontalalignment='right')
    if val_losses is not None:
        plt.text(epochs[-1], val_losses[-1], f'{val_losses[-1]:.4f}',
                 verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path)
    plt.close()

    print(f"损失曲线已保存至: {output_path}")


def plot_roc_curves(fold_roc_data, output_dir='./', output_file='roc_curves.jpg'):
    """绘制五折交叉验证的ROC曲线 - 同时提供平均ROC和总和ROC
    
    Args:
        fold_roc_data: 包含每折ROC数据的列表
        output_dir: 输出目录
        output_file: 输出文件名
    """
    from sklearn.metrics import roc_curve, auc
    
    # 创建包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ==================== 左图：平均ROC曲线 ====================
    
    # 存储每折的TPR，用于计算平均ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # 为每一折绘制ROC曲线
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, fold_data in enumerate(fold_roc_data):
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(fold_data['true_labels'], fold_data['predictions'])
        roc_auc = auc(fpr, tpr)
        
        # 绘制当前折的ROC曲线
        ax1.plot(fpr, tpr, color=colors[i], alpha=0.7, linewidth=2,
                 label=f'Fold {fold_data["fold"]} (AUC = {roc_auc:.3f})')
        
        # 插值到统一的FPR网格上
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    # 绘制对角线（随机分类器）
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # 绘制平均ROC曲线
    ax1.plot(mean_fpr, mean_tpr, color='black', linewidth=3,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # 添加置信区间
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2,
                     label='± 1 std. dev.')
    
    # 设置左图属性
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Average ROC - Cross Validation', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 右图：总和ROC曲线 ====================
    
    # 合并所有折的预测结果
    all_true_labels = np.concatenate([fold_data['true_labels'] for fold_data in fold_roc_data])
    all_predictions = np.concatenate([fold_data['predictions'] for fold_data in fold_roc_data])
    
    # 计算总和ROC曲线
    pooled_fpr, pooled_tpr, _ = roc_curve(all_true_labels, all_predictions)
    pooled_auc = auc(pooled_fpr, pooled_tpr)
    
    # 绘制总和ROC曲线
    ax2.plot(pooled_fpr, pooled_tpr, color='darkblue', linewidth=4,
             label=f'Pooled ROC (AUC = {pooled_auc:.3f})')
    
    # 为了对比，也绘制各折的ROC曲线（淡化显示）
    for i, fold_data in enumerate(fold_roc_data):
        fpr, tpr, _ = roc_curve(fold_data['true_labels'], fold_data['predictions'])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=colors[i], alpha=0.3, linewidth=1,
                 label=f'Fold {fold_data["fold"]} (AUC = {roc_auc:.3f})')
    
    # 绘制对角线（随机分类器）
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    # 设置右图属性
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Pooled ROC - All Predictions Combined', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算总样本数统计
    total_samples = len(all_true_labels)
    total_positive = np.sum(all_true_labels)
    total_negative = total_samples - total_positive
    
    print(f"ROC曲线已保存至: {output_path}")
    print(f"平均ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"总和ROC AUC: {pooled_auc:.4f}")
    print(f"总样本数: {total_samples} (正样本: {total_positive}, 负样本: {total_negative})")
    
    return mean_auc, std_auc, pooled_auc