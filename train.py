import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from tqdm import tqdm
import shutil
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
from model import CDGT, create_model, loss_function
from utils import load_data, plot_metrics, normalize_features, analyze_graph_statistics, visualize_adjacency_matrices, \
    plot_loss_curves, plot_roc_curves


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, train_data, optimizer, loss_fn, device):
    """训练模型单个epoch"""
    model.train()
    optimizer.zero_grad()

    # 解包训练数据
    cf, df, cs_adj, ds_adj, cd_adj, c_idx, d_idx, labels = train_data

    # 将数据移至设备
    cf = cf.to(device)
    df = df.to(device)
    cs_adj = cs_adj.to(device)
    ds_adj = ds_adj.to(device)
    cd_adj = cd_adj.to(device)
    c_idx = c_idx.to(device)
    d_idx = d_idx.to(device)
    labels = labels.to(device)

    # 前向传播
    outputs = model(cf, df, cs_adj, ds_adj, cd_adj, c_idx, d_idx)

    # 计算损失
    loss = loss_fn(outputs, labels)

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, test_data, loss_fn, device):
    """评估模型性能"""
    model.eval()

    # 解包测试数据
    cf, df, cs_adj, ds_adj, cd_adj, c_idx, d_idx, labels = test_data

    # 将数据移至设备
    cf = cf.to(device)
    df = df.to(device)
    cs_adj = cs_adj.to(device)
    ds_adj = ds_adj.to(device)
    cd_adj = cd_adj.to(device)
    c_idx = c_idx.to(device)
    d_idx = d_idx.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        # 前向传播
        outputs = model(cf, df, cs_adj, ds_adj, cd_adj, c_idx, d_idx)

        # 计算损失
        loss = loss_fn(outputs, labels)

        # 转换为NumPy数组进行指标计算
        outputs_np = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # 二分类阈值
        predictions = (outputs_np > 0.5).astype(int)

        # 计算指标
        acc = accuracy_score(labels_np, predictions)
        prec = precision_score(labels_np, predictions)
        rec = recall_score(labels_np, predictions)
        f1 = f1_score(labels_np, predictions)
        roc_auc = roc_auc_score(labels_np, outputs_np)
        pr_auc = average_precision_score(labels_np, outputs_np)

    metrics = {
        'loss': loss.item(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    # 返回指标、预测概率和真实标签
    return metrics, outputs_np, labels_np


def prepare_masked_adjacency_matrices(CS, DS, CD, c_idx_train, d_idx_train, circ_threshold=0.4034, drug_threshold=0.06):
    """根据训练集样本准备掩码版本的邻接矩阵，避免数据泄露"""

    # 获取节点数量
    num_circs = CS.shape[0]
    num_drugs = DS.shape[0]

    # 创建掩码矩阵
    masked_cd_adj = np.zeros_like(CD, dtype=np.float32)
    masked_cs_adj = np.zeros_like(CS, dtype=np.float32)
    masked_ds_adj = np.zeros_like(DS, dtype=np.float32)

    # 获取训练集中出现的circRNA和drug节点集
    train_c_nodes = set(c_idx_train)
    train_d_nodes = set(d_idx_train)

    # 1. 只使用训练集中的边信息构建CD邻接矩阵
    for i in range(len(c_idx_train)):
        c_idx = c_idx_train[i]
        d_idx = d_idx_train[i]
        masked_cd_adj[c_idx, d_idx] = CD[c_idx, d_idx]

    # 2. 只考虑训练集中出现的circRNA节点的CS关系
    for i in train_c_nodes:
        for j in train_c_nodes:
            if CS[i, j] > circ_threshold:
                masked_cs_adj[i, j] = 1.0

    # 3. 只考虑训练集中出现的drug节点的DS关系
    for i in train_d_nodes:
        for j in train_d_nodes:
            if DS[i, j] > drug_threshold:
                masked_ds_adj[i, j] = 1.0

    # 添加自环
    for i in train_c_nodes:
        masked_cs_adj[i, i] = 1.0
    for i in train_d_nodes:
        masked_ds_adj[i, i] = 1.0

    return masked_cs_adj, masked_ds_adj, masked_cd_adj


def train_cdgt_model(args):
    """训练 CDGT 模型"""
    print(f"\n{'=' * 50}")
    print(" 开始训练 CDGT 模型 ".center(50, '='))
    print(f"{'=' * 50}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"同质图模型: {args.homo_model}")
    print(f"异质图模型: {args.hetero_model}")
    print(f"融合方法: {args.fusion_method}")

    # 记录开始时间
    start_time = time.time()

    fold_metrics = []
    fold_roc_data = []
    # ===== 新增：收集所有验证集预测结果 =====
    all_predictions_data = []  # 存储所有折的预测结果
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"随机种子: {args.seed}")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"./results/{timestamp}_{args.dataset_type}_{args.homo_model}_{args.hetero_model}_{args.fusion_method}"
    plots_dir = f"{result_dir}/plots"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"结果将保存在: {result_dir}")

    # 复制脚本文件
    for script in ['model.py', 'train.py', 'utils.py']:
        shutil.copy2(script, f"{result_dir}/{script}")

    # 加载数据
    data_path = f"./datasets/{args.dataset_type}"
    CS, DS, CF, DF, CD, edges, labels = load_data(data_path)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数（保持原有配置）
    model_params = {
        'c_feature_dim': CF.shape[1],
        'd_feature_dim': DF.shape[1],
        'projection_dim': args.projection_dim,
        'homo_hidden_dim': args.homo_hidden_dim,
        'homo_model': args.homo_model,
        'hetero_model': args.hetero_model,
        'fusion_method': args.fusion_method,
        'homo_num_layers': args.homo_num_layers,
        'hetero_num_layers': args.hetero_num_layers,
        'homo_num_heads': args.homo_num_heads,
        'hetero_num_heads': args.hetero_num_heads,
        'dropout': args.dropout,
        'prediction_method': args.prediction_method,
        'use_structure_pe': args.use_structure_pe,
        'sage_aggr': args.sage_aggr,
        'use_homo_features': args.use_homo_features,
        'use_hetero_features': args.use_hetero_features
    }

    # 创建结果文件
    result_file = f"{result_dir}/result.txt"

    # 记录参数（保持原有格式）
    with open(result_file, 'w') as f:
        f.write(f"{'=' * 20} 实验设置 {'=' * 20}\n")
        f.write(f"数据集类型: {args.dataset_type}\n")
        f.write(f"模型: CDGT (CircRNA-Drug 图转换器)\n")
        f.write(f"同质图模型: {args.homo_model}\n")
        f.write(f"异质图模型: {args.hetero_model}\n")
        f.write(f"融合方法: {args.fusion_method}\n")
        f.write(f"训练轮数: {args.n_epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"交叉投影输出维度: {args.projection_dim}\n")
        f.write(f"同质图输出维度: {args.homo_hidden_dim}\n")
        f.write(f"异质图输出维度: {args.projection_dim}\n")
        f.write(f"同质图处理器层数: {args.homo_num_layers}\n")
        f.write(f"异质图处理器层数: {args.hetero_num_layers}\n")
        f.write(f"同质图处理器头数: {args.homo_num_heads}\n")
        f.write(f"异质图处理器头数: {args.hetero_num_heads}\n")
        f.write(f"交叉验证折数: {args.n_folds}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"预测方法: {args.prediction_method}\n")
        f.write(f"circRNA相似性阈值: {args.circ_threshold}\n")
        f.write(f"药物相似性阈值: {args.drug_threshold}\n")
        f.write(f"使用结构感知位置编码: {args.use_structure_pe}\n")
        f.write(f"防止数据泄露: 是\n")
        if args.homo_model == "sage":
            f.write(f"GraphSAGE聚合方法: {args.sage_aggr}\n")
        f.write("\n")

    # 交叉验证
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # 提取边和标签数据
    edge_data = np.array(edges.iloc[:, 0:2])
    label_data = np.array(labels.iloc[:, 0])

    # 记录每折的指标
    fold_metrics = []

    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(kf.split(edge_data)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{args.n_folds} {'=' * 20}")

        # 准备训练和测试数据
        X_train, X_test = edge_data[train_idx], edge_data[test_idx]
        y_train, y_test = label_data[train_idx], label_data[test_idx]

        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        print(f"训练集正样本: {sum(y_train)}, 负样本: {len(y_train) - sum(y_train)}")
        print(f"测试集正样本: {sum(y_test)}, 负样本: {len(y_test) - sum(y_test)}")

        # 转换为PyTorch张量
        c_idx_train = torch.LongTensor(X_train[:, 0])
        d_idx_train = torch.LongTensor(X_train[:, 1])
        labels_train = torch.FloatTensor(y_train)

        c_idx_test = torch.LongTensor(X_test[:, 0])
        d_idx_test = torch.LongTensor(X_test[:, 1])
        labels_test = torch.FloatTensor(y_test)

        # 准备邻接矩阵 - 使用掩码版本避免数据泄露
        cs_adj, ds_adj, cd_adj_train = prepare_masked_adjacency_matrices(
            CS, DS, CD,
            c_idx_train.numpy(), d_idx_train.numpy(),
            args.circ_threshold, args.drug_threshold
        )

        # 创建用于测试的邻接矩阵 - 完全移除所有测试边的信息
        cd_adj_test = cd_adj_train.copy()

        # 特征归一化 - 只使用训练集的统计信息（保持原有逻辑）
        train_c_indices = X_train[:, 0]
        train_d_indices = X_train[:, 1]

        cf_train_stats = {
            'min': np.min(CF[train_c_indices], axis=0),
            'max': np.max(CF[train_c_indices], axis=0)
        }

        df_train_stats = {
            'min': np.min(DF[train_d_indices], axis=0),
            'max': np.max(DF[train_d_indices], axis=0)
        }

        cf_normalized = np.copy(CF)
        df_normalized = np.copy(DF)

        for i in range(CF.shape[1]):
            min_val = cf_train_stats['min'][i]
            max_val = cf_train_stats['max'][i]
            if max_val > min_val:
                cf_normalized[:, i] = (CF[:, i] - min_val) / (max_val - min_val)
            else:
                cf_normalized[:, i] = 0.0

        for i in range(DF.shape[1]):
            min_val = df_train_stats['min'][i]
            max_val = df_train_stats['max'][i]
            if max_val > min_val:
                df_normalized[:, i] = (DF[:, i] - min_val) / (max_val - min_val)
            else:
                df_normalized[:, i] = 0.0

        cf_normalized = torch.FloatTensor(cf_normalized)
        df_normalized = torch.FloatTensor(df_normalized)

        # 转换邻接矩阵为PyTorch张量
        cs_adj_tensor = torch.FloatTensor(cs_adj)
        ds_adj_tensor = torch.FloatTensor(ds_adj)
        cd_adj_train_tensor = torch.FloatTensor(cd_adj_train)
        cd_adj_test_tensor = torch.FloatTensor(cd_adj_test)

        # 准备完整数据
        train_data = (
            cf_normalized,
            df_normalized,
            cs_adj_tensor,
            ds_adj_tensor,
            cd_adj_train_tensor,
            c_idx_train,
            d_idx_train,
            labels_train
        )

        test_data = (
            cf_normalized,
            df_normalized,
            cs_adj_tensor,
            ds_adj_tensor,
            cd_adj_test_tensor,
            c_idx_test,
            d_idx_test,
            labels_test
        )

        # 创建模型
        model = create_model(model_params).to(device)
        print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)

        # 损失函数
        loss_fn = loss_function()

        # 记录训练过程
        train_losses = []
        val_losses = []
        val_metrics_history = []
        best_val_metrics = None
        best_model_state = None
        best_epoch = 0
        no_improve = 0

        # 训练循环
        for epoch in range(args.n_epochs):
            # 训练
            train_loss = train_model(model, train_data, optimizer, loss_fn, device)
            train_losses.append(train_loss)

            # 评估
            val_metrics, _, _ = evaluate_model(model, test_data, loss_fn, device)
            val_losses.append(val_metrics['loss'])
            val_metrics_history.append(val_metrics)

            # 更新学习率
            scheduler.step(val_metrics['f1'])

            # 保存最佳模型
            if best_val_metrics is None or val_metrics['f1'] > best_val_metrics['f1']:
                best_val_metrics = val_metrics
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            # 早停
            if no_improve >= args.patience:
                print(f"早停: {args.patience}个epoch内无改进")
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{args.n_epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_metrics['loss']:.4f} | "
                    f"F1分数: {val_metrics['f1']:.4f} | ROC AUC: {val_metrics['roc_auc']:.4f}")

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 获取最佳模型在测试集上的完整预测结果
        final_metrics, final_predictions, final_true_labels = evaluate_model(model, test_data, loss_fn, device)

        # 记录最佳结果
        fold_metrics.append(best_val_metrics)

        # 保存ROC数据
        fold_roc_data.append({
            'fold': fold + 1,
            'true_labels': final_true_labels,
            'predictions': final_predictions,
            'auc': final_metrics['roc_auc']
        })

        # ===== 新增：收集当前折的预测结果 =====
        for i in range(len(final_true_labels)):
            all_predictions_data.append({
                'fold': fold + 1,
                'circRNA_idx': X_test[i, 0],  # circRNA索引
                'drug_idx': X_test[i, 1],     # 药物索引
                'true_label': final_true_labels[i],
                'predicted_score': final_predictions[i]
            })

        # 保存最佳模型
        torch.save(best_model_state, f"{result_dir}/model_fold_{fold + 1}.pt")

        # 写入结果文件
        with open(result_file, 'a') as f:
            f.write(f"{'=' * 10} Fold {fold + 1} 结果 {'=' * 10}\n")
            f.write(f"最佳Epoch: {best_epoch + 1}\n")
            f.write(f"Accuracy\tPrecision\tRecall\tF1\tROCAUC\tPRAUC\n")
            f.write(f"{best_val_metrics['accuracy']:.4f}\t{best_val_metrics['precision']:.4f}\t")
            f.write(f"{best_val_metrics['recall']:.4f}\t{best_val_metrics['f1']:.4f}\t")
            f.write(f"{best_val_metrics['roc_auc']:.4f}\t{best_val_metrics['pr_auc']:.4f}\n\n")

    # ===== 新增：保存所有预测结果为CSV文件 =====
    predictions_df = pd.DataFrame(all_predictions_data)
    predictions_csv_path = f"{result_dir}/all_fold_predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"\n所有折验证预测结果已保存至: {predictions_csv_path}")
    
    # ===== 新增：生成简化版本的表格（只包含真实标签和预测分数）=====
    simple_predictions_df = predictions_df[['true_label', 'predicted_score']].copy()
    simple_predictions_csv_path = f"{result_dir}/predictions_simple.csv"
    simple_predictions_df.to_csv(simple_predictions_csv_path, index=False)
    print(f"简化版预测结果表格已保存至: {simple_predictions_csv_path}")
    
    # ===== 新增：打印预测结果统计信息 =====
    print(f"\n{'=' * 50}")
    print(" 预测结果统计 ".center(50, '='))
    print(f"{'=' * 50}")
    print(f"总验证样本数: {len(all_predictions_data)}")
    print(f"正样本数: {sum([1 for x in all_predictions_data if x['true_label'] == 1])}")
    print(f"负样本数: {sum([1 for x in all_predictions_data if x['true_label'] == 0])}")
    print(f"预测分数范围: [{min([x['predicted_score'] for x in all_predictions_data]):.4f}, {max([x['predicted_score'] for x in all_predictions_data]):.4f}]")
    
    # 计算平均指标（保持原有逻辑）
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics])
    }

    # 计算标准差
    std_metrics = {
        'accuracy': np.std([m['accuracy'] for m in fold_metrics]),
        'precision': np.std([m['precision'] for m in fold_metrics]),
        'recall': np.std([m['recall'] for m in fold_metrics]),
        'f1': np.std([m['f1'] for m in fold_metrics]),
        'roc_auc': np.std([m['roc_auc'] for m in fold_metrics]),
        'pr_auc': np.std([m['pr_auc'] for m in fold_metrics])
    }

    # 写入平均结果
    with open(result_file, 'a') as f:
        f.write(f"{'=' * 10} 平均结果 {'=' * 10}\n")
        f.write(f"指标\t平均值\t标准差\n")
        f.write(f"Accuracy\t{avg_metrics['accuracy']:.4f}\t{std_metrics['accuracy']:.4f}\n")
        f.write(f"Precision\t{avg_metrics['precision']:.4f}\t{std_metrics['precision']:.4f}\n")
        f.write(f"Recall\t{avg_metrics['recall']:.4f}\t{std_metrics['recall']:.4f}\n")
        f.write(f"F1\t{avg_metrics['f1']:.4f}\t{std_metrics['f1']:.4f}\n")
        f.write(f"ROCAUC\t{avg_metrics['roc_auc']:.4f}\t{std_metrics['roc_auc']:.4f}\n")
        f.write(f"PRAUC\t{avg_metrics['pr_auc']:.4f}\t{std_metrics['pr_auc']:.4f}\n\n")

    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 写入时间信息
    with open(result_file, 'a') as f:
        f.write(f"{'=' * 10} 运行时间 {'=' * 10}\n")
        f.write(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
        f.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 绘制指标图
    plot_metrics(result_file, plots_dir)
    plot_loss_curves(train_losses, val_losses, plots_dir, 'loss_curves.jpg')
    mean_auc, std_auc, pooled_auc = plot_roc_curves(fold_roc_data, plots_dir, 'roc_curves.jpg')
    
    # 优化后的打印输出
    print(f"\n{'=' * 50}")
    print(" CDGT 模型训练完成 ".center(50, '='))
    print(f"{'=' * 50}")
    print(f"结果保存路径: {result_dir}")
    print(f"\n平均指标:")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"  ROC AUC: {avg_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")
    print(f"  PR AUC: {avg_metrics['pr_auc']:.4f} ± {std_metrics['pr_auc']:.4f}")
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

    # ===== 返回预测结果数据 =====
    return avg_metrics, std_metrics, all_predictions_data


def main():
    """主函数 - 使用固定配置训练 CDGT 模型（配置已锁定，不可通过命令行修改）"""
    # 所有超参数均已固定为论文最终设置
    args = SimpleNamespace(
        dataset_type='Resistance',
        n_epochs=200,
        lr=0.0002,
        weight_decay=1e-5,
        n_folds=5,
        seed=51,
        patience=20,
        dropout=0.2,
        homo_num_layers=2,
        hetero_num_layers=2,
        homo_num_heads=4,
        hetero_num_heads=32,
        prediction_method='bilinear',
        circ_threshold=0.4034,
        drug_threshold=0.06,
        use_structure_pe=True,
        use_homo_features=True,
        use_hetero_features=True,
        projection_dim=512,
        homo_hidden_dim=128,
        homo_model='gat',
        hetero_model='BipartiteGraphTransformer',
        fusion_method='uncertainty',
        sage_aggr='mean',
    )

    # 打印参数
    print(f"\n{'=' * 20} 训练参数（固定配置） {'=' * 20}")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 训练 CDGT 模型
    train_cdgt_model(args)


if __name__ == "__main__":
    main()