import numpy as np
import pandas as pd
from collections import Counter

class CART:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        """
        初始化 CART 分类树
        :param max_depth: 最大深度
        :param min_samples_split: 最小分裂样本数
        :param min_impurity_decrease: 最小不纯度减少量
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y):
        """
        训练 CART 树
        :param X: 特征矩阵
        :param y: 标签向量
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        递归构建树
        :param X: 特征矩阵
        :param y: 标签向量
        :param depth: 当前深度
        :return: 节点字典
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件
        if depth == self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return self._create_leaf(y)

        # 计算当前节点的基尼系数
        current_gini = self._gini(y)

        # 初始化最佳分裂
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        # 遍历所有特征
        for feature in range(n_features):
            # 获取当前特征的所有值
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)

            # 遍历所有可能的切分点
            for threshold in unique_values:
                # 划分数据
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # 计算分裂后的基尼系数
                left_gini = self._gini(y[left_indices])
                right_gini = self._gini(y[right_indices])
                n_left = len(left_indices)
                n_right = len(right_indices)
                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini

                # 计算信息增益
                gain = current_gini - weighted_gini

                # 更新最佳分裂
                if gain > best_gain and gain > self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = (X[left_indices], y[left_indices])
                    best_right = (X[right_indices], y[right_indices])

        # 如果没有找到最佳分裂，则创建叶子节点
        if best_gain <= self.min_impurity_decrease:
            return self._create_leaf(y)

        # 递归构建子树
        left_tree = self._build_tree(best_left[0], best_left[1], depth + 1)
        right_tree = self._build_tree(best_right[0], best_right[1], depth + 1)

        # 返回当前节点
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'depth': depth,
            'is_leaf': False  # 明确设置 is_leaf 为 False
        }

    def _create_leaf(self, y):
        """
        创建叶子节点
        :param y: 标签向量
        :return: 叶子节点字典
        """
        # 统计标签的频率
        class_counts = Counter(y)
        # 返回最常见的标签
        most_common_class = max(class_counts, key=class_counts.get)
        return {
            'class': most_common_class,
            'counts': class_counts,
            'is_leaf': True  # 明确设置 is_leaf 为 True
        }

    def _gini(self, y):
        """
        计算基尼系数
        :param y: 标签向量
        :return: 基尼系数
        """
        class_counts = Counter(y)
        n_samples = len(y)
        if n_samples == 0:
            return 0
        # 计算每个类别的概率
        probs = [count / n_samples for count in class_counts.values()]
        # 计算基尼系数
        gini = 1 - sum(p ** 2 for p in probs)
        return gini

    def predict(self, X):
        """
        预测样本的类别
        :param X: 特征矩阵
        :return: 预测的类别
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        """
        预测单个样本的类别
        :param x: 单个样本
        :param tree: 当前树节点
        :return: 预测的类别
        """
        if tree['is_leaf']:
            return tree['class']

        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

    def print_tree(self):
        """
        打印树结构
        """
        self._print_tree_recursive(self.tree, 0)

    def _print_tree_recursive(self, node, depth=0):
        """
        递归打印树结构
        :param node: 当前节点
        :param depth: 当前深度
        """
        if node['is_leaf']:
            print(f"{'  ' * depth}Leaf: class={node['class']}, counts={node['counts']}")
        else:
            print(f"{'  ' * depth}Feature {node['feature']} <= {node['threshold']}")
            print(f"{'  ' * depth}Left:")
            self._print_tree_recursive(node['left'], depth + 1)
            print(f"{'  ' * depth}Right:")
            self._print_tree_recursive(node['right'], depth + 1)


# 示例用法
if __name__ == "__main__":
    # 示例数据
    X = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [3, 1],
        [3, 2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    # 创建 CART 树
    cart = CART(max_depth=2, min_samples_split=2)
    cart.fit(X, y)

    # 打印树结构
    print("CART Tree Structure:")
    cart.print_tree()

    # 预测
    X_test = np.array([[1, 1], [3, 2], [2, 1.5]])
    predictions = cart.predict(X_test)
    print("\nPredictions:")
    print(predictions)