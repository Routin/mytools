import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import csv
import torch
import cv2
from PIL import Image
__all__ = ['plot_tensor_2d',
           'DynamicLossPlotter',
           'plot_losses',
           'plot_relative_position_curve',
           'plot_attention_heatmap',
           'plot_image_text_attention',
           'plot_multihead_attention_heatmap',
           'plot_tensor_to_image']
# 把二维矩阵画成灰度图像
def plot_tensor_2d(tensor, cmap='gray'):
    # 检查传入的tensor类型并转换为NumPy数组
    if isinstance(tensor, np.ndarray):
        array_2d = tensor
    elif isinstance(tensor, torch.Tensor):
        array_2d = tensor.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    # 将数组值缩放到0-255之间（如果需要的话）
    array_2d = ((array_2d - array_2d.min()) / (array_2d.max() - array_2d.min()) * 255).astype(np.uint8)

    # 使用Matplotlib绘制图像
    plt.imshow(array_2d, cmap='gray')
    plt.colorbar()
    plt.show()

# 动态绘制loss曲线
from IPython.display import display, clear_output
'''
# class DynamicLossPlotter 使用示例
plotter = DynamicLossPlotter(save_path="dynamic_loss_plot.png", loss_data_file="loss_data.csv")
for epoch in range(1, 101):
    loss = np.random.rand()  # 模拟获取一个新的loss值
    plotter.update(loss)  # 更新图表
    time.sleep(0.2)  # 模拟训练时间

plotter.close()
'''
class DynamicLossPlotter:
    def __init__(self, save_path=None, loss_data_file=None):
        self.loss_values = []
        self.fig, self.ax = plt.subplots()
        self.save_path = save_path  # 图片保存路径
        self.loss_data_file = loss_data_file  # Loss数据保存文件

    def to_numpy(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def update(self, new_loss):
        new_loss = self.to_numpy(new_loss)
        self.loss_values.append(new_loss)
        self._plot()

        if self.loss_data_file:
            self.save_loss_data()

    def _plot(self):
        self.ax.clear()
        self.ax.plot(self.loss_values, label='Training Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Over Time')
        self.ax.legend()

        if self.save_path:
            self.fig.savefig(self.save_path)  # 保存图像

        clear_output(wait=True)  # 清除输出单元格，仅保留最新的图表
        display(self.fig)  # 显示新的图表
        plt.pause(0.1)  # 暂停，使图形有时间更新

    def save_loss_data(self):
        with open(self.loss_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.loss_values)

    def close(self):
        plt.close(self.fig)

# 绘制多个loss曲线
def plot_losses(loss_data_dict):
    """
    绘制多个loss曲线。
    :param loss_data_dict: 一个字典，其中键是标签（用于图例）和值是loss值的列表。

    使用示例：
    loss_data_1 = [0.9, 0.8, 0.7, 0.6, 0.5]
    loss_data_2 = [0.95, 0.85, 0.75, 0.65, 0.55]
    loss_data_3 = [0.92, 0.82, 0.72, 0.62, 0.52]

    loss_data_dict = {
        'Model 1': loss_data_1,
        'Model 2': loss_data_2,
        'Model 3': loss_data_3,
    }
    plot_losses(loss_data_dict)
    """
    plt.figure()
    
    for label, loss_values in loss_data_dict.items():
        plt.plot(loss_values, label=label)
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Training Losses')
    plt.legend()
    plt.show()

# 绘制位置编码等方法的衰减曲线
def plot_relative_position_curve(X):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise ValueError("Input must be a NumPy array, PyTorch tensor, or TensorFlow tensor")

    if X.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    # 计算每行X[i]与X[0]的内积值
    inner_products = np.dot(X, X[0])

    # 创建X轴数据，即i的值
    x_axis = np.arange(len(X))

    # 创建一个平滑的曲线连接内积值，不显示数据点
    plt.plot(x_axis, inner_products, linestyle='-')
    plt.xlabel('Relative Position')
    plt.ylabel('Position Similarity')
    plt.title('Decay Curve')
    plt.grid(True)
    plt.show()

# 绘制注意力热力图
def plot_attention_heatmap(input_seq, output_seq, attention_weights, cmap='viridis'):
    """
    绘制注意力热力图。

    参数：
        input_seq (list): 输入序列的单词（或其他元素）列表。
        output_seq (list): 输出序列的单词（或其他元素）列表。
        attention_weights (numpy.array): 注意力权重矩阵。
        cmap (str): 颜色映射。默认为'viridis'。

    返回：
        None
    """
    # 创建图像
    fig, ax = plt.subplots()
    # 使用Matplotlib的imshow函数绘制热力图
    cax = ax.matshow(attention_weights, cmap=cmap)

    # 添加颜色条,设置轴标签
    fig.colorbar(cax)
    ax.set_xticklabels([''] + input_seq, rotation=90)
    ax.set_yticklabels([''] + output_seq)

    # 显示网格线,设置轴
    ax.grid(visible=True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    # 显示图像
    plt.show()

# 图像文本对的注意力热力图
def plot_image_text_attention(image_path, patch_size, output_seq, attention_weights):
    # 读取并调整图像尺寸
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    patch_height, patch_width = patch_size
    image_height, image_width, _ = image.shape
    rows = image_height // patch_height
    cols = image_width // patch_width

    for word_idx, word in enumerate(output_seq):
        # 提取与当前输出单词相关的注意力权重
        word_attention_weights = attention_weights[word_idx, :]

        # 归一化
        word_attention_weights = word_attention_weights / np.max(word_attention_weights)

        # 初始化一个全零的overlay
        overlay = np.zeros((image_height, image_width))

        for patch_idx, weight in enumerate(word_attention_weights):
            row = patch_idx // cols
            col = patch_idx % cols
            overlay[row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width] = weight

        # 转换overlay为颜色图像，并与原图像叠加
        overlay_color = cv2.applyColorMap(cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_color = cv2.addWeighted(image, 0.6, overlay_color, 0.4, 0)

        # 显示图像
        plt.figure(figsize=(10, 10))
        im = plt.imshow(cv2.cvtColor(overlay_color, cv2.COLOR_BGR2RGB))
        plt.colorbar(im, orientation='vertical')
        plt.title(f"Output word: {word}")
        plt.axis('off')
        plt.show()

def plot_multihead_attention_heatmap(input_seq, output_seq, attention_weights, cmap='viridis'):
    """
    绘制多头注意力热力图。

    参数：
        input_seq (list): 输入序列的单词（或其他元素）列表。
        output_seq (list): 输出序列的单词（或其他元素）列表。
        attention_weights (numpy.array): 注意力权重矩阵，形状为[num_heads, len(output_seq), len(input_seq)]。
        cmap (str): 颜色映射。默认为'viridis'。

    返回：
        None
    """
    
    num_heads = attention_weights.shape[0]
    
    fig, axes = plt.subplots(1, num_heads, figsize=(15, 5 * num_heads))
    
    if num_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        cax = ax.matshow(attention_weights[i], cmap=cmap)
        fig.colorbar(cax, ax=ax)
        ax.set_xticklabels([''] + input_seq, rotation=90)
        ax.set_yticklabels([''] + output_seq)
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_title(f'Head {i+1}')
        
    plt.tight_layout()
    plt.show()

def tensor_to_image_final(tensor):
    """
    将给定的tensor转换为PIL图像。
    """
    
    # 确保tensor至少有两个维度
    assert len(tensor.shape) >= 2, "预期tensor至少有2个维度"
    
    # 如果是灰度图像并且没有通道维度，增加一个通道维度
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    
    # 检查tensor的形状
    if tensor.shape[0] in [1, 3, 4]:  # 灰度或C x H x W
        tensor = tensor.permute(1, 2, 0)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:  # H x W x 1
        tensor = tensor.squeeze(-1)  # 删除通道维度，因为PIL可以处理 H x W 的灰度图像
    
    # 将tensor转换为numpy数组
    numpy_image = tensor.numpy()
    
    # 如果tensor值在0到1之间，乘以255
    if np.max(numpy_image) <= 1.0:
        numpy_image = (numpy_image * 255).astype(np.uint8)
    
    # 使用PIL从numpy数组创建图像
    image = Image.fromarray(numpy_image)
    
    return image

def plot_tensor_to_image(tensor):
    """
    将给定的tensor转换为PIL图像并显示。
    
    参数:
        tensor (torch.Tensor): 输入的图像tensor，可以有以下形状之一:
                               H x W (灰度图像)
                               1 x H x W (灰度图像)
                               H x W x 1 (灰度图像)
                               C x H x W (彩色图像, 其中C为通道数)
                               H x W x C (彩色图像)
    """
    
    # 使用之前的函数将tensor转换为图像
    image = tensor_to_image_final(tensor)
    
    # 使用matplotlib显示图像
    plt.figure(figsize=(5, 5))
    if image.mode == "L":  # 如果是灰度图像
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()