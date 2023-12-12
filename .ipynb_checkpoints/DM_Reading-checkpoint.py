import numpy as np

def get_stiffness_matrix(file_path):
    """
    从指定的文件路径读取并返回刚度矩阵。

    参数：
    file_path (str)：包含刚度矩阵数据的文件路径。

    返回：
    numpy.ndarray：读取并恢复的刚度矩阵。
    """
    # 获取节点个数及自由度
    def get_size(file_path):
        max_node = 0
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                node2 = int(data[2])
                max_node = max(max_node, node1, node2)
        return max_node * 6  # 这里我们假设每个节点都有6个自由度

    # 从文件中读取稀疏矩阵数据
    def read_data(file_path, size):
        matrix = np.zeros((size, size))
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                dof1 = int(data[1])
                node2 = int(data[2])
                dof2 = int(data[3])
                value = float(data[4])
                # 计算在刚度矩阵中的位置，注意我们这里索引要减1，因为Python是0基索引
                index1 = (node1 - 1) * 6 + dof1 - 1
                index2 = (node2 - 1) * 6 + dof2 - 1
                matrix[index1, index2] = value
        return matrix

    size = get_size(file_path)
    matrix = read_data(file_path, size)

    # 将读取到的下三角部分复制到上三角部分，恢复完整的刚度矩阵
    for i in range(size):
        for j in range(i+1, size):
            matrix[i, j] = matrix[j, i]
    
    return matrix
