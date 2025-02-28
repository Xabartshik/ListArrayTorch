import time
import random
import numpy as np
import torch
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
def matrix_multiply_lists(matrix1, matrix2):
    """
    Перемножает две матрицы, представленные как списки Python.

    Args:
        matrix1: Первая матрица (список списков).
        matrix2: Вторая матрица (список списков).

    Returns:
        Результирующая матрица (список списков) или ValueError, если матрицы несовместимы.
    """
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Матрицы не совместимы для умножения")

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]  # Инициализация результирующей матрицы

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]  # Вычисление элемента результирующей матрицы
    return result


def generate_matrix(rows, cols, method='list'):
    """
    Генерирует матрицу заданного размера, используя указанный метод.

    Args:
        rows: Количество строк.
        cols: Количество столбцов.
        method: Метод генерации ('list', 'numpy', 'torch_cpu', 'torch_gpu').

    Returns:
        Матрица в заданном формате.
    """
    if method == 'list':
        return [[random.random() for _ in range(cols)] for _ in range(rows)]  # Список списков
    elif method == 'numpy':
        return np.random.rand(rows, cols)  # NumPy array
    elif method == 'torch_cpu':
        return torch.rand(rows, cols)  # Torch tensor на CPU
    elif method == 'torch_gpu':
        if torch.cuda.is_available():
            return torch.rand(rows, cols).cuda()  # Torch tensor на GPU, если доступен
        else:
            print("Нищеброд, что, на нормальную видеокарту денег не хватило, а? А? А?")
            return torch.rand(rows, cols)  # Torch tensor на CPU, если GPU недоступен
    else:
        raise ValueError("Неправильный метод генерации матрицы")


matrix_size = 400

# Списки
start_time = time.time()  # Начало замера времени
matrix1_list = generate_matrix(matrix_size, matrix_size, 'list')
matrix2_list = generate_matrix(matrix_size, matrix_size, 'list')
result_list = matrix_multiply_lists(matrix1_list, matrix2_list)
end_time = time.time()  # Конец замера времени
list_time = end_time - start_time
print(f"List время вычисления: {list_time:.4f} секунд")


# NumPy
start_time = time.time()
matrix1_np = generate_matrix(matrix_size, matrix_size, 'numpy')
matrix2_np = generate_matrix(matrix_size, matrix_size, 'numpy')
result_np = np.dot(matrix1_np, matrix2_np)  # Использование np.dot для умножения матриц NumPy
end_time = time.time()
numpy_time = end_time - start_time
print(f"NumPy время вычисления: {numpy_time:.4f} секунд")


# Torch CPU
start_time = time.time()
matrix1_torch_cpu = generate_matrix(matrix_size, matrix_size, 'torch_cpu')
matrix2_torch_cpu = generate_matrix(matrix_size, matrix_size, 'torch_cpu')
result_torch_cpu = torch.mm(matrix1_torch_cpu, matrix2_torch_cpu) # Использование torch.mm для умножения тензоров
end_time = time.time()
torch_cpu_time = end_time - start_time
print(f"Torch CPU время вычисления: {torch_cpu_time:.4f} секунд")

#
# # Torch GPU
# start_time = time.time()
# matrix1_torch_gpu = generate_matrix(matrix_size, matrix_size, 'torch_gpu')
# matrix2_torch_gpu = generate_matrix(matrix_size, matrix_size, 'torch_gpu')
# result_torch_gpu = torch.mm(matrix1_torch_gpu, matrix2_torch_gpu)
# end_time = time.time()
# torch_gpu_time = end_time - start_time
# print(f"Torch GPU время вычисления: {torch_gpu_time:.4f} seconds")