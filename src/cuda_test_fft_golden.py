import numpy as np
import matplotlib.pyplot as plt
# from py_visualizer import display_image
def print_result(signal, result, result_golden):
    # 计算误差图像
    error = np.abs(result - result_golden)
    
    # 创建一个包含四个子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 绘制原始二维信号 (signal)
    im1 = axes[0, 0].imshow(np.abs(signal), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Input')
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')
    
    # 绘制第一个输入 (result)
    im2 = axes[0, 1].imshow(np.abs(result), cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Result')
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')
    
    # 绘制第二个输入 (result_golden)
    im3 = axes[1, 0].imshow(np.abs(result_golden), cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Golden result')
    fig.colorbar(im3, ax=axes[1, 0], orientation='vertical')
    
    # 绘制误差图像
    im4 = axes[1, 1].imshow(error, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Error Image')
    fig.colorbar(im4, ax=axes[1, 1], orientation='vertical')
    
    # 调整布局以防止重叠
    plt.tight_layout()
    
    # 显示图像
    plt.show()

def custom_allclose(a, b, scalar = 1000.0, equal_nan=False):
    """
    根据输入数组的数据类型自动选择适当的 rtol 和 atol，
    然后使用 np.allclose 进行比较。

    参数:
    a (array_like): 第一个输入数组。
    b (array_like): 第二个输入数组。
    equal_nan (bool, optional): 如果为 True，则两个 NaN 值被视为相等。默认为 False。

    返回:
    bool: 如果两个数组在给定的容差范围内相等，则返回 True，否则返回 False。
    """

    # 确保 a 和 b 是 NumPy 数组
    a = np.asarray(a)
    b = np.asarray(b)

    # 获取输入数组的数据类型
    dtype_a = a.dtype
    dtype_b = b.dtype

    # 选择更严格的 dtype 作为参考
    dtype = np.promote_types(dtype_a, dtype_b)
    if dtype == dtype_a: dtype = dtype_b
    else: dtype = dtype_a

    # 根据 dtype 选择合适的 rtol 和 atol
    if np.issubdtype(dtype, np.floating):
        if dtype == np.float32:
            rtol = 1e-5
            atol = 1e-8
        elif dtype == np.float64:
            rtol = 1e-10
            atol = 1e-12
        else:
            # 对于其他浮点类型，默认使用 np.allclose 的默认值
            rtol = 1e-5
            atol = 1e-8
    elif np.issubdtype(dtype, np.complexfloating):
        if dtype == np.complex64:
            rtol = 1e-5
            atol = 1e-8
        elif dtype == np.complex128:
            rtol = 1e-10
            atol = 1e-12
        else:
            # 对于其他复数类型，默认使用 np.allclose 的默认值
            rtol = 1e-5
            atol = 1e-8
    else:
        # 对于非浮点和非复数类型，直接使用 np.allclose 的默认值
        rtol = 1e-5
        atol = 1e-8

    # 调用 np.allclose 进行比较
    # print("relative error=", rtol * scalar, ", abs error=",atol* scalar)
    res = np.allclose(a, b, rtol=rtol * scalar, atol=atol* scalar, equal_nan=equal_nan)
    print("*            relative error=%e, abs error=%e"%(np.max(np.abs(a-b)), np.max(np.abs((a-b)/a))))
    return res

def check_rfft(signal, result, debug = False):
    # print("--->", int(np.issubdtype(signal.dtype, np.floating)))
    fft_result = [np.fft.fftn, np.fft.rfftn][int(np.issubdtype(signal.dtype, np.floating))](signal)
    assert(result.shape == fft_result.shape)
    are_close = custom_allclose(result, fft_result)
    if debug or not are_close:
        print("shpae : ", signal.shape, result.shape, fft_result.shape)
        print("type : ", signal.dtype, result.dtype, fft_result.dtype)
        print("min : ", np.min(signal), np.min(result), np.min(fft_result))
        print("max : ", np.max(signal), np.max(result), np.max(fft_result))
    if not are_close: print_result(signal, result, fft_result)
    return are_close
