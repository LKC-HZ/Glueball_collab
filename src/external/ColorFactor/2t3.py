import numpy as np

# 文件名
filename = "fabc.dat"

# 假设 SU(3)，张量大小 8x8x8
fabc = np.zeros((8, 8, 8), dtype=np.float64)

# 读取文件
with open(filename, 'r') as f:
    for line in f:
        # 跳过空行
        if not line.strip():
            continue
        # 解析行
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"行格式错误: {line}")
        a, b, c, val = parts
        a, b, c = int(a)-1, int(b)-1, int(c)-1  # 文件是从1开始的索引，转换为0开始
        val = float(val)
        fabc[a, b, c] = val

# # 测试：打印非零元素
# nz = np.argwhere(fabc != 0)
# for idx in nz:
#     i,j,k = idx
#     print(f"f[{i+1},{j+1},{k+1}] = {fabc[i,j,k]}")
    
def delta(a, b):
    """
    Kronecker delta function for a,b in 1..8
    Returns 1 if a==b, else 0
    """
    # if not (1 <= a <= 8) or not (1 <= b <= 8):
    #     raise ValueError("a和b必须在1到8之间")
    return 1 if a == b else 0

cf = 0

# for i in range(0, 8):
#     for j in range(0, 8):
#         for k in range(0, 8):
#             for l in range(0, 8):
#                 cf = cf + fabc[j,k,l]*fabc[i,j,k]*delta(i,l)

for a in range(0, 8):
    for b in range(0, 8):
        for c in range(0, 8):
                cf = cf + fabc[a,c,b]*fabc[a,b,c]*delta(c,c)
                
cf = cf/np.sqrt(24*8)

print(cf)

print (np.sqrt(3))