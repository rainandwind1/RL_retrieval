import torch
import numpy as np
# a = torch.tensor([[1,2,3],[2,3,4]])
# for i in a:
#     print(i)
# a = [torch.tensor([1,2,3]), torch.tensor([4,5,6]), torch.tensor([7,8,9])]
# c = np.stack(i) for i in a
# print(c)


# class Solution(object):
#     def write_mem(self, mem, i, j):
#         mem[i,:], mem[:, j] = 1
#         while i < mem.shape[0] and j < mem.shape[1]:
#             mem[i, j] = 1
#             i += 1
#             j += 1
#         if 0 not in mem:
#             return mem, False
#         return mem, True

#     def solveNQueens(self, n):
#         o = ["."*n for _ in range(n)]
#         mem = np.array([[0]*n for _ in range(n)])
#         self.res = []
#         def search_queens(row, pre_col, o, mem):
#             mem_ls, o_ls, flag_ls = [], [], []
#             for i in range(n):
#                 if i == pre_col:
#                     continue
                

#         for i in range(n):
#             mem_ls, o_ls, flag_ls = search_queens(0, i, o, mem)
#             for flag, idx in enumerate(flag_ls):
#                 if flag:


# class Solution:
#     def solveNQueens(self, n):
#         # 按行枚举
#         col, diag, reverse_diag = [0] * n, [0] * 2 * n, [0] * 2 * n
#         ret = []
#         grid = [['.'] * n for _ in range(n)]
#         def dfs(u):
#             # 终点返回
#             if u == n:
#                 ret.append(["".join(grid[i]) for i in range(n)])
#                 return
#             for i in range(n):
#                 if not col[i] and not diag[u + i] and not reverse_diag[n - u + i]:
#                     grid[u][i] = 'Q'
#                     col[i], diag[u + i], reverse_diag[n - u + i] = 1, 1, 1
#                     dfs(u + 1)
#                     grid[u][i] = '.'
#                     col[i], diag[u + i], reverse_diag[n - u + i] = 0, 0, 0
#         dfs(0)
#         return ret

# A = Solution()
# print(A.solveNQueens(4))


for idx, i in enumerate([4,5,6,7,89]):
    print(idx, i)
