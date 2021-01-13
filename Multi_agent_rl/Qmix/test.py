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


# for idx, i in enumerate([4,5,6,7,89]):
#     print(idx, i)


# class Solution:
#     def ladderLength(self, beginWord, endWord, wordList):
#         res_dict = {}
#         wordList.append(beginWord)
#         res, mem = [], []
#         for word in wordList:
#             res_dict[word] = []
#         for idx, word in enumerate(wordList):
#             for i in range(idx+1, len(wordList)):
#                 count = 0
#                 for n, c in enumerate(wordList[idx]):
#                     if c != wordList[i][n]:
#                         count += 1
#                     if count > 1:
#                         break
#                 if count == 1:
#                     res_dict[wordList[idx]].append(wordList[i])
#                     res_dict[wordList[i]].append(wordList[idx])
#         print(res_dict)
#         def search_res(search_ls, count, mem):
#             for word in search_ls:
#                 if word == endWord:
#                     res.append(count)
#                     return
#                 else:
#                     if res_dict[word] not in mem:
#                         search_res(res_dict[word], count + 1, mem + [res_dict[word]])
#         search_res(res_dict[beginWord], 1, mem)
#         print(res)
#         return min(res)+1 if res else 0



# A = Solution()
# print(A.ladderLength(beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"]))

# class Solution:
#     def sortByBits(self, arr):
#         arr.sort()
#         res_dict = {}
#         for n in arr:
#             res_dict[n] = str(bin(n)).count('1')
#         res_dict = sorted(res_dict.items(), key = lambda items:items[1])
#         res = [n[0] for n in res_dict]
#         return res

# A = Solution()
# print(A.sortByBits([1024,512,256,128,64,32,16,8,4,2,1]))


# class Solution:
#     def findLongestWord(self, s, d):
#         res = None
#         flag = False
#         idx = 0
#         see = []
#         while idx < len(d):
#             target = s
#             cur = 0
#             print(d[idx])
#             for n, c in enumerate(d[idx]):
#                 if c in target[cur:]:
#                     print(cur)
#                     if n+1 == len(d[idx]):
#                         flag = True
#                         break
#                     cur = cur + target[cur:].index(c) + 1  
#                 else:
#                     break
#             if flag:
#                 see.append(d[idx])
#                 if not res:
#                     res = d[idx]
#                 else:
#                     if len(d[idx]) > len(res) or (len(d[idx]) == len(res) and d[idx][0] < res[0]):
#                         res = d[idx]
#                 flag = False
#             idx += 1
#         print(see)
#         return res

# A = Solution()
# print(A.findLongestWord( "bab",["ba","ab","a","b"]))


# class Solution(object):
#     def findRotateSteps(self, ring, key):
#         button = 1
#         self.res = float('+inf')
#         self.step = 0
#         # cur direction:left, right
#         def search_res(cur, direction, key_idx, count):
#             self.step += 1
#             print(count, self.res, self.step)
#             if count >= self.res:
#                 return
#             if key_idx == len(key):
#                 self.res = min(self.res, count)
#                 return
#             if ring[cur] == key[key_idx]:
#                 count += button
#                 search_res(cur, 1, key_idx + 1, count) # right
#                 search_res(cur, -1, key_idx + 1, count) # left
#                 return
#             if direction == 1: # right cur ++++
#                 step = 0
#                 while ring[cur] != key[key_idx]:
#                     step += 1
#                     if cur < len(ring) - 1:
#                         cur += 1
#                     else:
#                         cur = 0
#                 search_res(cur, 1, key_idx, count + step) # right
#                 search_res(cur, -1, key_idx, count + step) # left
#             else: # left  cur ++++
#                 step = 0
#                 while ring[cur] != key[key_idx]:
#                     step += 1
#                     if cur > 0:
#                         cur -= 1
#                     else:
#                         cur = len(ring) - 1
#                 search_res(cur, 1, key_idx, count + step) # right
#                 search_res(cur, -1, key_idx, count + step) # left
        
#         search_res(0, 1, 0, 0) # right
#         search_res(0, -1, 0, 0) # left
#         return self.res

# A = Solution()
# print(A.findRotateSteps("zneyk",
# "nkkynyeekzznezy"))

# class Solution(object):
#     def findLongestChain(self, pairs):
#         pairs.sort(key=lambda x: (x[1], x[0]))
#         print(pairs)
#         count = 0
#         end = -float('inf')
#         for i, p in enumerate(pairs):
#             if p[0] > end:
#                 count += 1
#                 end = p[1]
#         return count   
        
# A = Solution()
# print(A.findLongestChain([[-6,9],[1,6],[8,10],[-1,4],[-6,-2],[-9,8],[-5,3],[0,3]]))

import collections
class Solution(object):
    def sortArrayByParityII(self, A):
        even_dic = collections.deque()
        odd_dic = collections.deque()
        res = []
        count = 0
        for num in A:
            print(num, count)
            if count % 2 == 0:
                if num % 2 == 0:
                    res.append(num)
                    count += 1
                else:
                    if even_dic:
                        res.append(even_dic.pop())
                        count += 1
                    odd_dic.append(num)
            else:
                if num % 2 != 0:
                    res.append(num)
                    count += 1
                else:
                    if odd_dic:
                        res.append(odd_dic.pop())
                        count += 1
                    even_dic.append(num)
        while count < len(A):
            if count % 2 == 0:
                res.append(even_dic.pop())
            else:
                res.append(odd_dic.pop())
            count += 1
        return res

A = Solution()
print(A.sortArrayByParityII([4,2,5,7]))
