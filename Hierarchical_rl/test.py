class Solution(object):
    def findSubsequences(self, nums):
        res = []
        mem = {}
        def dfs(target, cur_ls):
            for idx, num in enumerate(target):
                if num >= cur_ls[-1]:
                    if mem.get(tuple(cur_ls + [num]), 0) == 0:
                        res.append(cur_ls + [num])
                        mem[tuple(cur_ls + [num])] = 1
                    dfs(target[idx + 1:], cur_ls + [num])
        for idx in range(len(nums)):
            dfs(nums[idx + 1:], [nums[idx]])
        return res
            

A = Solution()
print(A.findSubsequences([4, 6, 7, 7]))