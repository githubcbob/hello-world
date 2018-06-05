

class Solution:
    def removeDuplicates(self,nums):
        for i in range(min(nums),max(nums)+1):
            while nums.count(i) != 1:
                nums.remove(i)
        nums.sort()
        print(nums)
        return max(nums)+1-min(nums)
