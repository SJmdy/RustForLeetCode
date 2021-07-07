//! 记录一些在笔试（或面试的笔试）中出现的题

use std::cmp::min;

/// 912. 排序数组 [快速排序]
///
/// 给你一个整数数组 nums，请你将该数组升序排列
///
/// LC: [912. 排序数组（快速排序）](https://leetcode-cn.com/problems/sort-an-array/)
///
/// 解题思路：快速排序 [✔] | 归并排序
pub fn sort_array1(nums: Vec<i32>) -> Vec<i32> {
    if nums.len() < 2 { return nums; }

    fn quick_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
        if start >= end || start >= nums.len() {
            return;
        }

        let pivot = nums[start];
        let (mut left_cur, mut right_cur) = (start, end);

        while left_cur < right_cur {
            while left_cur < right_cur && nums[right_cur] > pivot {
                right_cur -= 1;
            }
            while left_cur < right_cur && nums[left_cur] <= pivot {
                left_cur += 1;
            }
            let temp = nums[left_cur];
            nums[left_cur] = nums[right_cur];
            nums[right_cur] = temp;
        }
        nums[start] = nums[left_cur];
        nums[left_cur] = pivot;

        if left_cur > 0 {
            quick_sort(nums, start, left_cur - 1);
        }
        quick_sort(nums, left_cur + 1, end);
    }

    let mut nums = nums;
    let length = nums.len();
    quick_sort(&mut nums, 0, length - 1);
    return nums;
}


/// 912. 排序数组
///
/// 给你一个整数数组 nums，请你将该数组升序排列。
///
/// LC: [912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)
///
/// 解题思路：快速排序 | 归并排序 [✔]
pub fn sort_array2(nums: Vec<i32>) -> Vec<i32> {
    fn merge_sort(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        if nums.len() == 1 { return nums; }
        if nums.len() == 2 {
            if nums[0] > nums[1] {
                let temp = nums[0];
                nums[0] = nums[1];
                nums[1] = temp;
            }
            return nums;
        }

        let middle = (nums.len() + 1) / 2;
        let mut temp = vec![0; nums.len()];

        let left = merge_sort(nums[..middle].to_vec());
        let right = merge_sort(nums[middle..].to_vec());

        let (mut cur, mut cur_left, mut cur_right) = (0, 0, 0);

        while cur < temp.len() {
            if cur_left == left.len() {
                temp[cur] = right[cur_right];
                cur_right += 1;
            } else if cur_right == right.len() {
                temp[cur]= left[cur_left];
                cur_left += 1;
            } else {
                if left[cur_left] < right[cur_right] {
                    temp[cur] = left[cur_left];
                    cur_left += 1;
                } else {
                    temp[cur] = right[cur_right];
                    cur_right += 1;
                }
            }
            cur += 1;
        }
        return temp;
    }

    return merge_sort(nums);
}


/// 72. 编辑距离
///
/// 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
///
/// 你可以对一个单词进行如下三种操作：插入一个字符、删除一个字符、替换一个字符
///
/// LC: [编辑距离](https://leetcode-cn.com/problems/edit-distance/)
pub fn min_distance(word1: String, word2: String) -> i32 {
    let m = word1.len();
    let n = word2.len();
    if m == 0 { return n as i32; };
    if n == 0 { return m as i32; };

    let mut dp = vec![vec![0; n + 1]; m + 1];
    for i in 1..m + 1 {
        dp[i][0] = i;
    }
    for i in 1..n + 1 {
        dp[0][i] = i;
    }

    let word1 = word1.chars().collect::<Vec<char>>();
    let word2 = word2.chars().collect::<Vec<char>>();
    for i in 1..m + 1 {
        for j in 1..n + 1 {
            dp[i][j] = if word1[i - 1] == word2[j - 1] {
                dp[i - 1][j - 1]
            } else {
                min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1
            }
        }
    }

    return dp[m][n] as i32;
}


/// 523. 连续的子数组和 [✔]
///
/// 给定一个包含 非负数 的数组和一个目标 整数 k ，编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，且总和为 k 的倍数，即总和为 n * k ，其中 n 也是一个整数。
///
/// LC: [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)
///
/// 注：这么精妙的解法我是怎么想出来的
pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
    if nums.len() < 2 { return false; }
    let mut nums = nums;

    for i in 0..nums.len() {
        for j in 0..i {
            nums[j] += nums[i];
            if (k != 0 && nums[j] % k == 0) || (k == 0 && nums[j] == 0) {
                return true;
            }
        }
    }
    return false;
}