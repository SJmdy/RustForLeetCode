//! 使用动态规划解决的题目

use std::cmp::{max, min};
use std::collections::HashMap;

/// 132. 分割回文串 II [✔]
///
/// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。返回符合要求的 最少分割次数 。
///
/// LC: [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
///
/// 解题思路：回溯剪枝 [] | 动态规划 [✔]
///
/// dp[i]: s[0..i + 1]的最小切割数
///
/// dp[i] = 0 if 前i个字符是回文串
/// dp[i] = min(dp[j]) + 1，其中s[j + 1..i]是回文串
///
/// 使用is_partition辅助判断s[i..j + 1]是否是回文串
///
/// is_partition[i][j] = (is_partition[i + 1][j - 1]) && s[i] == s[j]
pub fn min_cut(s: String) -> i32 {
    if s.len() < 2 || s.eq(&s.chars().rev().collect::<String>()) {
        return 0;
    }

    let mut dp = vec![0; s.len()];
    let mut is_partition = vec![vec![false; s.len()]; s.len()];

    for i in 0..s.len() {
        is_partition[i][i] = true;
    }
    let chars = s.chars().collect::<Vec<char>>();
    for i in (0..chars.len()).rev() {
        for j in i + 1..chars.len() {
            is_partition[i][j] = if j == i + 1 {
                chars[i] == chars[j]
            } else {
                is_partition[i + 1][j - 1] && chars[i] == chars[j]
            }
        }
    }

    for i in 1..dp.len() {
        dp[i] = if is_partition[0][i] {
            0
        } else {
            let mut min_cuts = i32::max_value();
            for j in 0..i {
                // is_  partition[j][i]: s[j..i + 1]
                // dp[i]: s[0..i + 1]
                if is_partition[j + 1][i] {
                    if min_cuts > dp[j] {
                        min_cuts = dp[j];
                    }
                }
            }
            min_cuts + 1
        }
    }
    return dp[dp.len() - 1];
}


/// 1143. 最长公共子序列
///
/// 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
///
/// 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
///
/// 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
///
/// 若这两个字符串没有公共子序列，则返回 0。
///
/// LC: [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)
///
/// 解题思路：动态规划
///
/// ```text
/// dp[i][j]: text1[i]与text2[j]的最长公共子序列长度。
/// dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i - 1][j], dp[i][j - 1]), if text1[i] == text2[j];
/// dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]), if text1[i] != text2[j]
/// ```
pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
    if text1.is_empty() || text2.is_empty() { return 0; }
    let text1 = text1.chars().collect::<Vec<char>>();
    let text2 = text2.chars().collect::<Vec<char>>();
    let mut dp = vec![vec![0; text2.len()]; text1.len()];
    dp[0][0] = if text1[0] == text2[0] { 1 } else { 0 };

    for i in 1..text2.len() {
        dp[0][i] = if text2[i] == text1[0] { 1 } else { dp[0][i - 1] };
    }
    for i in 1..text1.len() {
        dp[i][0] = if text1[i] == text2[0] { 1 } else { dp[i - 1][0] };
    }

    for i in 1..text1.len() {
        for j in 1..text2.len() {
            dp[i][j] = if text1[i] == text2[j] {
                max(max(dp[i - 1][j - 1] + 1, dp[i - 1][j]), dp[i][j - 1])
            } else {
                max(dp[i][j - 1], dp[i - 1][j])
            }
        }
    }
    return dp[text1.len() - 1][text2.len() - 1];
}


/// 115. 不同的子序列 [✔]
///
/// 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
///
/// 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
///
/// 题目数据保证答案符合 32 位带符号整数范围。
///
/// LC: [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)
///
/// 解题思路：深搜 超时啦超时啦 | 动态规划 [✔]
///
/// dp[i][j]: s[0..=i]的子序列中出现t[0..=j]的个数。
///
/// 那么有：
///
/// 1. s[i] == t[j]，这时有两个选择：
///     (1) 使用s[i]匹配t[j]:       dp[i - 1][j - 1]
///     (2) 不使用s[i]匹配t[j]:      dp[i - 1][j]
///     因此再这种情况下，dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
/// 2. s[i] != t[j]，这种情况下，dp[i][j] = dp[i - 1][j];
pub fn num_distinct(s: String, t: String) -> i32 {
    if s.is_empty() || t.is_empty() || s.len() < t.len() { return 0; }
    if s.len() == t.len() { return if s.eq(&t) { 1 } else { 0 }; }

    let s = s.chars().collect::<Vec<char>>();
    let t = t.chars().collect::<Vec<char>>();
    let mut dp = vec![vec![0; t.len()]; s.len()];

    dp[0][0] = if s[0] == t[0] { 1 } else { 0 };
    for i in 1..s.len() {
        dp[i][0] = if s[i] == t[0] {
            dp[i - 1][0] + 1
        } else {
            dp[i - 1][0]
        }
    }

    for i in 1..s.len() {
        for j in 1..t.len() {
            dp[i][j] = if s[i] == t[j] {
                dp[i - 1][j] + dp[i - 1][j - 1]
            } else {
                dp[i - 1][j]
            }
        }
    }
    return dp[s.len() - 1][t.len() - 1];
}


/// 523. 连续的子数组和 [✔]
///
/// 给定一个包含 非负数 的数组和一个目标 整数 k ，编写一个函数来判断该数组是否含有连续的子数组，其大小至少为 2，且总和为 k 的倍数，即总和为 n * k ，其中 n 也是一个整数。
///
/// LC: [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)
///
/// 解题思路：暴力 | 动态规划-前缀和 [✔]
///
/// prefix_sum[i]: nums[0..i].sum()
pub fn check_subarray_sum(nums: Vec<i32>, k: i32) -> bool {
    if nums.len() < 2 { return false; }

    let mut prefix_sum = vec![0; nums.len() + 1];
    for i in 0..nums.len() {
        prefix_sum[i + 1] = prefix_sum[i] + nums[i];
    }
    for i in 0..prefix_sum.len() - 2 {
        for j in i + 2..prefix_sum.len() {
            let dis = prefix_sum[j] - prefix_sum[i];
            if (dis == k) || (k != 0 && dis % k == 0) {
                return true;
            }
        }
    }
    return false;
}


/// 300. 最长递增子序列 [✔]
///
/// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
///
/// 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
///
/// LC: [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
///
/// 解题思路：动态规划
///
/// dp[i]: 以**nums[i]为结尾字符**的最长严格递增子序列的长度。
///
/// 则有：
///
/// dp[i] = max(dp[j]) + 1, 0 <= j < i and nums[j] < nums[i]
pub fn length_of_lis(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }

    let mut dp = vec![1; nums.len()];

    for i in 1..dp.len() {
        for j in 0..i {
            if nums[i] > nums[j] {
                dp[i] = if dp[i] < dp[j] + 1 { dp[j] + 1 } else { dp[i] };
            }
        }
    }
    return dp.into_iter().max().unwrap();
}


/// 354. 俄罗斯套娃信封问题
///
/// 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
///
/// 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
///
/// 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
///
/// 注意：不允许旋转信封。
///
/// LC: [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)
///
/// 解题思路：
///
/// 以w为第一关键字进行升序排列，以h为第二关键字进行降序排列，求[h_i]的最长严格递增子序列
pub fn max_envelopes(envelopes: Vec<Vec<i32>>) -> i32 {
    if envelopes.is_empty() { return 0; }

    let mut envelopes = envelopes.into_iter().
        map(|r| { [r[0], r[1]] }).collect::<Vec<[i32; 2]>>();

    envelopes.sort_by(|a, b| {
        if a[0] == b[0] {
            b[1].cmp(&a[1])
        } else {
            a[0].cmp(&b[0])
        }
    });

    let nums = envelopes.into_iter().map(|r| { r[1] }).collect::<Vec<i32>>();
    let mut dp = vec![1; nums.len()];
    for i in 1..dp.len() {
        for j in 0..i {
            if nums[i] > nums[j] {
                if dp[i] < dp[j] + 1 {
                    dp[i] = dp[j] + 1;
                }
            }
        }
    }
    return dp.into_iter().max().unwrap();
}


/// 剑指 Offer 43. 1～n 整数中 1 出现的次数
///
/// 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。1 <= n < 2^31
///
/// 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
///
/// LC: [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)
pub fn count_digit_one(n: i32) -> i32 {
    return n;
}


/// 152. 乘积最大子数组 [✔]
///
/// 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
///
/// LC: [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)
/// 解题思路：动态规划
///
/// 1. max_product: 当前的最大值，始终大于0
/// 2. min_product: 当前的最小值，可能小于0
/// 3. res: 最终的最大值
///
/// 当当前的nums[i] < 0的时候，需要交换max_product与min_product
pub fn max_product(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }
    // max_product可能为负，但不影响结果。
    let (mut max_product, mut min_product, mut res) = (nums[0], nums[0], nums[0]);

    for i in 1..nums.len() {
        if nums[i] < 0 {
            std::mem::swap(&mut max_product, &mut min_product);
        }
        max_product = max(max_product * nums[i], nums[i]);
        min_product = min(min_product * nums[i], nums[i]);
        res = max(res, max_product);
    }
    return res;
}


/// 213. 打家劫舍 II [✔]
///
/// 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
///
/// 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
///
/// LC: [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/description/)
///
/// 解题思路：动态规划
///
/// 共两种情况：
///
/// 1. 取第一个屋子，那么结果为 nums[0] + Result(nums[2..])
/// 2. 不取第一个物资，那么结果为 Result(nums[1..nums.len() - 1])
///
/// dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
pub fn rob2(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }
    if nums.len() < 4 { return nums.into_iter().max().unwrap(); }

    let mut dp = vec![0; nums.len()];
    dp[1] = nums[1];
    for i in 2..dp.len() {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    let res = dp[dp.len() - 1];
    let mut dp = vec![0; nums.len()];
    dp[2] = nums[2];
    for i in 3..dp.len() - 1 {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return max(res, nums[0] + dp[dp.len() - 2]);
}


/// 78. 子集
///
/// 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
///
/// LC: [78. 子集](https://leetcode-cn.com/problems/subsets/)
pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {
    if nums.is_empty() { return Vec::new(); }

    let mut res = vec![Vec::new(), vec![nums[0]]];

    for i in 1..nums.len() {
        let mut new_subsets = Vec::new();
        for s in res.iter() {
            let mut t = s.clone();
            t.push(nums[i]);
            new_subsets.push(t);
        }
        res.append(&mut new_subsets);
    }
    return res;
}


/// 90. 子集 II
///
/// 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
///
/// LC: [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)
pub fn subsets_with_dup(nums: Vec<i32>) -> Vec<Vec<i32>> {
    if nums.is_empty() { return Vec::new(); }
    let mut nums = nums;
    nums.sort();

    let mut res = vec![Vec::new(), vec![nums[0]]];
    let mut cur = 1;
    for i in 1..nums.len() {
        if nums[i] != nums[i - 1] {
            let mut new_subsets = Vec::new();
            for s in res.iter() {
                let mut t = s.clone();
                t.push(nums[i]);
                new_subsets.push(t);
            }
            cur = res.len();
            res.append(&mut new_subsets);
        } else {
            let mut new_subsets = Vec::new();
            for s in res[cur..].iter() {
                let mut t = s.clone();
                t.push(nums[i]);
                new_subsets.push(t);
            }
            cur = res.len();
            res.append(&mut new_subsets)
        }
    }
    return res;
}


/// 91. 解码方法
///
/// 一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
///
/// 'A' -> 1 | 'B' -> 2 ... 'Z' -> 26
///
/// 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。
///
/// LC: [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)
pub fn num_decodings(s: String) -> i32 {
    if s.starts_with('0') { return 0; }
    if s.len() < 2 { return 1; }
    if s.contains("00") { return 0; }
    let nums = s.chars().into_iter().map(|c| {
        c.to_digit(10).unwrap()
    }).collect::<Vec<u32>>();

    let mut dp = vec![0; nums.len()];
    dp[0] = 1;
    for i in 1..dp.len() {
        if nums[i] == 0 {
            if i == 1 {
                dp[i] = dp[i - 1];
            } else {
                dp[i] = dp[i - 2];
            }
        } else {
            if nums[i - 1] != 0 && nums[i - 1] * 10 + nums[i] < 27 {
                if i == 1 {
                    dp[i] = dp[i - 1] + 1;
                } else {
                    dp[i] = dp[i - 1] + dp[i - 2];
                }
            } else {
                dp[i] = dp[i - 1];
            }
        }
    }
    return dp[dp.len() - 1] as i32;
}


/// 740. 删除并获得点数
///
/// 给你一个整数数组 nums ，你可以对它进行一些操作。 每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
///
/// LC: [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/)
pub fn delete_and_earn(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }

    let mut n_sum = HashMap::new();
    for n in nums.iter() {
        let entity = n_sum.entry(*n).or_insert(0);
        *entity += n;
    }

    let max_number = *nums.iter().max().unwrap() as usize;
    if max_number == 0 { return 0; }

    let mut dp = vec![0; max_number + 1];
    dp[1] = *n_sum.get(&1).unwrap_or(&0);

    for i in 2..dp.len() {
        dp[i] = max(dp[i - 1], dp[i - 2] + *n_sum.get(&(i as i32)).unwrap_or(&0));
    }
    return dp[max_number];
}


/// 523. 连续的子数组
///
/// 给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：子数组大小 至少为 2 ，且子数组元素总和为 k 的倍数。如果存在，返回 true ；否则，返回 false 。如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。
/// LC: [523. 连续的子数组](https://leetcode-cn.com/problems/continuous-subarray-sum/)
pub fn check_subarray_sum2(nums: Vec<i32>, k: i32) -> bool {
    if nums.len() < 2 { return false; }
    if k == 0 { return false; }
    if k == 1 { return true; }

    let mut prefix_sum = vec![0; nums.len() + 1];
    // prefix_sum[i] = nums[..i].sum()

    for i in 1..prefix_sum.len() {
        prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1];
    }

    for i in 0..prefix_sum.len() {
        for j in i + 2..prefix_sum.len() {
            if prefix_sum[j] - prefix_sum[i] % k == 0 {
                return true;
            }
        }
    }
    return false;
}