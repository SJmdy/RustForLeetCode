//! 与数组有关的问题，如滑窗法等


use std::cmp::min;
use std::collections::{LinkedList, HashMap};

/// 54. 螺旋矩阵
///
/// 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
///
/// LC: [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)
pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    if matrix.is_empty() { return Vec::new(); }

    let row = matrix.len();
    let col = matrix[0].len();
    if col == 0 { return Vec::new(); }

    let layers = (min(row, col) + 1) / 2;

    let mut layer = 0;
    let mut res = Vec::new();
    while layer < layers {
        // (layer, layer)[包含] -> (layer, col - layer - 1)[包含]
        for i in layer..=col - layer - 1 {
            res.push(matrix[layer][i]);
        }

        // (layer, col - layer - 1)[不包含] -> (row - layer - 1, col - layer - 1)[包含]
        for i in layer + 1..=row - layer - 1 {
            res.push(matrix[i][col - layer - 1]);
        }

        // (row - layer - 1, col - layer - 1)[不包含] -> (row - layer - 1, layer)[不包含]
        if row - layer - 1 == layer {
            // 防止重复
            break;
        }
        for i in (layer + 1..col - layer - 1).rev() {
            res.push(matrix[row - layer - 1][i]);
        }

        // (row - layer - 1, layer)[包含] -> (layer, layer)[不包含]
        if col - layer - 1 == layer {
            // 防止重复
            break;
        }
        for i in (layer + 1..=row - layer - 1).rev() {
            res.push(matrix[i][layer]);
        }
        layer += 1;
    }
    return res;
}


/// 59. 螺旋矩阵 II
///
/// 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
///
/// LC: [](https://leetcode-cn.com/problems/spiral-matrix-ii/)
pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
    if n == 0 {
        return Vec::new();
    }

    let n = n as usize;

    let mut matrix = vec![vec![0; n]; n];
    let mut num = 1;
    let (row, col) = (n, n);

    let layers = (n + 1) / 2;
    for layer in 0..layers {
        // (layer, layer)[包含] -> (layer, col - layer - 1)[包含]
        for i in layer..=col - layer - 1 {
            matrix[layer][i] = num;
            num += 1;
        }

        // (layer, col - layer - 1)[不包含] -> (row - layer - 1, col - layer - 1)[包含]
        for i in layer + 1..=row - layer - 1 {
            matrix[i][col - layer - 1] = num;
            num += 1;
        }

        // (row - layer - 1, col - layer - 1)[不包含] -> (row - layer - 1, layer)[包含]
        if row - layer - 1 == layer {
            break;
        }
        for i in (layer..col - layer - 1).rev() {
            matrix[row - layer - 1][i] = num;
            num += 1;
        }

        // (row - layer - 1, layer)[不包含] -> (layer, layer)[包含]
        if col - layer - 1 == layer {
            break;
        }
        for i in (layer + 1..row - layer - 1).rev() {
            matrix[i][layer] = num;
            num += 1;
        }
    }

    return matrix;
}


/// 面试题 17.21. 直方图的水量
///
/// 给定一个直方图(也称柱状图)，假设有人从上面源源不断地倒水，最后直方图能存多少水量?直方图的宽度为 1。
///
/// LC: [面试题 17.21 直方图的水量](https://leetcode-cn.com/problems/volume-of-histogram-lcci/)
///
/// 解题思路：双指针
///
/// 从某个不为零的值对应的坐标出发，记作left_cur，记下一个索引right_cur应该满足以下条件（或）：
///
/// (1) height[right_cur] >= height[left_cur]，且取满足条件的right_cur的最小值
///
/// (2) height[right_cur] < height[left_cur] && height[right_cur] != 0，且取满足条件的height[right_cur]
/// 的最大值。
///
/// 当(1)无法满足时，选取(2)，若(2)也无法满足，则终止
pub fn trap(height: Vec<i32>) -> i32 {
    if height.len() < 2 { return 0; }

    let mut res = 0;
    let mut cur = 0;

    while cur < height.len() {
        if height[cur] == 0 {
            cur += 1;
            continue;
        }

        // height[cur] != 0
        let mut right_cur = cur + 1;
        let mut highest_cur = cur;

        while right_cur < height.len() {
            if height[right_cur] > 0 {
                if height[right_cur] > height[cur] {
                    highest_cur = right_cur;
                    break;
                }
                if highest_cur == cur || height[highest_cur] <= height[right_cur] {
                    highest_cur = right_cur;
                }
            }
            right_cur += 1;
        }
        if highest_cur != cur {
            res += ((highest_cur - cur) as i32 - 1) * min(height[cur], height[highest_cur]) - height[cur + 1..highest_cur].iter().sum::<i32>();
            cur = highest_cur;
        } else {
            break;
        }
    }
    return res;
}


/// 11. 盛最多水的容器
///
/// 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
///
/// 说明：你不能倾斜容器。
///
/// LC: [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)
///
/// 解题思路：双指针
pub fn max_area(height: Vec<i32>) -> i32 {
    if height.len() < 2 { return 0; }

    let (mut left_cur, mut right_cur) = (0, height.len() - 1);
    let mut res = 0;

    while left_cur < right_cur {
        let now = (right_cur - left_cur) as i32 * min(height[left_cur], height[right_cur]);
        if res < now {
            res = now;
        }
        if height[left_cur] < height[right_cur] {
            left_cur += 1;
        } else {
            right_cur -= 1;
        }
    }

    return res;
}


/// 73. 矩阵置零
///
///
/// 给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。
///
/// LC: [73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)
pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    if matrix.is_empty() { return; }
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    let m = matrix.len();
    let n = matrix[0].len();

    for (i, row) in matrix.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if *v == 0 {
                rows.push(i);
                cols.push(j);
            }
        }
    }

    for row in rows.iter() {
        for i in 0..n {
            matrix[*row][i] = 0;
        }
    }

    for col in cols.iter() {
        for i in 0..m {
            matrix[i][*col] = 0;
        }
    }
}


/// 456. 132 模式
///
/// 给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。
///
/// 如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。
///
/// 进阶：很容易想到时间复杂度为 O(n^2) 的解决方案，你可以设计一个时间复杂度为 O(n logn) 或 O(n) 的解决方案吗？
pub fn find132pattern(nums: Vec<i32>) -> bool {
    if nums.len() < 3 { return false; }

    let mut left_min = vec![0; nums.len()];
    // left_min[i]: nums[0..i + 1]中最小的元素
    left_min[0] = nums[0];

    for i in 1..nums.len() {
        left_min[i] = min(left_min[i - 1], nums[i]);
    }

    // 单调栈：栈底元素最大
    let mut stack = LinkedList::new();

    for i in (0..nums.len()).rev() {
        let mut second_big = i32::min_value();
        while !stack.is_empty() && *stack.back().unwrap() < nums[i] {
            second_big = stack.pop_back().unwrap();
        }
        if left_min[i] < second_big {
            return true;
        }
        stack.push_back(nums[i]);
    }
    return false;
}


/// 1047. 删除字符串中的所有相邻重复项 []
///
/// 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。在 S 上反复执行重复项删除操作，直到无法继续删除。在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
///
/// LC: [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)
///
/// 解题思路：滑动窗口 [✔] | 栈 []
///
/// **注：**滑动窗口不是最佳解法，效率非常低
pub fn remove_duplicates(s: String) -> String {
    fn deal(s: String) -> String {
        if s.is_empty() { return s; }

        let chars = s.chars().collect::<Vec<char>>();
        let mut mask = vec![false; chars.len()];

        let (mut left_cur, mut right_cur) = (0, 1);

        while right_cur < chars.len() {
            if chars[left_cur] != chars[right_cur] {
                left_cur += 1;
                right_cur += 1;
            } else {
                mask[left_cur] = true;
                mask[right_cur] = true;
                left_cur = right_cur + 1;
                right_cur = left_cur + 1;
            }
        }

        let s = chars.into_iter().enumerate()
            .filter(|(idx, _)| { !mask[*idx] })
            .map(|(_, c)| { c.to_string() })
            .collect::<String>();
        return if mask.contains(&true) {
            deal(s)
        } else {
            s
        };
    }
    return deal(s);
}


/// 696. 计数二进制子串
///
/// 给定一个字符串 s，计算具有相同数量 0 和 1 的非空（连续）子字符串的数量，并且这些子字符串中的所有 0 和所有 1 都是连续的。
///
/// 重复出现的子串要计算它们出现的次数。
///
/// LC: [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/)
pub fn count_binary_substrings(s: String) -> i32 {
    if s.len() < 2 { return 0; }

    let chars = s.chars().collect::<Vec<char>>();

    let (mut start, mut cur) = (0, 0);
    let (mut zeros, mut ones) = (0, 0);
    let mut res = 0;

    while start < chars.len() {
        while cur < chars.len() && chars[cur] == chars[start] {
            cur += 1;
        }
        if chars[start] == '0' {
            zeros = cur - start;
        } else {
            ones = cur - start;
        }
        res += min(zeros, ones);
        start = cur;
    }

    return res as i32;
}

/// 190. 颠倒二进制位
///
/// 颠倒给定的 32 位无符号整数的二进制位。
///
/// LC: [190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/)
pub fn reverse_bits(x: u32) -> u32 {
    let x = format!("{:032b}", x);
    let mut x = x.chars().collect::<Vec<char>>();

    let (mut left_cur, mut right_cur) = (0, x.len() - 1);

    while left_cur < right_cur {
        x.swap(left_cur, right_cur);
        left_cur += 1;
        right_cur -= 1;
    }

    let x = x.into_iter().map(|c| { c }).collect::<String>();
    return u32::from_str_radix(x.as_str(), 2).unwrap();
}


/// 74. 搜索二维矩阵
///
/// 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
///
/// 每行中的整数从左到右按升序排列。
///
/// 每行的第一个整数大于前一行的最后一个整数。
///
/// LC: [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)
pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    if matrix.is_empty() { return false; }
    let m = matrix.len();
    let n = matrix[0].len();

    if target < matrix[0][0] || target > matrix[m - 1][n - 1] {
        return false;
    }

    let mut row: Option<usize> = None;
    for i in 0..m {
        if matrix[i][0] <= target {
            if matrix[i][0] == target {
                return true;
            }
            if i == m - 1 || matrix[i + 1][0] > target {
                row = Some(i);
                break;
            }
        }
    }

    if row.is_none() { return false; }
    let row = row.unwrap();

    for v in matrix[row].iter() {
        if *v == target {
            return true;
        }
    }
    return false;
}


/// 88. 合并两个有序数组
///
/// 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
///
/// 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。
///
/// LC: [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    if m == 0 {
        *nums1 = (*nums2).clone();
        return;
    }
    if n == 0 { return; }
    let (mut cur1, mut cur2, mut cur) = (m - 1, n - 1, nums1.len() as i32 - 1);

    loop {
        match (cur1 == -1, cur2 == -1) {
            (true, true) | (false, true) => break,
            (true, false) => {
                let (left, _) = nums1.split_at_mut(cur as usize + 1);
                left.copy_from_slice(&nums2[..cur2 as usize + 1]);
                break;
            }
            (false, false) => {
                if nums1[cur1 as usize] > nums2[cur2 as usize] {
                    nums1[cur as usize] = nums1[cur1 as usize];
                    cur1 -= 1;
                } else {
                    nums1[cur as usize] = nums2[cur2 as usize];
                    cur2 -= 1;
                };
                cur -= 1;
            }
        };
    };
}


/// 80. 删除有序数组中的重复项 II
///
/// 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
///
/// 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成
pub fn remove_duplicates2(nums: &mut Vec<i32>) -> i32 {
    if nums.len() < 3 { return nums.len() as i32; }

    let mut cur = 2;
    while cur < nums.len() {
        if nums[cur] == nums[cur - 2] {
            nums.remove(cur);
        } else {
            cur += 1;
        }
    }
    return nums.len() as i32;
}


/// [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)
///
/// 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
///
/// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
///
/// 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
pub fn search(nums: Vec<i32>, target: i32) -> bool {
    // return nums.contains(&target);
    if nums.is_empty() { return false; }

    let (mut left, mut right) = (0, nums.len() - 1);
    while left <= right {
        let mid = (left + right) / 2;
        if nums[mid] == target { return true; }
        if nums[mid] > nums[right] {
            left = mid + 1;
        } else {
            right -= 1;
        }
    }
    return false;
}


/// 153. 寻找旋转排序数组中的最小值
///
/// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
///
/// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2] 若旋转 4 次，则可以得到 [0,1,2,4,5,6,7]
///
/// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
///
/// 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
///
/// LC: [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)
pub fn find_min(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return -1; }
    let (mut left_cur, mut right_cur) = (0, nums.len() - 1);

    while left_cur < right_cur {
        let mid = (left_cur + right_cur) / 2;
        if nums[mid] < nums[nums.len() - 1] {
            right_cur = mid;
        } else {
            left_cur = mid + 1;
        }
    }
    return nums[left_cur];
}

/// [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)
///
/// 给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。
///
/// 示例 1: 输入: nums = [0,1] 输出: 2 说明: [0, 1] 是具有相同数量 0 和 1 的最长连续子数组。
///
/// 示例 2: 输入: nums = [0,1,0] 输出: 2 说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
pub fn find_max_length(nums: Vec<i32>) -> i32 {
    if nums.len() < 2 { return 0; }

    let mut res = 0;
    let mut count = 0;
    let mut count_index: HashMap<i32, usize> = std::collections::HashMap::new();
    for (idx, n) in nums.into_iter().enumerate() {
        count += if n == 1 { 1 } else { -1 };
        if count == 0 {
            res = idx + 1;
            continue;
        }
        match count_index.get(&count) {
            Some(index) => res = std::cmp::max(res, idx - *index),
            None => { count_index.insert(count, idx); }
        }
    }
    return res as i32;
}


// /// [1418. 点菜展示表](https://leetcode-cn.com/problems/display-table-of-food-orders-in-a-restaurant/)
// ///
// /// 给你一个数组 orders，表示客户在餐厅中完成的订单，确切地说， orders[i]=[customerNamei,tableNumberi,foodItemi] ，其中 customerNamei 是客户的姓名，tableNumberi 是客户所在餐桌的桌号，而 foodItemi 是客户点的餐品名称。
// ///
// /// 请你返回该餐厅的 点菜展示表 。在这张表中，表中第一行为标题，其第一列为餐桌桌号 “Table” ，后面每一列都是按字母顺序排列的餐品名称。接下来每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量
// ///
// /// 注意：客户姓名不是点菜展示表的一部分。此外，表中的数据行应该按餐桌桌号升序排列。
// pub fn display_table(orders: Vec<Vec<String>>) -> Vec<Vec<String>> {
//     if orders.is_empty() { return Vec::new(); }
//
//
//     return Vec::new();
// }

/// [912. 排序数组（快排）](https://leetcode-cn.com/problems/sort-an-array/)
///   给你一个整数数组 nums，请你将该数组升序排列。
///
///  示例 1：输入：nums = [5,2,3,1] 输出：[1,2,3,5]
///
///  示例 2：输入：nums = [5,1,1,2,0,0] 输出：[0,0,1,1,2,5]
pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
    if nums.len() < 2 { return nums; }
    let mut nums = nums;

    fn quick_sort(nums: &mut Vec<i32>, start: usize, end: usize) {
        if start >= end { return; }
        let (mut head, mut tail) = (start, end);
        while head < tail {
            while head < tail && nums[tail] > nums[start] {
                tail -= 1;
            }
            while head < tail && nums[head] <= nums[start] {
                head += 1;
            }
            // nums.swap(head, tail);
            let temp = nums[head];
            nums[head] = nums[tail];
            nums[tail] = temp;
        }
        // nums.swap(head, start);
        let temp = nums[start];
        nums[start] = nums[head];
        nums[head] = temp;
        if head != start {
            quick_sort(nums, start, head - 1);
        }
        quick_sort(nums, head + 1, end);
    }
    let length = nums.len();
    quick_sort(&mut nums, 0, length - 1);
    return nums;
}


/// 275. H 指数 II
///
/// 给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照 升序排列 。编写一个方法，计算出研究者的 h 指数。
///
/// h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。（其余的 N - h 篇论文每篇被引用次数不多于 h 次。）"
///
/// 示例: 输入: citations = [0,1,3,5,6] 输出: 3
/// 解释: 给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 0, 1, 3, 5, 6 次。 由于研究者有 3 篇论文每篇至少被引用了 3 次，其余两篇论文每篇被引用不多于 3 次，所以她的 h 指数是 3。
///
/// 设当前
pub fn h_index(citations: Vec<i32>) -> i32 {
    let mut index = -1;

    for (paper, cite) in citations.iter().enumerate() {
        if *cite as usize >= citations.len() - paper {
            index = paper as i32;
            break;
        }
    }
    return if index != -1 { citations.len() as i32 - index } else { 0 };
}