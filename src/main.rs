use crate::common::{ListNode, TreeNode};

mod dp;
mod hash;
mod array;
mod common;
mod simple;
mod recursion;
mod math_method;
mod list_or_tree;
mod bit_operator;
mod stack_or_queue;
mod written_test_collect;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_array_search() {
        assert_eq!(true, array::search([2, 5, 6, 0, 0, 1, 2].to_vec(), 0));
        assert_eq!(false, array::search([2, 5, 6, 0, 0, 1, 2].to_vec(), 3));
        assert_eq!(true, array::search([0, 1, 1, 1].to_vec(), 0));
        assert_eq!(true, array::search([0, 1, 1, 1].to_vec(), 1));
        assert_eq!(false, array::search([0, 1, 1, 1].to_vec(), 2));
        assert_eq!(true, array::search([1, 1, 0, 1, 1].to_vec(), 0));
        assert_eq!(false, array::search([1, 1, 0, 1, 1].to_vec(), 2));
        assert_eq!(true, array::search([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1].to_vec(), 2));
    }

    #[test]
    fn t_array_find_min() {
        assert_eq!(1, array::find_min([3, 4, 5, 1, 2].to_vec()));
        assert_eq!(0, array::find_min([4, 5, 6, 7, 0, 1, 2].to_vec()));
        assert_eq!(11, array::find_min([11, 13, 15, 17].to_vec()));
        assert_eq!(11, array::find_min([13, 15, 17, 11].to_vec()));
        assert_eq!(1, array::find_min([3, 1, 2].to_vec()));
    }
}


fn main() {
    // 131. 分割回文串
    let s = "aab".to_string();
    let r = recursion::partition(s);
    println!("r: {:?}", r);

    // 132. 分割回文串 II [✔]
    let s = "aaasss".to_string();
    let r = recursion::min_cut(s);
    println!("r: {:?}", r);

    // 132. 分割回文串 II [✔]
    let s = "fifgbeajcacehiicccfecbfhhgfiiecdcjjffbghdidbhbdbfbfjccgbbdcjheccfbhafehieabbdfeigbiaggchaeghaijfbjhi".to_string();
    let r = dp::min_cut(s);
    println!("r: {:?}", r);

    // 1047. 删除字符串中的所有相邻重复项 [✔]
    let s = "abcc".to_string();
    let _ = array::remove_duplicates(s.clone());
    let r = stack_or_queue::remove_duplicates(s);
    println!("r: {:?}", r);

    // 224. 基本计算器
    let s = "(1+(4+5+2)-3)+(6+8)".to_string();
    let _ = recursion::calculate(s.clone());
    let r = stack_or_queue::calculate(s);
    println!("r: {:?}", r);

    // 227. 基本计算器 II [✔]
    let s = "  42 ".to_string();
    let r = stack_or_queue::calculate_2(s);
    println!("r: {:?}", r);

    // 1143. 最长公共子序列
    let text1 = "abxxc".to_string();
    let text2 = "defx".to_string();
    let r = dp::longest_common_subsequence(text1, text2);
    println!("r: {:?}", r);

    // 331. 验证二叉树的前序序列化
    let s = "#,#,#".to_string();
    let r = stack_or_queue::is_valid_serialization(s);
    println!("r: {:?}", r);

    // 705. 设计哈希集合
    let mut my_hashset = hash::MyHashSet::new();
    my_hashset.add(12);
    my_hashset.remove(1);
    let _ = my_hashset.contains(12);
    my_hashset.remove(12);
    let _ = my_hashset.contains(12);

    // 706. 设计哈希映射
    let mut my_hashmap = hash::MyHashMap::new();
    my_hashmap.put(0, 1);
    my_hashmap.remove(0);
    my_hashmap.get(0);

    // 54. 螺旋矩阵
    let matrix = [[1], [2]].to_vec();
    let matrix = matrix.into_iter().map(|r| { r.to_vec() }).collect::<Vec<Vec<i32>>>();
    let r = array::spiral_order(matrix);
    println!("r: {:?}", r);

    // 59. 螺旋矩阵 II
    let n = 1;
    let r = array::generate_matrix(n);
    println!("r: {:?}", r);

    // 72. 编辑距离
    let word1 = "123".to_string();
    let word2 = "456".to_string();
    let _ = written_test_collect::min_distance(word1, word2);

    // 115. 不同的子序列
    let s = "rabbbit".to_string();
    let t = "rabbit".to_string();
    // 深搜超时！！！
    let _ = recursion::num_distinct(s.clone(), t.clone());
    let r = dp::num_distinct(s, t);
    println!("115. r: {:?}", r);

    // 92. 反转链表 II
    let _ = ListNode::new(10);
    let _ = TreeNode::new(10);
    let _ = list_or_tree::reverse_between(None, 0, 0);

    // 523. 连续的子数组和
    let nums = [23, 2, 6, 4, 7].to_vec();
    let k = 22;
    let r = written_test_collect::check_subarray_sum(nums, k);
    println!("r: {:?}", r);

    // 523. 连续的子数组和
    let nums = [0, 0].to_vec();
    let k = 0;
    let r = dp::check_subarray_sum(nums, k);
    println!("r: {:?}", r);

    // 525. 连续数组
    let nums = [0, 1, 0, 1].to_vec();
    let r = hash::find_max_length(nums);
    println!("r: {:?}", r);

    // 98. 验证二叉搜索树
    let _ = list_or_tree::is_valid_bst1(None);
    let r = list_or_tree::is_valid_bst2(None);
    println!("r: {:?}", r);

    // 面试题 17.21. 直方图的水量
    let height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1].to_vec();
    let r = array::trap(height);
    println!("r: {}", r);

    // 912. 排序数组 [快速排序]
    let nums = [2, 1, 3].to_vec();
    let r = written_test_collect::sort_array1(nums);
    println!("r: {:?}", r);

    // 912. 排序数组 [归并排序]
    let nums = [2, 1, 3].to_vec();
    let r = written_test_collect::sort_array2(nums);
    println!("r: {:?}", r);

    // 11. 盛最多水的容
    let height = [1, 2, 1].to_vec();
    let r = array::max_area(height);
    println!("r: {:?}", r);

    // 124. 二叉树中的最大路径和
    let _ = list_or_tree::max_path_sum(None);

    // 150. 逆波兰表达式求值
    let tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"].to_vec();
    let tokens = tokens.into_iter().map(|t| { t.to_string() }).collect::<Vec<String>>();
    let r = stack_or_queue::eval_rpn(tokens);
    println!("r: {:?}", r);

    // 503. 下一个更大元素 II
    let nums = vec![1, 2, 1];
    let r = stack_or_queue::next_greater_elements(nums);
    println!("r: {:?}", r);

    // 101. 对称二叉树
    let _ = list_or_tree::is_symmetric(None);

    // 73. 矩阵置零
    let matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]].to_vec();
    let mut matrix = matrix.into_iter().map(|r| { r.to_vec() }).collect::<Vec<Vec<i32>>>();
    array::set_zeroes(&mut matrix);
    drop(matrix);

    // 338. 比特位计数
    let num = 5;
    let r = bit_operator::count_bits(num);
    println!("r: {:?}", r);

    // 300. 最长递增子序列 [✔]
    let nums = [7, 7, 7, 7, 7, 7, 7].to_vec();
    let r = dp::length_of_lis(nums);
    println!("r: {:?}", r);

    // 354. 俄罗斯套娃信封问题
    let envelopes = [[5, 4], [6, 4], [6, 7], [2, 3]].to_vec();
    let envelopes = envelopes.into_iter().map(|r| { r.to_vec() }).collect();
    let r = dp::max_envelopes(envelopes);
    println!("r: {:?}", r);

    // 191. 位1的个数
    let n = 0b00000000000000000000000010000000u32;
    let r = bit_operator::hamming_weight(n);
    println!("r: {:?}", r);

    // 剑指 Offer 43. 1～n 整数中 1 出现的次数
    let n = 19;
    let r = dp::count_digit_one(n);
    println!("r: {:?}", r);

    // 456. 132 模式
    let nums = [-1, 3, 2, 0].to_vec();
    let r = array::find132pattern(nums);
    println!("r: {:?}", r);

    // 61. 旋转链表
    let head = [0, 1, 2].to_vec();
    let head = common::ListNode::deserialize(head);
    let r = list_or_tree::rotate_right(head, 4);
    let r = common::ListNode::serialize(r);
    println!("head: {:?}", r);

    // 696. 计数二进制子串
    let s = "10101".to_string();
    let r = array::count_binary_substrings(s);
    println!("r: {:?}", r);

    // 173. 二叉搜索树迭代器
    let mut bst = list_or_tree::BSTIterator::new(None);
    let _ = bst.has_next();
    let _ = bst.next();

    // 152. 乘积最大子数组 [✔]
    let nums = [-2, 0, -1].to_vec();
    let r = dp::max_product(nums);
    println!("r: {}", r);

    // 190. 颠倒二进制位
    let x = 0b00000010100101000001111010011100u32;
    let r = array::reverse_bits(x);
    println!("r: {:b}", r);

    // 74. 搜索二维矩阵
    let matrix = [[1, 3]].to_vec();
    let matrix = matrix.into_iter().map(|r| { r.to_vec() }).collect::<Vec<Vec<i32>>>();
    let r = array::search_matrix(matrix, 3);
    println!("r: {:?}", r);

    // 1006. 笨阶乘
    let n = 2;
    let r = math_method::clumsy(n);
    println!("r: {}", r);

    // 213. 打家劫舍 II
    let nums = [0].to_vec();
    let r = dp::rob2(nums);
    println!("r: {}", r);

    // 78. 子集
    let nums = [0].to_vec();
    let r = dp::subsets(nums);
    println!("r: {:?}", r);

    // 90.子集
    let nums = [1, 0, 0, 0].to_vec();
    let r = dp::subsets_with_dup(nums);
    println!("r: {:?}", r);

    // 88. 合并两个有序数组
    let mut nums1 = [0, 0, 0].to_vec();
    let mut nums2 = [1, 2, 3].to_vec();
    array::merge(&mut nums1, 0, &mut nums2, 3);
    println!("r: {:?}", nums1);

    // 80. 删除有序数组中的重复项 II
    let mut nums = [0, 0, 1, 1, 1, 1, 2, 3, 3].to_vec();
    let r = array::remove_duplicates2(&mut nums);
    println!("r: {:?} | nums: {:?}", r, nums);

    // 81. 搜索旋转排序数组 II
    let nums = [0, 1, 1, 1].to_vec();
    let target = 0;
    let r = array::search(nums, target);
    println!("r: {:?}", r);

    // 153. 寻找旋转排序数组中的最小值
    let nums = [3, 1, 2].to_vec();
    let r = array::find_min(nums);
    println!("r: {:?}", r);

    // 91. 解码方法
    let s = "2101".to_string();
    let r = dp::num_decodings(s);
    println!("r: {:?}", r);

    // 783. 二叉搜索树节点最小距离
    let _ = list_or_tree::min_diff_in_bst(None);

    // 1486. 数组异或操作
    let n = 2;
    let start = 5;
    let r = simple::xor_operation(n, start);
    println!("r: {:?}", r);

    // 740. 删除并获得点数
    let nums = [8, 7, 3, 8, 1, 4, 10, 10, 10, 2].to_vec();
    let r = dp::delete_and_earn(nums);
    println!("r: {:?}", r);

    // 525. 连续数组
    let nums = [1, 0, 1].to_vec();
    let r = array::find_max_length(nums);
    println!("r: {:?}", r);

    // 168. Excel表列名称
    let column_number = 52;
    let r = simple::convert_to_title(column_number);
    println!("r: {:?}", r);

    // 198. 打家劫舍
    let nums = [2, 17, 9, 3, 1].to_vec();
    let r = dp::rob(nums);
    println!("r: {:?}", r);

    // 912. 排序数组
    let nums = [5, 1, 1, 2, 0, 0].to_vec();
    let r = array::sort_array(nums);
    println!("r: {:?}", r);

    // 275. H 指数 II
    let citations = [0].to_vec();
    let r = array::h_index(citations);
    println!("r: {:?}", r);
}

