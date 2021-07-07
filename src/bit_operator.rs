//! 使用位运算解决的问题

/// 338. 比特位计数
///
/// 给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。
///
/// LC: [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)
///
/// 解题思路：偶数的比特位的最后一位为0，奇数为1，因此若num时奇数，那么它的比特位1的数量就是前一个数加1；
/// 偶数的比特位数量则与它的一半相同。
pub fn count_bits(num: i32) -> Vec<i32> {
    let num = num as usize;
    let mut dp = vec![0; num + 1];
    for i in 1..dp.len() {
        dp[i] = if i & 1 == 0 {
            // 偶数
            dp[i / 2]
        } else {
            // 奇数
            dp[i - 1] + 1
        }
    }
    return dp;
}


/// 191. 位1的个数
///
/// 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。
///
/// 提示：
///
/// 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
///
/// 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
///
/// LC: [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)
pub fn hamming_weight (n: u32) -> i32 {
    let mut res = 0;
    let mut n = n;
    while n != 0 {
        res += n & 1;
        n = n >> 1;
    }
    return res as i32;
}
