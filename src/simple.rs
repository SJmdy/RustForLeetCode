/// 1486. 数组异或操作
///
/// 给你两个整数，n 和 start 。
// 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且 n == nums.length 。
// 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
///
/// LC: [1486. 数组异或操作](https://leetcode-cn.com/problems/xor-operation-in-an-array/)
pub fn xor_operation(n: i32, start: i32) -> i32 {
    if n == 0 { return 0; }

    let mut res = start;
    for i in 1..n {
        res ^= start + 2 * i;
    }
    return res;
}


/// [168. Excel表列名称](https://leetcode-cn.com/problems/excel-sheet-column-title/)
///
/// 给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。
pub fn convert_to_title(column_number: i32) -> String {
    let mut s = String::new();
    let mut column_number = column_number;
    while column_number > 0 {
        column_number -= 1;
        s.push(char::from((column_number as u8 % 26) + 65));
        column_number /= 26;
    }
    return s.chars().rev().collect::<String>();
}