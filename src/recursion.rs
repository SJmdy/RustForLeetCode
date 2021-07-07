//! 使用递归 / 回溯解决的题目


use std::collections::LinkedList;

/// 131. 分割回文串 [✔]
///
/// 给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。
///
/// LC: [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/solution/hui-su-you-hua-jia-liao-dong-tai-gui-hua-by-liweiw/
///
/// 解题思路：回溯剪枝 [✔] 动态规划 []
///
/// 从当前的字符串s开始，若s[..i]是回文串，则将s加入到path中去，并继续对剩下的字符串s[i..]进行分割
/// 当当前字符串s为空时，将path加入到res中去。
pub fn partition(s: String) -> Vec<Vec<String>> {
    if s.is_empty() { return Vec::new(); }

    let mut res = Vec::new();
    let mut path = Vec::new();

    fn dfs(s: String, path: &mut Vec<String>, res: &mut Vec<Vec<String>>) {
        if s.is_empty() {
            res.push(path.clone());
        }

        for i in 1..s.len() + 1 {
            let front = &s[..i];
            if front.eq(front.chars().rev().collect::<String>().as_str()) {
                path.push(front.to_string());
                dfs(s[i..].to_string(), path, res);
                path.pop();
            }
        }
    }
    dfs(s, &mut path, &mut res);
    return res;
}


/// 132. 分割回文串 II [✔]
///
/// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。返回符合要求的 最少分割次数 。
///
/// LC: [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
///
/// 解题思路：回溯剪枝 [✔] | 动态规划 []
///
/// 注意：回溯剪枝会超时
pub fn min_cut(s: String) -> i32 {
    fn dfs(s: String) -> i32 {
        if s.len() < 2 || s.eq(&s.chars().rev().collect::<String>()) {
            return 0;
        }

        let mut times = Vec::new();
        for i in 1..s.len() + 1 {
            let front = &s[..i];
            if front.eq(front.chars().rev().collect::<String>().as_str()) {
                times.push(dfs(s[i..].to_string()));
            }
        }
        return 1 + times.iter().min().unwrap();
    }
    return dfs(s);
}


/// 224. 基本计算器
///
/// 实现一个基本的计算器来计算一个简单的字符串表达式 s 的值。
///
/// 注：'s' must consist of values in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '(', ')', ' '] only
/// 注：此法Python会超时
///
/// LC: [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)
///
/// 解题思路：递归求解 [✔] | 栈
pub fn calculate(s: String) -> i32 {
    fn calcu(s: String) -> i32 {
        // s: 去掉' '后的字符串
        if s.is_empty() { return 0; }

        // (1+(4+5+2)-3)+(6+8)

        // nums[i]: 数字
        // ops[i]: 操作符 ops[i] == True: 加， False：减
        let chars = s.chars().collect::<Vec<char>>();
        let mut nums = Vec::new();
        let mut ops = Vec::new();

        if s.starts_with('+') || s.starts_with('-') {
            nums.push(0);
        }

        let mut left_cur = 0;

        while left_cur < chars.len() {
            match chars[left_cur] {
                '(' => {
                    let mut stack = LinkedList::new();
                    stack.push_back('(');
                    let mut bracket_cur = left_cur + 1;

                    // left_cur, bracket_cur 分别指向两个括号。
                    while bracket_cur < chars.len() {
                        match chars[bracket_cur] {
                            '(' => stack.push_back('('),
                            ')' => { stack.pop_back(); }
                            _ => {}
                        }
                        if stack.is_empty() {
                            break;
                        }
                        bracket_cur += 1;
                    }
                    nums.push(calcu(s[left_cur + 1..bracket_cur].to_string()));
                    left_cur = bracket_cur + 1;
                }
                '+' => {
                    ops.push(true);
                    left_cur += 1;
                }
                '-' => {
                    ops.push(false);
                    left_cur += 1;
                }
                ')' => { left_cur += 1; }
                // 数字
                _ => {
                    let mut digit_cur = left_cur + 1;
                    while digit_cur < chars.len() && chars[digit_cur].is_digit(10) {
                        digit_cur += 1;
                    }
                    let digit = i32::from_str_radix(&s[left_cur..digit_cur], 10).unwrap();
                    nums.push(digit);
                    left_cur = digit_cur;
                }
            };
        }

        let mut res = nums[0];
        for (idx, op) in ops.iter().enumerate() {
            match *op {
                true => res += nums[idx + 1],
                false => res -= nums[idx + 1]
            };
        }
        return res;
    }

    let s = s.chars().into_iter().filter(|c| { *c != ' ' }).collect::<String>();
    return calcu(s);
}


/// 115. 不同的子序列
///
/// 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
///
/// 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
///
/// 题目数据保证答案符合 32 位带符号整数范围。
///
/// LC: [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)
///
/// 解题思路：深搜 超时啦超时啦 [✔] | 动态规划
pub fn num_distinct(s: String, t: String) -> i32 {
    if s.is_empty() || t.is_empty() || s.len() < t.len() { return 0; }
    if s.len() == t.len() { return if s.eq(&t) { 1 } else { 0 }; }

    let s = s.chars().collect::<Vec<char>>();
    let t = t.chars().collect::<Vec<char>>();

    fn dfs(s: &Vec<char>, s_cur: usize, t: &Vec<char>, t_cur: usize, res: &mut i32) {
        if t_cur == t.len() {
            *res += 1;
            return;
        }
        if s_cur == s.len() { return; }

        for i in s_cur..s.len() {
            if s[i] == t[t_cur] {
                dfs(s, i + 1, t, t_cur + 1, res);
            }
        }
    }
    let mut res = 0;
    dfs(&s, 0, &t, 0, &mut res);
    return res;
}


