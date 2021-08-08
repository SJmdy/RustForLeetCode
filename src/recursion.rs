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


/// [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
///
/// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。返回符合要求的 最少分割次数 。
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


/// [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)
///
/// 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
///
/// 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）。 题目数据保证答案符合 32 位带符号整数范围。
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


/// [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/))
///
/// 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
    if word.is_empty() { return true; }
    let rows = board.len();
    if rows == 0 { return false; }
    let cols = board[0].len();
    if cols == 0 { return false; }

    fn dfs(board: &Vec<Vec<char>>, x: i32, y: i32, mask: &mut Vec<Vec<bool>>, word: &Vec<char>, cur: usize) -> bool {
        if cur >= word.len() {
            return true;
        }

        let offset = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        for [ox, oy] in offset.iter() {
            let (nx, ny) = (x + *ox, y + *oy);
            if nx < 0 || ny < 0 { continue; }
            let (nx, ny) = (nx as usize, ny as usize);
            if nx >= board.len() || ny >= board[0].len() { continue; }

            if !mask[nx][ny] && board[nx][ny] == word[cur] {
                mask[nx][ny] = true;
                if dfs(board, nx as i32, ny as i32, mask, word, cur + 1) {
                    return true;
                }
                mask[nx][ny] = false;
            }
        }
        return false;
    }

    let word = word.chars().collect::<Vec<char>>();
    let mut mask = vec![vec![false; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            if board[i][j] == word[0] {
                mask[i][j] = true;
                if dfs(&board, i as i32, j as i32, &mut mask, &word, 1) { return true; }
                mask[i][j] = false;
            }
        }
    }
    return false;
}


/// 剑指 Offer 13. 机器人的运动范围
///
/// 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
pub fn moving_count(m: i32, n: i32, k: i32) -> i32 {
    if k < 0 { return 0; }
    if m == 0 { return 0; }
    if n == 0 { return 0; }

    let (m, n) = (m as usize, n as usize);
    let mut board = vec![vec![0; n]; m];

    fn digit_sum(n: usize) -> i32 {
        let mut n = n;
        let mut r = 0;
        while n > 0 {
            r += n % 10;
            n /= 10;
        }
        return r as i32;
    }

    for i in 0..m {
        for j in 0..n {
            board[i][j] = digit_sum(i) + digit_sum(j);
        }
    }

    let mut res = 1;
    let offset = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    let mut mask = vec![vec![false; n]; m];

    let mut path = LinkedList::new();
    path.push_back([0, 0]);
    mask[0][0] = true;

    while !path.is_empty() {
        let [x, y] = path.pop_back().unwrap();
        for [ox, oy] in offset.iter() {
            let (nx, ny) = (x + *ox, y + *oy);
            if nx < 0 || ny < 0 || nx >= m as i32 || ny >= n as i32 || mask[nx as usize][ny as usize] || board[nx as usize][ny as usize] > k { continue; }
            mask[nx as usize][ny as usize] = true;
            path.push_back([nx, ny]);
            res += 1;
        }
    }
    return res;
}
