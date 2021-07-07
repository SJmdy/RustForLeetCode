//! 使用栈或队列解决的问题

use std::collections::LinkedList;

/// 1047. 删除字符串中的所有相邻重复项 [✔]
///
/// 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。在 S 上反复执行重复项删除操作，直到无法继续删除。在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
///
/// LC: [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)
///
/// 解题思路：滑动窗口 [] | 栈 [✔]
///
/// **注：**滑动窗口不是最佳解法，效率非常低
pub fn remove_duplicates(s: String) -> String {
    if s.is_empty() { return s; }

    let mut stack = LinkedList::new();

    for c in s.chars().into_iter() {
        if stack.is_empty() || c != *stack.back().unwrap() {
            stack.push_back(c);
        } else {
            stack.pop_back();
        }
    }
    return stack.into_iter().map(|c| { c.to_string() }).collect::<String>();
}


/// 224. 基本计算器 [✔]
///
/// 实现一个基本的计算器来计算一个简单的字符串表达式 s 的值。
///
/// 注：'s' must consist of values in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '(', ')', ' '] only
///
/// LC: [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)
///
/// 解题思路：递归求解 | 栈 [✔]
///
/// op: 当前所应该进行的操作，+ 或 -
///
/// ops: 栈，存放`()`外部的操作符，整体对`()`中的内容是 + 或 -。例如 `-(1 - (2 + 3))`，那么第一个括号对应的符号为
/// `-`，第二个括号对应的符号为 `+`；对于任何公式`s`，都可以转化为`+(s)`的形式，因此ops初始化压入`+`。
///
/// op具体为`+` 或 `-` 取决于ops栈顶符号和当前的操作符；若当前的符号为`'+'`，那么op与ops栈顶保持已知；否则，op为ops栈顶的
/// 相反操作
pub fn calculate(s: String) -> i32 {
    if s.is_empty() { return 0; }
    let mut ops = LinkedList::new();
    // +1: 表示加法
    ops.push_back(true);

    let mut res = 0;

    let chars = s.chars().filter(|c| { *c != ' ' }).collect::<Vec<char>>();

    let s = chars.iter().map(|c| { *c }).collect::<String>();
    let mut left_cur = 0;
    let mut op = true;

    while left_cur < chars.len() {
        match chars[left_cur] {
            '-' => {
                op = !(*ops.back().unwrap());
                left_cur += 1;
            }
            '+' => {
                op = *ops.back().unwrap();
                left_cur += 1;
            }
            '(' => {
                ops.push_back(op);
                left_cur += 1;
            }
            ')' => {
                ops.pop_back();
                left_cur += 1;
            }
            // 数字
            _ => {
                let mut digit_cur = left_cur + 1;
                while digit_cur < chars.len() && chars[digit_cur].is_digit(10) {
                    digit_cur += 1;
                }
                let digit = i32::from_str_radix(&s[left_cur..digit_cur], 10).unwrap();
                res = match op {
                    true => {
                        // println!("+ {:?}", digit);
                        res + digit
                    }
                    false => {
                        // println!("- {:?}", digit);
                        res - digit
                    }
                };
                left_cur = digit_cur;
            }
        }
    }
    return res;
}


/// 227. 基本计算器 II [✔]
///
/// 给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。整数除法仅保留整数部分。
///
/// LC: [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)
///
/// 解题思路：栈
pub fn calculate_2(s: String) -> i32 {
    if s.is_empty() { return 0; }

    let chars = s.chars().filter(|c| { *c != ' ' }).collect::<Vec<char>>();
    let s = chars.iter().map(|c| { *c }).collect::<String>();

    let mut calcu = LinkedList::new();
    // flag: true: 正数; false: 负数
    let mut flag = true;

    fn get_num(cur: usize, s: &String, chars: &Vec<char>) -> (i32, usize) {
        // 获取从cur开始的数字, chars[cur]是数字
        let mut digit_cur = cur + 1;
        while digit_cur < chars.len() && chars[digit_cur].is_digit(10) {
            digit_cur += 1;
        }
        return (i32::from_str_radix(&s[cur..digit_cur], 10).unwrap(), digit_cur);
    }

    let mut left_cur = 0;
    while left_cur < chars.len() {
        match chars[left_cur] {
            '-' => {
                flag = false;
                left_cur += 1;
            }
            '+' => {
                flag = true;
                left_cur += 1;
            }
            '*' => {
                // 获取 * 后的数字
                let (digit, digit_cur) = get_num(left_cur + 1, &s, &chars);
                let prev_digit = calcu.pop_back().unwrap();
                calcu.push_back(prev_digit * digit);
                left_cur = digit_cur;
            }
            '/' => {
                // 获取 / 后的数字
                let (digit, digit_cur) = get_num(left_cur + 1, &s, &chars);
                let prev_digit = calcu.pop_back().unwrap();
                calcu.push_back(prev_digit / digit);
                left_cur = digit_cur;
            }
            _ => {
                let (digit, digit_cur) = get_num(left_cur, &s, &chars);
                calcu.push_back(if flag { digit } else { 0 - digit });
                left_cur = digit_cur;
            }
        }
    }
    return calcu.iter().sum();
}


/// 331. 验证二叉树的前序序列化
///
/// 序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 #。
/// 例如，上面的二叉树可以被序列化为字符串 "9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。
///
/// 给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。
/// 每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。
///
/// 你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 "1,,3"
///
/// LC: [331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)
pub fn is_valid_serialization(preorder: String) -> bool {
    if preorder.is_empty() { return true; }

    let chars = preorder.split(',').into_iter().map(|s| { s.to_string() }).collect::<Vec<String>>();

    let mut stack = LinkedList::new();
    // 在给定的pre_order序列代表的树的顶端额外增加一个节点，可以认为该节点右子树为空，左子树为pre_order
    // 代表的树，因此压入1而不是2（右子树已经被占用）
    stack.push_back(1);

    let mut left_cur = 0;
    while left_cur < chars.len() {
        let top = match stack.back_mut() {
            None => return false,
            Some(v) => v
        };
        *top -= 1;
        if *top == 0 {
            stack.pop_back();
        }
        if !chars[left_cur].eq("#") {
            stack.push_back(2);
        }
        left_cur += 1;
    }
    return stack.is_empty();
}


/// 150. 逆波兰表达式求值
///
/// 根据 逆波兰表示法，求表达式的值。
///
/// 有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
///
/// 说明：整数除法只保留整数部分。给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
///
/// 逆波兰表达式：
///
/// 逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。
///
/// 平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
///
/// 该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
///
/// 逆波兰表达式主要有以下两个优点：
///
/// 去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
///
/// 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。
pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    if tokens.is_empty() { return 0; }

    let mut calcu = LinkedList::new();

    for t in tokens.iter() {
        match i32::from_str_radix(t.as_str(), 10) {
            Ok(v) => calcu.push_back(v),
            Err(_) => {
                let a = calcu.pop_back().unwrap();
                let b = calcu.pop_back().unwrap();
                match t.as_str() {
                    "+" => calcu.push_back(b + a),
                    "-" => calcu.push_back(b - a),
                    "*" => calcu.push_back(b * a),
                    "/" => calcu.push_back(b / a),
                    _ => {}
                }
            }
        };
    }
    return calcu.pop_back().unwrap();
}


/// 503. 下一个更大元素 II [✔]
///
/// 给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。
///
/// 解题思路：单调栈
///
/// 栈存放元素的索引，从栈底到栈顶索引对应的元素的值递减。当栈顶的索引对应的元素小于当前的元素时，在
/// res数组中进行更新。
pub fn next_greater_elements(nums: Vec<i32>) -> Vec<i32> {
    if nums.is_empty() { return Vec::new(); }

    let mut stack = LinkedList::new();
    let length = nums.len();
    let mut res = vec![-1; length];
    for i in 0..nums.len() * 2 - 1 {
        while !stack.is_empty() && nums[*stack.back().unwrap()] < nums[i % length] {
            res[stack.pop_back().unwrap()] = nums[i % length];
        }
        stack.push_back(i % length);
    }
    return res;
}


