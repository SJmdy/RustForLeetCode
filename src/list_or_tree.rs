//! 使用链表解决的题目，但是链表题为什么要用Rust写呢

use std::rc::Rc;
use std::cell::RefCell;
use std::cmp::max;
use std::option::Option::Some;
use std::collections::LinkedList;
use crate::common::{ListNode, TreeNode};


/// 92. 反转链表 II
///
/// 给你单链表的头节点 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
///
/// LC: [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
pub fn reverse_between(head: Option<Box<ListNode>>, left: i32, right: i32) -> Option<Box<ListNode>> {
    let (_, _) = (left, right);
    return head;
    // if head.is_none() || left == right { return head; }
    //
    // let mut cur = head.clone();
    // for _ in 0..right {
    //     cur = cur.as_ref().unwrap().next;
    // }
    // let second_half = cur.take();
    //
    // if left == 1 {
    //     let mut fast_cur = head.clone();
    //     let mut head = fast_cur.clone();
    //     let mut slow_cur: Option<Box<ListNode>> = None;
    //
    //     while fast_cur.is_some() {
    //         let next = fast_cur.as_mut().unwrap().next.take();
    //         fast_cur.as_mut().unwrap().next = slow_cur;
    //         slow_cur = fast_cur;
    //         fast_cur = next;
    //     }
    //     head.as_mut().unwrap().next = second_half;
    //     return slow_cur;
    // }
    // return None;
}


/// 98. 验证二叉搜索树
///
/// 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
///
/// 假设一个二叉搜索树具有如下特征：
///
/// 节点的左子树只包含小于当前节点的数。节点的右子树只包含大于当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。
///
/// LC: [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)
///
/// /// 解题思路：二叉搜索树的中序遍历序列递增 [✔] | 二叉搜索树左右子树有界
pub fn is_valid_bst1(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if root.is_none() { return true; }

    fn dfs(cur: Rc<RefCell<TreeNode>>, values: &mut Vec<i32>) {
        if let Some(left) = &cur.borrow().left {
            dfs(Rc::clone(left), values);
        }
        values.push(cur.borrow().val);
        if let Some(right) = &cur.borrow().right {
            dfs(Rc::clone(right), values);
        }
    }

    let mut values = Vec::new();
    dfs(root.unwrap(), &mut values);

    for i in 1..values.len() {
        if values[i] <= values[i - 1] {
            return false;
        }
    }
    return true;
}


/// 98. 验证二叉搜索树
///
/// 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
///
/// 假设一个二叉搜索树具有如下特征：
///
/// 节点的左子树只包含小于当前节点的数。节点的右子树只包含大于当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。
///
/// LC: [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)
///
/// /// 解题思路：二叉搜索树的中序遍历序列递增 | 二叉搜索树左右子树有界 [✔]
pub fn is_valid_bst2(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if root.is_none() { return true; }

    fn dfs(cur: Rc<RefCell<TreeNode>>, lower: i64, upper: i64) -> bool {
        let current_val = cur.borrow().val as i64;
        if current_val <= lower || current_val >= upper {
            return false;
        }

        let mut flag = true;
        if let Some(left) = &cur.borrow().left {
            flag &= dfs(Rc::clone(left), lower, current_val);
        }
        if !flag { return flag; }

        if let Some(right) = &cur.borrow().right {
            flag &= dfs(Rc::clone(right), current_val, upper);
        }
        return flag;
    }

    return dfs(root.unwrap(), i64::min_value(), i64::max_value());
}


/// 124. 二叉树中的最大路径和
///
/// 路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
///
/// 路径和 是路径中各节点值的总和。
///
/// 给你一个二叉树的根节点 root ，返回其 最大路径和 。
///
/// LC: [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() { return 0; }

    fn dfs(cur: Rc<RefCell<TreeNode>>, res: &mut i32) -> i32 {
        let v = cur.borrow().val;

        let left = if let Some(left) = &cur.borrow().left {
            max(dfs(Rc::clone(left), res), 0)
        } else {
            0
        };
        let right = if let Some(right) = &cur.borrow().right {
            max(dfs(Rc::clone(right), res), 0)
        } else {
            0
        };

        if *res < v + left + right {
            *res = v + left + right
        }

        return v + max(left, right);
    }

    let mut res = i32::min_value();
    dfs(root.unwrap(), &mut res);
    return res;
}


/// 101. 对称二叉树
///
/// 给定一个二叉树，检查它是否是镜像对称的。
///
/// LC: [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if root.is_none() { return true; }

    let cur = root.unwrap();

    fn dfs(cur1: Rc<RefCell<TreeNode>>, cur2: Rc<RefCell<TreeNode>>) -> bool {
        if cur1.borrow().val != cur2.borrow().val {
            return false;
        }
        let f1 = match (&cur1.borrow().left, &cur2.borrow().right) {
            (None, None) => true,
            (Some(left), Some(right)) => dfs(Rc::clone(left), Rc::clone(right)),
            _ => false
        };
        if !f1 { return false; }

        let f2 = match (&cur1.borrow().right, &cur2.borrow().left) {
            (None, None) => true,
            (Some(right), Some(left)) => dfs(Rc::clone(right), Rc::clone(left)),
            _ => false
        };
        return f2;
    }

    return match (&cur.borrow().left, &cur.borrow().right) {
        (None, None) => true,
        (Some(left), Some(right)) => dfs(Rc::clone(left), Rc::clone(right)),
        _ => false
    };
}


/// 61. 旋转链表
///
/// 给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
///
/// LC: [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)
pub fn rotate_right(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    if k == 0 || head.is_none() { return head; }

    let mut length = 1;
    let mut cur = head.as_ref().unwrap();

    while let Some(next) = cur.next.as_ref() {
        cur = next;
        length += 1;
    }
    let k = k % length;
    if k == 0 { return head; }

    let mut head = head;
    let mut cur = head.as_mut().unwrap();
    for _ in 0..length - k - 1 {
        cur = cur.next.as_mut().unwrap();
    }

    let mut new_head = cur.next.take();
    let mut cur = new_head.as_mut().unwrap();
    for _ in 0..k - 1 {
        cur = cur.next.as_mut().unwrap();
    }
    cur.next = head;
    return new_head;
}


/// 173. 二叉搜索树迭代器
///
/// 实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
///
/// BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。int next()将指针向右移动，然后返回指针处的数字。
///
/// 注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。
///
/// 你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。
///
/// LC: [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)
pub struct BSTIterator {
    pub stack: LinkedList<Option<Rc<RefCell<TreeNode>>>>,
    pub cur: Option<Rc<RefCell<TreeNode>>>,
}


impl BSTIterator {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        return BSTIterator {
            cur: root,
            stack: LinkedList::new(),
        };
    }

    pub fn next(&mut self) -> i32 {
        if !self.has_next() { return -1; }
        let mut cur = self.cur.clone();

        while cur.is_some() {
            self.stack.push_back(cur.clone());
            cur = cur.unwrap().borrow().left.clone();
        }
        cur = self.stack.pop_back().unwrap();
        let val = cur.as_ref().unwrap().borrow().val;
        self.cur = cur.unwrap().borrow().right.clone();
        return val;
    }

    pub fn has_next(&self) -> bool {
        return self.cur.is_some() || !self.stack.is_empty();
    }
}


/// 783. 二叉搜索树节点最小距离
///
/// 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
///
/// LC: [783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)
pub fn min_diff_in_bst(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() { return 0; }

    let mut cur = root.clone();
    let mut stack = LinkedList::new();
    let mut res = Vec::new();

    while !stack.is_empty() || cur.is_some() {
        while cur.is_some() {
            stack.push_back(cur.clone());
            cur = cur.unwrap().borrow().left.clone();
        }
        cur = stack.pop_back().unwrap();
        res.push(cur.as_ref().unwrap().borrow().val);
        cur = cur.unwrap().borrow().right.clone();
    }

    let mut diff = i32::max_value();
    if res.len() == 1 { return 0; }
    for i in 0..res.len() - 1 {
        if diff > res[i + 1] - res[i] {
            diff = res[i + 1] - res[i];
        }
    }
    return diff;
}
