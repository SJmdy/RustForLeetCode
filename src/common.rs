//! 定义一些通用的数据结构及对应的方法

use std::rc::Rc;
use std::cell::RefCell;
use std::option::Option::Some;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    pub fn new(val: i32) -> Self {
        return ListNode {
            val,
            next: None,
        };
    }

    pub fn serialize(head: Option<Box<ListNode>>) -> Vec<i32> {
        if head.is_none() { return Vec::new(); }

        let mut cur = head.as_ref().unwrap();
        let mut res = vec![cur.val];
        while let Some(next) = &cur.next {
            cur = next;
            res.push(cur.val);
        }
        return res;
    }

    pub fn deserialize(nums: Vec<i32>) -> Option<Box<ListNode>> {
        if nums.is_empty() { return None; }
        let mut head = ListNode { val: -1, next: None };
        let mut cur = &mut head;

        for v in nums.into_iter() {
            let next = ListNode { val: v, next: None };
            cur.next = Some(Box::new(next));
            cur = cur.next.as_mut().unwrap();
        }
        return head.next.take();
    }
}


#[derive(PartialEq, Eq, Clone, Debug)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    pub fn new(val: i32) -> Self {
        return TreeNode {
            val,
            left: None,
            right: None,
        };
    }
}