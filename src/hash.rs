//! 用Hash解决的题目

use std::collections::HashMap;
use std::cmp::max;

/// 705. 设计哈希集合
///
/// 不使用任何内建的哈希表库设计一个哈希集合（HashSet）。实现 MyHashSet 类：
///
/// void add(key) 向哈希集合中插入值 key ；
///
/// bool contains(key) 返回哈希集合中是否存在这个值 key ；
///
/// void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。
///
/// LC: [705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)
pub struct MyHashSet {
    values: Vec<i32>,
}

impl MyHashSet {
    /** Initialize your data structure here. */
    pub fn new() -> Self {
        return MyHashSet {
            values: Vec::new()
        };
    }

    pub fn add(&mut self, key: i32) {
        if !self.values.contains(&key) {
            self.values.push(key);
        }
    }

    pub fn remove(&mut self, key: i32) {
        match self.values.iter().position(|v| { *v == key }) {
            Some(pos) => { self.values.remove(pos); }
            None => {}
        };
    }

    /** Returns true if this set contains the specified element */
    pub fn contains(&self, key: i32) -> bool {
        self.values.contains(&key)
    }
}


/// 706. 设计哈希映射
///
/// 不使用任何内建的哈希表库设计一个哈希映射（HashMap）。实现 MyHashMap 类：
///
/// MyHashMap() 用空映射初始化对象
///
/// void put(int key, int value) 向 HashMap 插入一个键值对 (key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
///
/// int get(int key) 返回特定的 key 所映射的 value ；如果映射中不包含 key 的映射，返回 -1 。
///
/// void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value
///
/// LC: [706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)
pub struct MyHashMap {
    values: Vec<i32>,
}


/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MyHashMap {
    /** Initialize your data structure here. */
    pub fn new() -> Self {
        return MyHashMap {
            values: vec![-1; 1e6 as usize + 1]
        };
    }

    /** value will always be non-negative. */
    pub fn put(&mut self, key: i32, value: i32) {
        self.values[key as usize] = value;
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    pub fn get(&self, key: i32) -> i32 {
        return self.values[key as usize];
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    pub fn remove(&mut self, key: i32) {
        self.values[key as usize] = -1;
    }
}


/// 525. 连续数组 [✔]
///
/// 给定一个二进制数组, 找到含有相同数量的 0 和 1 的最长连续子数组（的长度）。
///
/// LC: [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)
pub fn find_max_length(nums: Vec<i32>) -> i32 {
    if nums.len() < 2 { return 0; }

    let mut count = 0;
    let mut record = HashMap::new();
    let mut nums = nums;
    nums.insert(0, 2);
    record.insert(0, 0);

    let mut res = 0;
    for i in 1..nums.len() {
        if nums[i] == 1 {
            count += 1;
        } else {
            count -= 1;
        }
        if record.contains_key(&count) {
            res = max(res, i - record[&count]);
        } else {
            record.insert(count, i);
        }
    }
    return res as i32;

    // let mut count = 0;
    // let mut index_record = HashMap::new();
    // index_record.insert(0, 0);
    // let mut res = 0;
    //
    // for i in 0..nums.len() {
    //     if nums[i] == 1 {
    //         count += 1;
    //     } else {
    //         count -= 1;
    //     }
    //     if !index_record.contains_key(&count) {
    //         index_record.insert(count, i + 1);
    //     } else {
    //         let prev_index = index_record[&count];
    //         if res < i + 1 - prev_index {
    //             res = i + 1 - prev_index;
    //         }
    //     }
    // }
    // return res as i32;
}
