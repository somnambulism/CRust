#[derive(Debug, Clone, PartialEq)]
pub enum T {
    ConstInt(i32),
    ConstLong(i64),
}

pub const INT_ZERO: T = T::ConstInt(0);
pub const INT_ONE: T = T::ConstInt(1);
