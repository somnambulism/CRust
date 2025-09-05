use super::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum T {
    ConstInt(i32),
    ConstLong(i64),
    ConstUInt(u32),
    ConstULong(u64),
}

pub const INT_ZERO: T = T::ConstInt(0);
pub const INT_ONE: T = T::ConstInt(1);

pub fn type_of_const(c: &T) -> Type {
    match c {
        T::ConstInt(_) => Type::Int,
        T::ConstLong(_) => Type::Long,
        T::ConstUInt(_) => Type::UInt,
        T::ConstULong(_) => Type::ULong,
    }
}
