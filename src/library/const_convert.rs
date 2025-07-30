use crate::library::{r#const::T, types::Type};

pub fn const_convert(target_type: &Type, c: &T) -> T {
    match (target_type, c) {
        // i as i64 preserves value
        (Type::Long, T::ConstInt(i)) => T::ConstLong(*i as i64),
        // l as i32 wraps module 2*32
        (Type::Int, T::ConstLong(l)) => T::ConstInt(*l as i32),
        // Otherwise c already has the correct type
        _ => c.clone(),
    }
}
