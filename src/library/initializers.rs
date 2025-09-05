use crate::library::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticInit {
    IntInit(i32),
    LongInit(i64),
    UIntInit(u32),
    ULongInit(u64),
}

pub fn zero(t: &Type) -> StaticInit {
    match t {
        Type::Int => StaticInit::IntInit(0 as i32),
        Type::Long => StaticInit::LongInit(0 as i64),
        Type::UInt => StaticInit::UIntInit(0 as u32),
        Type::ULong => StaticInit::ULongInit(0 as u64),
        Type::FunType { .. } => {
            panic!("Internal error: zero doesn't make sense for function type");
        }
    }
}

impl StaticInit {
    pub fn is_zero(&self) -> bool {
        match self {
            StaticInit::IntInit(i) => *i == 0,
            StaticInit::LongInit(l) => *l == 0,
            StaticInit::UIntInit(u) => *u == 0,
            StaticInit::ULongInit(ul) => *ul == 0,
        }
    }
}
