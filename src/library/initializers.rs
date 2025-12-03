use crate::library::types::Type;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StaticInit {
    IntInit(i32),
    LongInit(i64),
    UIntInit(u32),
    ULongInit(u64),
    DoubleInit(f64),
    ZeroInit(i64),
}

pub fn zero(t: &Type) -> StaticInit {
    StaticInit::ZeroInit(t.get_size())
}

impl StaticInit {
    pub fn is_zero(&self) -> bool {
        match self {
            StaticInit::IntInit(i) => *i == 0,
            StaticInit::LongInit(l) => *l == 0,
            StaticInit::UIntInit(u) => *u == 0,
            StaticInit::ULongInit(ul) => *ul == 0,
            // NOTE: Consider all doubles non-zero since we don't know if it's zero or
            // negative zero
            StaticInit::DoubleInit(_) => false,
            StaticInit::ZeroInit(_) => true,
        }
    }
}
