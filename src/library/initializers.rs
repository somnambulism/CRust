use crate::library::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum StaticInit {
    CharInit(i8),
    UCharInit(u8),
    IntInit(i32),
    LongInit(i64),
    UIntInit(u32),
    ULongInit(u64),
    DoubleInit(f64),
    ZeroInit(i64),
    StringInit(String, bool), // (is_null_terminated)
    // pointer to static variable
    PointerInit(String),
}

pub fn zero(t: &Type) -> StaticInit {
    StaticInit::ZeroInit(t.get_size())
}

impl StaticInit {
    pub fn is_zero(&self) -> bool {
        match self {
            StaticInit::CharInit(c) => *c == 0,
            StaticInit::IntInit(i) => *i == 0,
            StaticInit::LongInit(l) => *l == 0,
            StaticInit::UCharInit(c) => *c == 0,
            StaticInit::UIntInit(u) => *u == 0,
            StaticInit::ULongInit(ul) => *ul == 0,
            // NOTE: Consider all doubles non-zero since we don't know if it's zero or
            // negative zero
            StaticInit::DoubleInit(_) => false,
            StaticInit::ZeroInit(_) => true,
            StaticInit::PointerInit(_) | StaticInit::StringInit(..) => false,
        }
    }
}
