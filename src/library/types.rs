#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Char,
    SChar,
    UChar,
    Int,
    Long,
    UInt,
    ULong,
    Double,
    Pointer(Box<Type>),
    Array {
        elem_type: Box<Type>,
        size: i64,
    },
    FunType {
        param_types: Vec<Type>,
        ret_type: Box<Type>,
    },
}

impl Type {
    pub fn get_size(&self) -> i64 {
        match self {
            &Type::Char | &Type::SChar | &Type::UChar => 1,
            &Type::Int | &Type::UInt => 4,
            &Type::Long | &Type::ULong | &Type::Double | &Type::Pointer(_) => 8,
            &Type::Array {
                ref elem_type,
                size,
            } => size * elem_type.get_size(),
            &Type::FunType { .. } => {
                panic!("Internal error: function type doesn't have size.")
            }
        }
    }

    pub fn get_alignment(&self) -> i8 {
        match self {
            &Type::Char | &Type::SChar | &Type::UChar => 1,
            &Type::Int | &Type::UInt => 4,
            &Type::Long | &Type::ULong | &Type::Double | &Type::Pointer(_) => 8,
            &Type::Array { ref elem_type, .. } => elem_type.get_alignment(),
            &Type::FunType { .. } => {
                panic!("Internal error: function type doesn't have alignment.")
            }
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            &Type::Int | &Type::Long | &Type::Char | &Type::SChar => true,
            &Type::UInt | &Type::ULong | &Type::Pointer(_) | &Type::UChar => false,
            &Type::Double | &Type::FunType { .. } | &Type::Array { .. } => {
                panic!(
                    "Internal error: signedness doesn't make sense for type {:?}",
                    self
                )
            }
        }
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Type::Pointer(_))
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Type::Char
            | Type::UChar
            | Type::SChar
            | Type::Int
            | Type::UInt
            | Type::Long
            | Type::ULong => true,
            Type::Double | Type::Array { .. } | Type::Pointer(_) | Type::FunType { .. } => false,
        }
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Type::Array { .. })
    }

    pub fn is_character(&self) -> bool {
        self.get_size() == 1
    }

    pub fn is_arithmetic(&self) -> bool {
        match self {
            Type::Int
            | Type::UInt
            | Type::Long
            | Type::ULong
            | Type::Char
            | Type::UChar
            | Type::SChar
            | Type::Double => true,
            Type::FunType { .. } | Type::Pointer(..) | Type::Array { .. } => false,
        }
    }
}
