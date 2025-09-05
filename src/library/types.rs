#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Long,
    UInt,
    ULong,
    FunType {
        param_types: Vec<Type>,
        ret_type: Box<Type>,
    },
}

impl Type {
    pub fn get_size(&self) -> i8 {
        match self {
            &Type::Int | &Type::UInt => 4,
            &Type::Long | &Type::ULong => 8,
            &Type::FunType { .. } => {
                panic!("Internal error: function type doesn't have size.")
            }
        }
    }

    pub fn get_alignment(&self) -> i8 {
        match self {
            &Type::Int | &Type::UInt => 4,
            &Type::Long | &Type::ULong => 8,
            &Type::FunType { .. } => {
                panic!("Internal error: function type doesn't have alignment.")
            }
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            &Type::Int | &Type::Long => true,
            &Type::UInt | &Type::ULong => false,
            &Type::FunType { .. } => {
                panic!("Internal error: signedness doesn't make sense for function types.")
            }
        }
    }
}
