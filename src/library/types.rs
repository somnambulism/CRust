#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Long,
    FunType {
        param_types: Vec<Type>,
        ret_type: Box<Type>,
    },
}

impl Type {
    pub fn get_alignment(&self) -> i8 {
        match self {
            &Type::Int => 4,
            &Type::Long => 8,
            &Type::FunType { .. } => {
                panic!("Internal error: function type doesn't have alignment.")
            }
        }
    }
}
