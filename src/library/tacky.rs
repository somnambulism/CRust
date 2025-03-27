#[derive(Debug, PartialEq)]
pub enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TackyVal {
    Constant(i32),
    Var(String),
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Return(TackyVal),
    Unary {
        op: UnaryOperator,
        src: TackyVal,
        dst: TackyVal,
    },
}

#[derive(Debug, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub function: FunctionDefinition,
}
