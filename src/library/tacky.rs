#[derive(Debug, PartialEq)]
pub enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Mod,
    BitwiseAnd,
    BitwiseOr,
    Xor,
    LeftShift,
    RightShift,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
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
    Binary {
        op: BinaryOperator,
        src1: TackyVal,
        src2: TackyVal,
        dst: TackyVal,
    },
    Copy {
        src: TackyVal,
        dst: TackyVal,
    },
    Jump(String),
    JumpIfZero(TackyVal, String),
    JumpIfNotZero(TackyVal, String),
    Label(String),
    FunCall {
        f: String,
        args: Vec<TackyVal>,
        dst: TackyVal,
    },
}

#[derive(Debug, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Instruction>,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub functions: Vec<FunctionDefinition>,
}
