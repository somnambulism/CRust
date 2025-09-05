use crate::library::{r#const::T, initializers::StaticInit, types::Type};

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
    Constant(T),
    Var(String),
}

#[derive(Debug, PartialEq)]
pub enum Instruction {
    Return(TackyVal),
    SignExtend {
        src: TackyVal,
        dst: TackyVal,
    },
    ZeroExtend {
        src: TackyVal,
        dst: TackyVal,
    },
    Truncate {
        src: TackyVal,
        dst: TackyVal,
    },
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
pub enum TopLevel {
    FunctionDefinition {
        name: String,
        global: bool,
        params: Vec<String>,
        body: Vec<Instruction>,
    },
    StaticVariable {
        name: String,
        t: Type,
        global: bool,
        init: StaticInit,
    },
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}
