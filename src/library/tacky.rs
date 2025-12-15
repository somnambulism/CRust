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
    BitshiftLeft,
    BitshiftRight,
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
    DoubleToInt {
        src: TackyVal,
        dst: TackyVal,
    },
    IntToDouble {
        src: TackyVal,
        dst: TackyVal,
    },
    DoubleToUInt {
        src: TackyVal,
        dst: TackyVal,
    },
    UIntToDouble {
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
    GetAddress {
        src: TackyVal,
        dst: TackyVal,
    },
    Load {
        src_ptr: TackyVal,
        dst: TackyVal,
    },
    Store {
        src: TackyVal,
        dst_ptr: TackyVal,
    },
    AddPtr {
        ptr: TackyVal,
        index: TackyVal,
        scale: i64,
        dst: TackyVal,
    },
    CopyToOffset {
        src: TackyVal,
        dst: String,
        offset: i64,
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
        init: Vec<StaticInit>,
    },
    StaticConstant {
        name: String,
        t: Type,
        init: StaticInit,
    },
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}
