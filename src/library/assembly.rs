use crate::library::initializers::StaticInit;

#[derive(Clone, Debug, PartialEq)]
pub enum Reg {
    AX,
    CX,
    DX,
    R8,
    R9,
    R10,
    R11,
    SP,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Imm(i64),
    Reg(Reg),
    Pseudo(String),
    Stack(isize),
    Data(String),
}

#[derive(Clone, Debug)]
pub enum UnaryOperator {
    Neg,
    Not,
}

#[derive(Clone, Debug)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mult,
    And,
    Or,
    Xor,
    Sal,
    Sar,
}

#[derive(Clone, Debug)]
pub enum CondCode {
    E,
    NE,
    G,
    GE,
    L,
    LE,
    A,
    AE,
    B,
    BE,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsmType {
    Longword,
    Quadword,
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Mov(AsmType, Operand, Operand),
    Movsx(Operand, Operand),
    MovZeroExtend(Operand, Operand),
    Unary(UnaryOperator, AsmType, Operand),
    Binary {
        op: BinaryOperator,
        t: AsmType,
        src: Operand,
        dst: Operand,
    },
    Cmp(AsmType, Operand, Operand),
    Idiv(AsmType, Operand),
    Div(AsmType, Operand),
    Cdq(AsmType),
    Jmp(String),
    JmpCC(CondCode, String),
    SetCC(CondCode, Operand),
    Label(String),
    Push(Operand),
    Call(String),
    Ret,
}

#[derive(Debug)]
pub enum TopLevel {
    Function {
        name: String,
        global: bool,
        instructions: Vec<Instruction>,
    },
    StaticVariable {
        name: String,
        alignment: i8,
        global: bool,
        init: StaticInit,
    },
}

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}
