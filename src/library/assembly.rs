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
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM7,
    XMM14,
    XMM15,
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
    Shr,
}

#[derive(Clone, Debug)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mult,
    DivDouble,
    And,
    Or,
    Xor,
    Sal,
    Sar,
    Shl,
    Shr,
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
    Double,
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Mov(AsmType, Operand, Operand),
    Movsx(Operand, Operand),
    MovZeroExtend(Operand, Operand),
    Cvttsd2si(AsmType, Operand, Operand),
    Cvtsi2sd(AsmType, Operand, Operand),
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
    StaticConstant {
        name: String,
        alignment: i8,
        init: StaticInit,
    },
}

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}
