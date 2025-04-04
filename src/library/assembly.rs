#[derive(Clone, Debug, PartialEq)]
pub enum Reg {
    AX,
    CX,
    DX,
    R10,
    R11,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Imm(i32),
    Reg(Reg),
    Pseudo(String),
    Stack(i32),
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
}

#[derive(Clone, Debug)]
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOperator, Operand),
    Binary {
        op: BinaryOperator,
        src: Operand,
        dst: Operand,
    },
    Cmp(Operand, Operand),
    Idiv(Operand),
    Cdq,
    Jmp(String),
    JmpCC(CondCode, String),
    SetCC(CondCode, Operand),
    Label(String),
    AllocateStack(i32),
    Ret,
}

#[derive(Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug)]
pub struct Program {
    pub function: FunctionDefinition,
}
