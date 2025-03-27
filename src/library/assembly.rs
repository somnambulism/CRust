#[derive(Clone, Debug)]
pub enum Reg {
    AX,
    R10,
}

#[derive(Clone, Debug)]
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
pub enum Instruction {
    Mov(Operand, Operand),
    Unary(UnaryOperator, Operand),
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
