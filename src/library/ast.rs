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
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

#[derive(Debug, PartialEq)]
pub enum Exp {
    Constant(i32),
    Unary(UnaryOperator, Box<Exp>),
    Binary(BinaryOperator, Box<Exp>, Box<Exp>),
}

#[derive(Debug, PartialEq)]
pub enum Statement {
    Return(Exp),
}

#[derive(Debug, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub body: Statement,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub function: FunctionDefinition,
}
