#[derive(Debug, PartialEq)]
pub enum UnaryOperator {
    Complement,
    Negate,
}

#[derive(Debug, PartialEq)]
pub enum Exp {
    Constant(i32),
    Unary(UnaryOperator, Box<Exp>),
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
