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
pub enum CompoundAssignOperator {
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    AmpersandEqual,
    PipeEqual,
    CaretEqual,
    LeftShiftEqual,
    RightShiftEqual,
}

#[derive(Debug, PartialEq)]
pub enum Exp {
    Constant(i32),
    Var(String),
    Unary(UnaryOperator, Box<Exp>),
    Binary(BinaryOperator, Box<Exp>, Box<Exp>),
    Assignment(Box<Exp>, Box<Exp>),
    CompoundAssign(CompoundAssignOperator, Box<Exp>, Box<Exp>),
    PrefixIncrement(Box<Exp>),
    PrefixDecrement(Box<Exp>),
    PostfixIncrement(Box<Exp>),
    PostfixDecrement(Box<Exp>),
    Conditional {
        condition: Box<Exp>,
        then_result: Box<Exp>,
        else_result: Box<Exp>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Declaration {
    pub name: String,
    pub init: Option<Exp>,
}

#[derive(Debug, PartialEq)]
pub enum Statement {
    Return(Exp),
    Expression(Exp),
    If {
        condition: Exp,
        then_clause: Box<Statement>,
        else_clause: Option<Box<Statement>>,
    },
    Null,
}

#[derive(Debug, PartialEq)]
pub enum BlockItem {
    S(Statement),
    D(Declaration),
}

#[derive(Debug, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub body: Vec<BlockItem>,
}

#[derive(Debug, PartialEq)]
pub struct Program {
    pub function: FunctionDefinition,
}
