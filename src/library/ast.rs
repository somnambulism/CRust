use std::collections::HashSet;

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Complement,
    Negate,
    Not,
}

#[derive(Debug, PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone)]
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
    Constant(i64),
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
    FunCall {
        f: String,
        args: Vec<Exp>,
    },
}

#[derive(Debug, PartialEq)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug, PartialEq)]
pub struct VariableDeclaration {
    pub name: String,
    pub init: Option<Exp>,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug, PartialEq)]
pub enum ForInit {
    InitDecl(VariableDeclaration),
    InitExp(Option<Exp>),
}

pub type SwitchCases = HashSet<Option<i64>>;

#[derive(Debug, PartialEq)]
pub enum Statement {
    Return(Exp),
    Expression(Exp),
    If {
        condition: Exp,
        then_clause: Box<Statement>,
        else_clause: Option<Box<Statement>>,
    },
    Compound(Block),
    Break(String),
    Continue(String),
    While {
        condition: Exp,
        body: Box<Statement>,
        id: String,
    },
    DoWhile {
        body: Box<Statement>,
        condition: Exp,
        id: String,
    },
    For {
        init: ForInit,
        condition: Option<Exp>,
        post: Option<Exp>,
        body: Box<Statement>,
        id: String,
    },
    Switch {
        condition: Exp,
        body: Box<Statement>,
        cases: SwitchCases,
        id: String,
    },
    Case {
        condition: i64,
        body: Box<Statement>,
        switch_label: String,
    },
    Default {
        body: Box<Statement>,
        switch_label: String,
    },
    Null,
    Labelled {
        label: String,
        statement: Box<Statement>,
    },
    Goto(String),
}

#[derive(Debug, PartialEq)]
pub enum BlockItem {
    S(Statement),
    D(Declaration),
}

#[derive(Debug, PartialEq)]
pub struct Block(pub Vec<BlockItem>);

#[derive(Debug, PartialEq)]
pub struct FunctionDeclaration {
    pub name: String,
    pub params: Vec<String>,
    pub body: Option<Block>,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug, PartialEq)]
pub enum Declaration {
    FunDecl(FunctionDeclaration),
    VarDecl(VariableDeclaration),
}

#[derive(Debug, PartialEq)]
pub struct Program(pub Vec<Declaration>);
