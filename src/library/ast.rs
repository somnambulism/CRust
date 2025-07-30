use std::fmt::Debug;

// Unary and binary operators; used in exp AST both with and without type
// information
pub mod ops {
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
}

// Exp AST definition without type info
pub mod untyped_exp {
    use super::ops::{BinaryOperator, CompoundAssignOperator, UnaryOperator};
    use crate::library::ast::ExpTrait;
    use crate::library::r#const::T;
    use crate::library::types::Type;

    #[derive(Debug, PartialEq, Clone)]
    pub enum Exp {
        Constant(T),
        Var(String),
        Cast {
            target_type: Type,
            e: Box<Exp>,
        },
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

    impl ExpTrait for Exp {}
}

// Exp AST definition with type info
pub mod typed_exp {
    use crate::library::{
        ast::{
            ExpTrait,
            ops::{BinaryOperator, CompoundAssignOperator, UnaryOperator},
        },
        r#const::T,
        types::Type,
    };

    #[derive(Debug, PartialEq, Clone)]
    pub enum InnerExp {
        Constant(T),
        Var(String),
        Cast {
            target_type: Type,
            e: Box<TypedExp>,
        },
        Unary(UnaryOperator, Box<TypedExp>),
        Binary(BinaryOperator, Box<TypedExp>, Box<TypedExp>),
        Assignment(Box<TypedExp>, Box<TypedExp>),
        CompoundAssign(CompoundAssignOperator, Box<TypedExp>, Box<TypedExp>),
        PrefixIncrement(Box<TypedExp>),
        PrefixDecrement(Box<TypedExp>),
        PostfixIncrement(Box<TypedExp>),
        PostfixDecrement(Box<TypedExp>),
        Conditional {
            condition: Box<TypedExp>,
            then_result: Box<TypedExp>,
            else_result: Box<TypedExp>,
        },
        FunCall {
            f: String,
            args: Vec<TypedExp>,
        },
    }

    #[derive(Debug, PartialEq, Clone)]
    pub struct TypedExp {
        pub e: InnerExp,
        pub t: Type,
    }

    impl TypedExp {
        pub fn get_type(&self) -> Type {
            self.t.clone()
        }

        pub fn set_type(e: InnerExp, new_type: Type) -> TypedExp {
            TypedExp { e, t: new_type }
        }
    }

    impl ExpTrait for TypedExp {}
}

pub trait ExpTrait: Debug + PartialEq {}

pub mod storage_class {
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub enum StorageClass {
        Static,
        Extern,
    }
}

pub mod block_items {
    use std::collections::HashSet;

    use crate::library::types::Type;

    use super::{ExpTrait, storage_class::StorageClass};

    #[derive(Debug, PartialEq)]
    pub struct VariableDeclaration<Exp: ExpTrait> {
        pub name: String,
        pub var_type: Type,
        pub init: Option<Exp>,
        pub storage_class: Option<StorageClass>,
    }

    #[derive(Debug, PartialEq)]
    pub enum ForInit<Exp: ExpTrait> {
        InitDecl(VariableDeclaration<Exp>),
        InitExp(Option<Exp>),
    }

    pub type SwitchCases = HashSet<Option<i64>>;

    #[derive(Debug, PartialEq)]
    pub enum Statement<Exp: ExpTrait> {
        Return(Exp),
        Expression(Exp),
        If {
            condition: Exp,
            then_clause: Box<Statement<Exp>>,
            else_clause: Option<Box<Statement<Exp>>>,
        },
        Compound(Block<Exp>),
        Break(String),
        Continue(String),
        While {
            condition: Exp,
            body: Box<Statement<Exp>>,
            id: String,
        },
        DoWhile {
            body: Box<Statement<Exp>>,
            condition: Exp,
            id: String,
        },
        For {
            init: ForInit<Exp>,
            condition: Option<Exp>,
            post: Option<Exp>,
            body: Box<Statement<Exp>>,
            id: String,
        },
        Switch {
            condition: Exp,
            body: Box<Statement<Exp>>,
            cases: SwitchCases,
            id: String,
        },
        Case {
            condition: i64,
            body: Box<Statement<Exp>>,
            switch_label: String,
        },
        Default {
            body: Box<Statement<Exp>>,
            switch_label: String,
        },
        Null,
        Labelled {
            label: String,
            statement: Box<Statement<Exp>>,
        },
        Goto(String),
    }

    #[derive(Debug, PartialEq)]
    pub enum BlockItem<Exp: ExpTrait> {
        S(Statement<Exp>),
        D(Declaration<Exp>),
    }

    #[derive(Debug, PartialEq)]
    pub struct Block<Exp: ExpTrait>(pub Vec<BlockItem<Exp>>);

    #[derive(Debug, PartialEq)]
    pub struct FunctionDeclaration<Exp: ExpTrait> {
        pub name: String,
        pub fun_type: Type,
        pub params: Vec<String>,
        pub body: Option<Block<Exp>>,
        pub storage_class: Option<StorageClass>,
    }

    #[derive(Debug, PartialEq)]
    pub enum Declaration<Exp: ExpTrait> {
        FunDecl(FunctionDeclaration<Exp>),
        VarDecl(VariableDeclaration<Exp>),
    }

    #[derive(Debug, PartialEq)]
    pub struct Program<Exp: ExpTrait>(pub Vec<Declaration<Exp>>);
}
