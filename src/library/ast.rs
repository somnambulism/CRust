use std::fmt::Debug;

// Unary and binary operators; used in exp AST both with and without type
// information
pub mod ops {
    #[derive(Debug, PartialEq, Clone)]
    pub enum UnaryOperator {
        Complement,
        Negate,
        Not,
        Incr,
        Decr,
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
        BitwiseXor,
        BitshiftLeft,
        BitshiftRight,
        And,
        Or,
        Equal,
        NotEqual,
        LessThan,
        LessOrEqual,
        GreaterThan,
        GreaterOrEqual,
    }
}

// Exp AST definition without type info
pub mod untyped_exp {
    use super::ops::{BinaryOperator, UnaryOperator};
    use crate::library::ast::{ExpTrait, InitTrait};
    use crate::library::r#const::T;
    use crate::library::types::Type;

    #[derive(Debug, PartialEq, Clone)]
    pub enum Exp {
        Constant(T),
        Var(String),
        String(String),
        Cast {
            target_type: Type,
            e: Box<Exp>,
        },
        Unary(UnaryOperator, Box<Exp>),
        Binary(BinaryOperator, Box<Exp>, Box<Exp>),
        Assignment(Box<Exp>, Box<Exp>),
        CompoundAssign(BinaryOperator, Box<Exp>, Box<Exp>),
        PostfixIncr(Box<Exp>),
        PostfixDecr(Box<Exp>),
        Conditional {
            condition: Box<Exp>,
            then_result: Box<Exp>,
            else_result: Box<Exp>,
        },
        FunCall {
            f: String,
            args: Vec<Exp>,
        },
        Dereference(Box<Exp>),
        AddrOf(Box<Exp>),
        Subscript {
            ptr: Box<Exp>,
            index: Box<Exp>,
        },
    }

    impl ExpTrait for Exp {}

    #[derive(Debug, PartialEq, Clone)]
    pub enum Initializer {
        SingleInit(Exp),
        CompoundInit(Vec<Box<Initializer>>),
    }

    impl InitTrait for Initializer {}
}

// Exp AST definition with type info
pub mod typed_exp {
    use crate::library::{
        ast::{
            ExpTrait, InitTrait,
            ops::{BinaryOperator, UnaryOperator},
        },
        r#const::T,
        types::Type,
    };

    #[derive(Debug, PartialEq, Clone)]
    pub enum InnerExp {
        Constant(T),
        Var(String),
        String(String),
        Cast {
            target_type: Type,
            e: Box<TypedExp>,
        },
        Unary(UnaryOperator, Box<TypedExp>),
        Binary(BinaryOperator, Box<TypedExp>, Box<TypedExp>),
        Assignment(Box<TypedExp>, Box<TypedExp>),
        CompoundAssignment {
            op: BinaryOperator,
            lhs: Box<TypedExp>,
            rhs: Box<TypedExp>,
            /**
             * Type of lhs op rhs;
             * may need to convert lhs to this type before op, and convert result back
             * to lhs type before assignment
             */
            result_t: Type,
        },
        PostfixIncr(Box<TypedExp>),
        PostfixDecr(Box<TypedExp>),
        Conditional {
            condition: Box<TypedExp>,
            then_result: Box<TypedExp>,
            else_result: Box<TypedExp>,
        },
        FunCall {
            f: String,
            args: Vec<TypedExp>,
        },
        Dereference(Box<TypedExp>),
        AddrOf(Box<TypedExp>),
        Subscript {
            ptr: Box<TypedExp>,
            index: Box<TypedExp>,
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

    #[derive(Debug, PartialEq, Clone)]
    pub enum Initializer {
        SingleInit(TypedExp),
        CompoundInit(Type, Vec<Box<Initializer>>),
    }

    impl InitTrait for Initializer {}
}

pub trait ExpTrait: Debug + PartialEq {}

pub trait InitTrait: Debug + PartialEq {}

pub mod storage_class {
    #[derive(Debug, PartialEq, Clone, Copy)]
    pub enum StorageClass {
        Static,
        Extern,
    }
}

pub mod block_items {
    use crate::library::{ast::InitTrait, r#const::T, types::Type};

    use super::{ExpTrait, storage_class::StorageClass};

    #[derive(Debug, PartialEq)]
    pub struct VariableDeclaration<Init: InitTrait> {
        pub name: String,
        pub var_type: Type,
        pub init: Option<Init>,
        pub storage_class: Option<StorageClass>,
    }

    #[derive(Debug, PartialEq)]
    pub enum ForInit<Init: InitTrait, Exp: ExpTrait> {
        InitDecl(VariableDeclaration<Init>),
        InitExp(Option<Exp>),
    }

    #[derive(Debug, PartialEq)]
    pub enum Statement<Init: InitTrait, Exp: ExpTrait> {
        Return(Exp),
        Expression(Exp),
        If {
            condition: Exp,
            then_clause: Box<Statement<Init, Exp>>,
            else_clause: Option<Box<Statement<Init, Exp>>>,
        },
        Compound(Block<Init, Exp>),
        Break(String),
        Continue(String),
        While {
            condition: Exp,
            body: Box<Statement<Init, Exp>>,
            id: String,
        },
        DoWhile {
            body: Box<Statement<Init, Exp>>,
            condition: Exp,
            id: String,
        },
        For {
            init: ForInit<Init, Exp>,
            condition: Option<Exp>,
            post: Option<Exp>,
            body: Box<Statement<Init, Exp>>,
            id: String,
        },
        Switch {
            control: Exp,
            body: Box<Statement<Init, Exp>>,
            id: String,
            cases: Vec<(Option<T>, String)>,
        },
        Case(
            Exp, // exp must be constant; validate during semantic analysis
            Box<Statement<Init, Exp>>,
            String,
        ),
        Default(Box<Statement<Init, Exp>>, String),
        Null,
        LabelledStatement(String, Box<Statement<Init, Exp>>),
        Goto(String),
    }

    #[derive(Debug, PartialEq)]
    pub enum BlockItem<Init: InitTrait, Exp: ExpTrait> {
        S(Statement<Init, Exp>),
        D(Declaration<Init, Exp>),
    }

    #[derive(Debug, PartialEq)]
    pub struct Block<Init: InitTrait, Exp: ExpTrait>(pub Vec<BlockItem<Init, Exp>>);

    #[derive(Debug, PartialEq)]
    pub struct FunctionDeclaration<Init: InitTrait, Exp: ExpTrait> {
        pub name: String,
        pub fun_type: Type,
        pub params: Vec<String>,
        pub body: Option<Block<Init, Exp>>,
        pub storage_class: Option<StorageClass>,
    }

    #[derive(Debug, PartialEq)]
    pub enum Declaration<Init: InitTrait, Exp: ExpTrait> {
        FunDecl(FunctionDeclaration<Init, Exp>),
        VarDecl(VariableDeclaration<Init>),
    }

    #[derive(Debug, PartialEq)]
    pub struct Program<Init: InitTrait, Exp: ExpTrait>(pub Vec<Declaration<Init, Exp>>);
}
