use std::{
    collections::HashMap,
    mem::{Discriminant, discriminant},
};

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::library::{ast::storage_class::StorageClass, r#const::T, types::Type};

use super::{
    ast::block_items::{Block as BlockStruct, Program as ProgramStruct},
    ast::block_items::{
        BlockItem, Declaration, ForInit, FunctionDeclaration, Statement, VariableDeclaration,
    },
    ast::ops::{BinaryOperator, UnaryOperator},
    ast::untyped_exp::Exp,
    tok_stream::TokenStream,
    tokens::Token,
};

type Block = BlockStruct<Exp>;
type Program = ProgramStruct<Exp>;

pub struct Parser {
    tokens: TokenStream,
}

fn get_precedence(token: &Token) -> Option<i8> {
    match token {
        Token::Star | Token::Slash | Token::Percent => Some(50),
        Token::Plus | Token::Hyphen | Token::DoublePlus | Token::DoubleHyphen => Some(45),
        Token::LeftShift | Token::RightShift => Some(40),
        Token::LessThan | Token::LessOrEqual | Token::GreaterThan | Token::GreaterOrEqual => {
            Some(35)
        }
        Token::DoubleEqual | Token::NotEqual => Some(30),
        Token::Ampersand => Some(25),
        Token::Caret => Some(20),
        Token::Pipe => Some(15),
        Token::LogicalAnd => Some(10),
        Token::LogicalOr => Some(5),
        Token::QuestionMark => Some(3),
        Token::EqualSign
        | Token::PlusEqual
        | Token::HyphenEqual
        | Token::StarEqual
        | Token::SlashEqual
        | Token::PercentEqual
        | Token::AmpersandEqual
        | Token::PipeEqual
        | Token::CaretEqual
        | Token::DoubleLeftBracketEqual
        | Token::DoubleRightBracketEqual => Some(1),
        _ => None,
    }
}

fn get_compound_operator(token: &Token) -> Option<BinaryOperator> {
    match token {
        Token::EqualSign => None,
        Token::PlusEqual => Some(BinaryOperator::Add),
        Token::HyphenEqual => Some(BinaryOperator::Subtract),
        Token::SlashEqual => Some(BinaryOperator::Divide),
        Token::PercentEqual => Some(BinaryOperator::Mod),
        Token::StarEqual => Some(BinaryOperator::Multiply),
        Token::AmpersandEqual => Some(BinaryOperator::BitwiseAnd),
        Token::PipeEqual => Some(BinaryOperator::BitwiseOr),
        Token::CaretEqual => Some(BinaryOperator::BitwiseXor),
        Token::DoubleLeftBracketEqual => Some(BinaryOperator::BitshiftLeft),
        Token::DoubleRightBracketEqual => Some(BinaryOperator::BitshiftRight),
        t => panic!(
            "Internal error: not a compound assignment operator: {:?}",
            t
        ),
    }
}

fn is_assignment(token: &Token) -> bool {
    matches!(
        token,
        Token::EqualSign
            | Token::PlusEqual
            | Token::HyphenEqual
            | Token::StarEqual
            | Token::SlashEqual
            | Token::PercentEqual
            | Token::AmpersandEqual
            | Token::CaretEqual
            | Token::PipeEqual
            | Token::DoubleLeftBracketEqual
            | Token::DoubleRightBracketEqual
    )
}

// Helper function to check whether a token is a type specifier
fn is_type_specifier(token: &Token) -> bool {
    matches!(
        token,
        Token::KWInt | Token::KWLong | Token::KWSigned | Token::KWUnsigned | Token::KWDouble
    )
}

// Helper function to check whether a token is a specifier
fn is_specifier(token: &Token) -> bool {
    match token {
        Token::KWStatic | Token::KWExtern => true,
        other => is_type_specifier(other),
    }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: TokenStream::new(tokens),
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        match self.tokens.take_token() {
            Ok(token) if token == expected => Ok(()),
            Ok(token) => Err(format!("Expected {:?}, found {:?}", expected, token)),
            Err(e) => Err(e),
        }
    }

    fn parse_id(&mut self) -> Result<String, String> {
        match self.tokens.take_token() {
            Ok(Token::Identifier(x)) => Ok(x),
            other => Err(format!("Expected identifier, found {:?}", other.unwrap())),
        }
    }

    // Specifiers

    // <type-specifier> ::= "int" | "long" | "unsigned" | "signed"
    fn parse_type_specifier(&mut self) -> Result<Token, String> {
        let spec = self.tokens.take_token()?;
        if is_type_specifier(&spec) {
            Ok(spec)
        } else {
            Err(format!("Expected a type specifier, found {:?}", spec))
        }
    }

    // Helper to consume a list of type specifiers from start of the token stream:
    // { <type-specifier> }+
    fn parse_type_specifier_list(&mut self) -> Result<Vec<Token>, String> {
        let mut specs = vec![];

        let first = self.parse_type_specifier()?;
        specs.push(first);

        while let Ok(tok) = self.tokens.peek() {
            if is_type_specifier(tok) {
                let spec = self.parse_type_specifier()?;
                specs.push(spec);
            } else {
                break;
            }
        }

        Ok(specs)
    }

    // <specifier> ::= <type-specifier> | "static" | "extern"
    fn parse_specifier(&mut self) -> Result<Token, String> {
        let spec = self.tokens.take_token()?;
        if is_specifier(&spec) {
            Ok(spec)
        } else {
            Err(format!(
                "Expected a type or storage-class specifier, found {:?}",
                spec
            ))
        }
    }

    // Helper to consume a list of specifiers from start of the token stream:
    // { <specifier> }+
    fn parse_specifier_list(&mut self) -> Result<Vec<Token>, String> {
        let spec = self.parse_specifier()?;
        let mut specifiers = vec![spec];
        while let Ok(token) = self.tokens.peek() {
            if is_specifier(token) {
                specifiers.push(self.parse_specifier()?);
            } else {
                break;
            }
        }
        Ok(specifiers)
    }

    // Convert a single token to a storage class
    fn parse_storage_class(&mut self, token: &Token) -> Result<StorageClass, String> {
        match token {
            Token::KWExtern => Ok(StorageClass::Extern),
            Token::KWStatic => Ok(StorageClass::Static),
            other => Err(format!(
                "Expected a storage class specifier, found {:?}",
                other
            )),
        }
    }

    // Conver list of specifiers to a type
    fn parse_type(&mut self, specifier_list: Vec<Token>) -> Result<Type, String> {
        if specifier_list.len() == 1 && matches!(&specifier_list[0], Token::KWDouble) {
            return Ok(Type::Double);
        }

        if specifier_list.is_empty() {
            return Err("Invalid type specifier".to_string());
        }

        let mut counts: HashMap<Discriminant<Token>, usize> = HashMap::new();
        for tok in specifier_list {
            let d = discriminant(&tok);
            *counts.entry(d).or_insert(0) += 1;
        }

        let double_disc = discriminant(&Token::KWDouble);
        if counts.get(&double_disc).copied().unwrap_or(0) > 0 {
            return Err("Invalid type specifier".to_string());
        }

        let signed_disc = discriminant(&Token::KWSigned);
        let unsigned_disc = discriminant(&Token::KWUnsigned);
        let long_disc = discriminant(&Token::KWLong);

        if counts.get(&signed_disc).copied().unwrap_or(0) > 0
            && counts.get(&unsigned_disc).copied().unwrap_or(0) > 0
        {
            return Err("Invalid type specifier".to_string());
        }

        for (&disc, &cnt) in counts.iter() {
            if disc == long_disc {
                if cnt > 2 {
                    return Err("Invalid type specifier".to_string());
                }
            } else if cnt > 1 {
                return Err("Invalid type specifier".to_string());
            }
        }

        let unsigned = counts.get(&unsigned_disc).copied().unwrap_or(0) > 0;
        let long = counts.get(&long_disc).copied().unwrap_or(0) > 0;

        if unsigned && long {
            Ok(Type::ULong)
        } else if unsigned {
            Ok(Type::UInt)
        } else if long {
            Ok(Type::Long)
        } else {
            Ok(Type::Int)
        }
    }

    // Convert list of specifiers to type and storage class
    fn parse_type_and_storage_class(
        &mut self,
        specifier_list: Vec<Token>,
    ) -> Result<(Type, Option<StorageClass>), String> {
        let (types, storage_class): (Vec<_>, Vec<_>) = specifier_list
            .into_iter()
            .partition(|tok| is_type_specifier(tok));
        let typ = self.parse_type(types)?;
        let storage_class = match storage_class.as_slice() {
            [] => None,
            [sc] => Some(self.parse_storage_class(sc)?),
            _ => {
                return Err("Internal error - not a storage class".to_string());
            }
        };
        Ok((typ, storage_class))
    }

    // Constants

    // Convert a single constant token to into constant AST node
    fn parse_signed_const(token: Token) -> Result<Exp, String> {
        let (v, is_int) = match token {
            Token::ConstInt(i) => (i, true),
            Token::ConstLong(l) => (l, false),
            other => {
                return Err(format!(
                    "Expected a signed integer constant, found {:?}",
                    other
                ));
            }
        };

        // 2^63 - 1
        let max_long = BigInt::from(2u64).pow(63) - 1;
        if v > max_long {
            return Err("Constant is too large to represent an int or long".to_string());
        }

        // 2^31 - 1
        let max_int = BigInt::from(2u64).pow(31) - 1;
        if is_int && v <= max_int {
            Ok(Exp::Constant(T::ConstInt(v.to_i32().unwrap())))
        } else {
            Ok(Exp::Constant(T::ConstLong(v.to_i64().unwrap())))
        }
    }

    fn parse_unsigned_const(token: Token) -> Result<Exp, String> {
        let (v, is_uint) = match token {
            Token::ConstUInt(ui) => (ui, true),
            Token::ConstULong(ul) => (ul, false),
            other => {
                return Err(format!(
                    "Expected an unsigned integer constant, found {:?}",
                    other
                ));
            }
        };

        let max_ulong = BigInt::from(2u64).pow(64) - 1;
        if v > max_ulong {
            return Err(
                "Constant is too large to represent an unsigned int or unsigned long".to_string(),
            );
        }

        let max_uint = BigInt::from(2u32).pow(32) - 1;
        if is_uint && v <= max_uint {
            Ok(Exp::Constant(T::ConstUInt(v.to_u32().unwrap())))
        } else {
            Ok(Exp::Constant(T::ConstULong(v.to_u64().unwrap())))
        }
    }

    // <const> ::= <int> | <long> | <uint> | <ulong>

    /* Just remove the next token from the stream and pass it off to the
    appropriate helper function to convert it to a const.T */
    fn parse_const(&mut self) -> Result<Exp, String> {
        let const_tok = self.tokens.take_token()?;
        match const_tok {
            Token::ConstInt(_) | Token::ConstLong(_) => Self::parse_signed_const(const_tok),
            Token::ConstUInt(_) | Token::ConstULong(_) => Self::parse_unsigned_const(const_tok),
            Token::ConstDouble(d) => Ok(Exp::Constant(T::ConstDouble(d))),
            other => Err(format!("Expected a constant token, found {:?}", other)),
        }
    }
    // Expressions

    // <unop> ::= "-" | "~" | "!"
    fn parse_unop(&mut self) -> Result<UnaryOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::Tilde) => Ok(UnaryOperator::Complement),
            Ok(Token::Hyphen) => Ok(UnaryOperator::Negate),
            Ok(Token::Bang) => Ok(UnaryOperator::Not),
            Ok(Token::DoublePlus) => Ok(UnaryOperator::Incr),
            Ok(Token::DoubleHyphen) => Ok(UnaryOperator::Decr),
            other => Err(format!(
                "Expected a unary operator, found {:?}",
                other.unwrap()
            )),
        }
    }

    // <binop> ::= "-" | "+" | "*" | "/" | "%" | "&" | "|"
    //          | "<<" | ">>" | "&&" | "||"
    //          | "==" | "!=" | "<" | "<=" | ">" | ">="
    fn parse_binop(&mut self) -> Result<BinaryOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::Plus) => Ok(BinaryOperator::Add),
            Ok(Token::Hyphen) => Ok(BinaryOperator::Subtract),
            Ok(Token::Star) => Ok(BinaryOperator::Multiply),
            Ok(Token::Slash) => Ok(BinaryOperator::Divide),
            Ok(Token::Percent) => Ok(BinaryOperator::Mod),
            Ok(Token::Ampersand) => Ok(BinaryOperator::BitwiseAnd),
            Ok(Token::Pipe) => Ok(BinaryOperator::BitwiseOr),
            Ok(Token::Caret) => Ok(BinaryOperator::BitwiseXor),
            Ok(Token::LeftShift) => Ok(BinaryOperator::BitshiftLeft),
            Ok(Token::RightShift) => Ok(BinaryOperator::BitshiftRight),
            Ok(Token::LogicalAnd) => Ok(BinaryOperator::And),
            Ok(Token::LogicalOr) => Ok(BinaryOperator::Or),
            Ok(Token::DoubleEqual) => Ok(BinaryOperator::Equal),
            Ok(Token::NotEqual) => Ok(BinaryOperator::NotEqual),
            Ok(Token::LessThan) => Ok(BinaryOperator::LessThan),
            Ok(Token::LessOrEqual) => Ok(BinaryOperator::LessOrEqual),
            Ok(Token::GreaterThan) => Ok(BinaryOperator::GreaterThan),
            Ok(Token::GreaterOrEqual) => Ok(BinaryOperator::GreaterOrEqual),
            other => Err(format!(
                "Expected a binary operator, found {:?}",
                other.unwrap()
            )),
        }
    }

    // Helper function to parse the middle of a conditional expression:
    // "?" <exp> ":"
    fn parse_conditional_middle(&mut self) -> Result<Exp, String> {
        self.expect(Token::QuestionMark)?;
        let e = self.parse_expression(0)?;
        self.expect(Token::Colon)?;
        Ok(e)
    }

    // <primary-exp> ::= <int> | <identifier> | <identifier> "(" [ <argument-list ] ")" | "(" <exp> ")"
    fn parse_primary_expression(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // constant
            Token::ConstInt(_)
            | Token::ConstLong(_)
            | Token::ConstUInt(_)
            | Token::ConstULong(_)
            | Token::ConstDouble(_) => self.parse_const(),
            // variable or function call
            Token::Identifier(_) => {
                let id = self.parse_id()?;
                let exp = if self.tokens.peek()? == &Token::OpenParen {
                    // It's a function call - consume open paren, then parse args
                    self.tokens.take_token()?;
                    let args = if self.tokens.peek()? == &Token::CloseParen {
                        vec![]
                    } else {
                        self.parse_argument_list()?
                    };
                    self.expect(Token::CloseParen)?;
                    Exp::FunCall { f: id, args }
                } else {
                    // It's a variable
                    Exp::Var(id)
                };

                Ok(exp)
            }
            // cast or parenthesized expression
            Token::OpenParen => {
                // Consume open paren
                self.tokens.take_token()?;
                let e = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                Ok(e)
            }
            t => Err(format!("Expected a factor, found {:?}", t)),
        }
    }

    fn parse_postfix_exp(&mut self) -> Result<Exp, String> {
        let primary = self.parse_primary_expression()?;
        self.postfix_helper(primary)
    }

    fn postfix_helper(&mut self, primary: Exp) -> Result<Exp, String> {
        match self.tokens.peek()? {
            Token::DoubleHyphen => {
                self.tokens.take_token()?;
                let decr_exp = Exp::PostfixDecr(primary.into());
                self.postfix_helper(decr_exp)
            }
            Token::DoublePlus => {
                self.tokens.take_token()?;
                let incr_exp = Exp::PostfixIncr(primary.into());
                self.postfix_helper(incr_exp)
            }
            _ => Ok(primary),
        }
    }

    // <factor> ::= <unop> <factor> | <postfix-exp>
    fn parse_factor(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // unary expression
            Token::Hyphen
            | Token::Tilde
            | Token::Bang
            | Token::DoublePlus
            | Token::DoubleHyphen => {
                let operator = self.parse_unop()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Unary(operator, Box::new(inner_exp)))
            }
            Token::OpenParen if is_type_specifier(self.tokens.peek_nth(1)?) => {
                // It's a cast expression
                self.tokens.take_token()?;
                let type_specifiers = self.parse_type_specifier_list()?;
                let target_type = self.parse_type(type_specifiers)?;
                self.expect(Token::CloseParen)?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Cast {
                    target_type,
                    e: Box::new(inner_exp),
                })
            }
            _ => self.parse_postfix_exp(),
        }
    }

    // <argument-list> ::= <exp> { "," <exp> }
    fn parse_argument_list(&mut self) -> Result<Vec<Exp>, String> {
        let mut args = vec![];

        args.push(self.parse_expression(0)?);

        while self.tokens.peek()? == &Token::Comma {
            self.tokens.take_token()?;
            args.push(self.parse_expression(0)?);
        }

        Ok(args)
    }

    // <exp> ::= <factor> | <exp> <binop> <exp>
    fn parse_expression(&mut self, min_prec: i8) -> Result<Exp, String> {
        let mut left = self.parse_factor()?;
        let mut next_token = self.tokens.peek()?.clone();

        while let Some(prec) = get_precedence(&next_token) {
            if prec < min_prec {
                break;
            }

            if is_assignment(&next_token) {
                self.tokens.take_token()?;
                let right = self.parse_expression(prec)?;
                left = match get_compound_operator(&next_token) {
                    None => Exp::Assignment(left.into(), right.into()),
                    Some(op) => Exp::CompoundAssign(op, left.into(), right.into()),
                };
            } else if next_token == Token::QuestionMark {
                let middle = self.parse_conditional_middle()?;
                let right = self.parse_expression(prec)?;
                left = Exp::Conditional {
                    condition: left.into(),
                    then_result: middle.into(),
                    else_result: right.into(),
                };
            } else {
                let operator = self.parse_binop()?;
                let right = self.parse_expression(prec + 1)?;
                left = Exp::Binary(operator, Box::new(left), Box::new(right));
            }

            next_token = self.tokens.peek()?.clone();
        }

        Ok(left)
    }

    fn parse_optional_exp(&mut self, delim: Token) -> Result<Option<Exp>, String> {
        if self.tokens.peek()? == &delim {
            self.tokens.take_token()?;
            return Ok(None);
        } else {
            let e = self.parse_expression(0)?;
            self.expect(delim)?;
            Ok(Some(e))
        }
    }

    // Declarations

    // <param-list> ::= "void"
    //               | { <type-specifier> }+ <identifier> { "," { <type-specifier> }+ <identifier> }
    fn parse_param_list(&mut self) -> Result<Vec<(Type, String)>, String> {
        if self.tokens.peek()? == &Token::KWVoid {
            self.tokens.take_token()?;
            Ok(vec![])
        } else {
            let mut params = vec![];

            loop {
                let specifiers = self.parse_type_specifier_list()?;
                let next_param_t = self.parse_type(specifiers)?;
                let next_param_name = self.parse_id()?;
                let next_param = (next_param_t, next_param_name);
                params.push(next_param);

                match self.tokens.peek()? {
                    Token::Comma => {
                        self.tokens.take_token()?;
                    }
                    _ => break,
                }
            }

            Ok(params)
        }
    }

    // <function-declaration> ::= { <specifier> }+ <identifier> "(" <param-list ")" ( <block> | ";" )
    // <variable-declaration> ::= { <specifier> }+ <identifier> [ "=" <exp> ] ";"
    // Use a common function to parse both symbols
    fn parse_declaration(&mut self) -> Result<Declaration<Exp>, String> {
        let specifiers = self.parse_specifier_list()?;
        let (typ, storage_class) = self.parse_type_and_storage_class(specifiers)?;
        let name = self.parse_id()?;
        match self.tokens.peek()? {
            // It's a function declaration
            Token::OpenParen => {
                self.tokens.take_token()?;
                let params_with_types = self.parse_param_list()?;
                let (param_types, param_names): (Vec<Type>, Vec<String>) =
                    params_with_types.into_iter().unzip();
                let fun_type = Type::FunType {
                    param_types,
                    ret_type: Box::new(typ),
                };
                self.expect(Token::CloseParen)?;
                let body = if let Ok(Token::Semicolon) = self.tokens.peek() {
                    self.tokens.take_token()?;
                    None
                } else {
                    Some(self.parse_block()?)
                };
                Ok(Declaration::FunDecl(FunctionDeclaration {
                    name,
                    fun_type,
                    storage_class,
                    params: param_names,
                    body,
                }))
            }
            // It's a variable
            tok => {
                let init = if tok == &Token::EqualSign {
                    self.tokens.take_token()?;
                    Some(self.parse_expression(0)?)
                } else {
                    None
                };
                self.expect(Token::Semicolon)?;
                Ok(Declaration::VarDecl(VariableDeclaration {
                    name,
                    var_type: typ,
                    init,
                    storage_class,
                }))
            }
        }
    }

    // Statements and blocks

    // <for-init> ::= <variable-declaration> | [ <exp> ] ";"
    fn parse_for_init(&mut self) -> Result<ForInit<Exp>, String> {
        if is_specifier(self.tokens.peek()?) {
            match self.parse_declaration()? {
                Declaration::VarDecl(vd) => Ok(ForInit::InitDecl(vd)),
                _ => {
                    return Err("Found a function declaration in a for loop header".to_string());
                }
            }
        } else {
            let opt_e = self.parse_optional_exp(Token::Semicolon)?;
            Ok(ForInit::InitExp(opt_e))
        }
    }

    // <statement> ::= "return" <exp> ";"
    //              | "if" "(" <exp> ")" <statement> [ "else" <statement> ]
    //              | <label> ":" <statement> ";"
    //              | "goto" <label> ";"
    //              | <block>
    //              | <break> ";"
    //              | <continue> ";"
    //              | "while" "(" <exp> ")" <statement>
    //              | "do" <statement> "while" "(" <exp> ")" ";"
    //              | "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
    //              | "switch" "(" <exp> ")" <statement>
    //              | "case" <int> ":" <statement>
    //              | "default" ":" <statement>
    //              | <exp> ";"
    //              | ";"
    fn parse_statement(&mut self) -> Result<Statement<Exp>, String> {
        match self.tokens.npeek(2) {
            [Token::KWIf, _] => self.parse_if_statement(),
            [Token::OpenBrace, _] => Ok(Statement::Compound(self.parse_block()?)),
            [Token::KWDo, _] => self.parse_do_loop(),
            [Token::KWWhile, _] => self.parse_while_loop(),
            [Token::KWFor, _] => self.parse_for_loop(),
            [Token::KWBreak, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Break("".to_string()))
            }
            [Token::KWContinue, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Continue("".to_string()))
            }
            [Token::KWReturn, _] => {
                // consume return keyword
                self.tokens.take_token()?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Return(exp))
            }
            [Token::KWGoto, _] => {
                self.tokens.take_token()?;
                let lbl = self.parse_id()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Goto(lbl))
            }
            [Token::KWSwitch, _] => self.parse_switch_statement(),
            [Token::KWCase, _] => {
                self.tokens.take_token()?;
                let case_val = self.parse_expression(0)?;
                self.expect(Token::Colon)?;
                let stmt = Box::new(self.parse_statement()?);
                Ok(Statement::Case(case_val, stmt, "".to_string()))
            }
            [Token::KWDefault, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Colon)?;
                let stmt = Box::new(self.parse_statement()?);
                Ok(Statement::Default(stmt, "".to_string()))
            }
            [Token::Identifier(_), Token::Colon] => {
                // consume label
                let tok = self.tokens.take_token()?;
                let lbl = match tok {
                    Token::Identifier(name) => name,
                    other => {
                        return Err(format!("Expected identifier, found {:?}", other));
                    }
                };
                self.tokens.take_token()?; // consume colon
                let stmt = self.parse_statement()?;
                Ok(Statement::LabelledStatement(lbl, Box::new(stmt)))
            }
            _ => {
                let exp = self.parse_optional_exp(Token::Semicolon)?;
                if let Some(e) = exp {
                    Ok(Statement::Expression(e))
                } else {
                    Ok(Statement::Null)
                }
            }
        }
    }

    // "if" "(" <exp> ")" <statement> [ "else" <statement> ]
    fn parse_if_statement(&mut self) -> Result<Statement<Exp>, String> {
        self.expect(Token::KWIf)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let then_clause = self.parse_statement()?;

        let else_clause = if let Ok(Token::KWElse) = self.tokens.peek() {
            // there is an else clause - consume the else keyword
            self.tokens.take_token()?;
            Some(Box::new(self.parse_statement()?))
        } else {
            // there's no else clause
            None
        };

        Ok(Statement::If {
            condition,
            then_clause: then_clause.into(),
            else_clause,
        })
    }

    // "do" <statement> "while" "(" <exp> ")" ";"
    fn parse_do_loop(&mut self) -> Result<Statement<Exp>, String> {
        self.expect(Token::KWDo)?;
        let body = Box::new(self.parse_statement()?);
        self.expect(Token::KWWhile)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        self.expect(Token::Semicolon)?;
        Ok(Statement::DoWhile {
            body,
            condition,
            id: "".to_string(),
        })
    }

    // "while" "(" <exp> ")" <statement>
    fn parse_while_loop(&mut self) -> Result<Statement<Exp>, String> {
        self.expect(Token::KWWhile)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::While {
            condition,
            body,
            id: "".to_string(),
        })
    }

    // "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
    fn parse_for_loop(&mut self) -> Result<Statement<Exp>, String> {
        self.expect(Token::KWFor)?;
        self.expect(Token::OpenParen)?;
        let init = self.parse_for_init()?;
        let condition = self.parse_optional_exp(Token::Semicolon)?;
        let post = self.parse_optional_exp(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::For {
            init,
            condition,
            post,
            body,
            id: "".to_string(),
        })
    }

    // "switch" "(" <exp> ")" <statement>
    fn parse_switch_statement(&mut self) -> Result<Statement<Exp>, String> {
        self.expect(Token::KWSwitch)?;
        self.expect(Token::OpenParen)?;
        let control = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::Switch {
            control,
            body,
            cases: Vec::new(),
            id: "".to_string(),
        })
    }

    // <block_item> ::= <statement> | <declaration>
    fn parse_block_item(&mut self) -> Result<BlockItem<Exp>, String> {
        if is_specifier(self.tokens.peek()?) {
            Ok(BlockItem::D(self.parse_declaration()?))
        } else {
            Ok(BlockItem::S(self.parse_statement()?))
        }
    }

    fn parse_block(&mut self) -> Result<Block, String> {
        self.expect(Token::OpenBrace)?;
        let mut block = Vec::new();
        while self.tokens.peek()? != &Token::CloseBrace {
            let block_item = self.parse_block_item()?;
            block.push(block_item);
        }
        self.expect(Token::CloseBrace)?;
        Ok(BlockStruct(block))
    }

    // Top level

    // <program> ::= { <declaration> }*
    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut fun_decls = vec![];

        while !self.tokens.is_empty() {
            let next_decl = self.parse_declaration()?;
            fun_decls.push(next_decl);
        }

        Ok(ProgramStruct(fun_decls))
    }
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn signed_long_constant() {
        let tok_list = vec![Token::ConstLong(
            BigInt::from_i64(4611686018427387904).unwrap(),
        )];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_const().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstLong(4611686018427387904)));
    }

    #[test]
    fn unsigned_int_constant() {
        let tok_list = vec![Token::ConstUInt(BigInt::from_str("4294967291").unwrap())];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_const().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstUInt(4294967291)));
    }

    #[test]
    fn unsigned_long_constant() {
        let tok_list = vec![Token::ConstULong(
            BigInt::from_str("18446744073709551611").unwrap(),
        )];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_const().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstULong(18446744073709551611)));
    }

    #[test]
    fn expression() {
        let mut parser = Parser::new(vec![Token::ConstInt(BigInt::from(100)), Token::Semicolon]);
        assert_eq!(
            parser.parse_expression(40).unwrap(),
            Exp::Constant(T::ConstInt(100))
        );
    }

    #[test]
    fn statement() {
        let mut parser = Parser::new(vec![
            Token::KWReturn,
            Token::ConstInt(BigInt::from(4)),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::Return(Exp::Constant(T::ConstInt(4)))
        );
    }

    #[test]
    #[should_panic]
    fn error() {
        let mut parser = Parser::new(vec![Token::KWInt]);
        let _ = parser.parse_program();
    }

    #[test]
    fn labelled_statement() {
        let mut parser = Parser::new(vec![
            Token::Identifier("label".to_string()),
            Token::Colon,
            Token::KWReturn,
            Token::ConstInt(BigInt::from(42)),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::LabelledStatement(
                "label".to_string(),
                Box::new(Statement::Return(Exp::Constant(T::ConstInt(42)))),
            )
        );
    }

    #[test]
    fn goto_statement() {
        let mut parser = Parser::new(vec![
            Token::KWGoto,
            Token::Identifier("label".to_string()),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::Goto("label".to_string())
        );
    }
}
