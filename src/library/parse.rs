use std::collections::HashSet;

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::library::{ast::storage_class::StorageClass, r#const::T, types::Type};

use super::{
    ast::block_items::{Block as BlockStruct, Program as ProgramStruct},
    ast::block_items::{
        BlockItem, Declaration, ForInit, FunctionDeclaration, Statement, VariableDeclaration,
    },
    ast::ops::{BinaryOperator, CompoundAssignOperator, UnaryOperator},
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
        | Token::MinusEqual
        | Token::StarEqual
        | Token::SlashEqual
        | Token::PercentEqual
        | Token::AmpersandEqual
        | Token::PipeEqual
        | Token::CaretEqual
        | Token::LeftShiftEqual
        | Token::RightShiftEqual => Some(1),
        _ => None,
    }
}

// Helper function to check whether a token is a type specifier
fn is_type_specifier(token: &Token) -> bool {
    matches!(token, Token::KWInt | Token::KWLong)
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

    // <type-specifier> ::= "int" | "long"
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
        match specifier_list.as_slice() {
            [Token::KWInt] => Ok(Type::Int),
            [Token::KWInt, Token::KWLong]
            | [Token::KWLong, Token::KWInt]
            | [Token::KWLong]
            | [Token::KWLong, Token::KWLong]
            | [Token::KWLong, Token::KWLong, Token::KWInt] => Ok(Type::Long),
            _ => Err("Invalid type specifier".to_string()),
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

    // <const> ::= <int> | <long>
    // Convert a single token to into constant AST node
    fn parse_const(&mut self) -> Result<Exp, String> {
        let (v, is_int) = match self.tokens.take_token()? {
            Token::ConstInt(i) => (i, true),
            Token::ConstLong(l) => (l, false),
            other => {
                return Err(format!("Expected a constant, found {:?}", other));
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

    // Expressions

    // <unop> ::= "-" | "~" | "!"
    fn parse_unop(&mut self) -> Result<UnaryOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::Tilde) => Ok(UnaryOperator::Complement),
            Ok(Token::Hyphen) => Ok(UnaryOperator::Negate),
            Ok(Token::Bang) => Ok(UnaryOperator::Not),
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
            Ok(Token::Caret) => Ok(BinaryOperator::Xor),
            Ok(Token::LeftShift) => Ok(BinaryOperator::LeftShift),
            Ok(Token::RightShift) => Ok(BinaryOperator::RightShift),
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

    fn parse_compound_assign_op(&mut self) -> Result<CompoundAssignOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::PlusEqual) => Ok(CompoundAssignOperator::PlusEqual),
            Ok(Token::MinusEqual) => Ok(CompoundAssignOperator::MinusEqual),
            Ok(Token::StarEqual) => Ok(CompoundAssignOperator::StarEqual),
            Ok(Token::SlashEqual) => Ok(CompoundAssignOperator::SlashEqual),
            Ok(Token::PercentEqual) => Ok(CompoundAssignOperator::PercentEqual),
            Ok(Token::AmpersandEqual) => Ok(CompoundAssignOperator::AmpersandEqual),
            Ok(Token::PipeEqual) => Ok(CompoundAssignOperator::PipeEqual),
            Ok(Token::CaretEqual) => Ok(CompoundAssignOperator::CaretEqual),
            Ok(Token::LeftShiftEqual) => Ok(CompoundAssignOperator::LeftShiftEqual),
            Ok(Token::RightShiftEqual) => Ok(CompoundAssignOperator::RightShiftEqual),
            other => Err(format!(
                "Expected a compound assignment operator, found {:?}",
                other.unwrap()
            )),
        }
    }

    // Helper function to parse postfix increment and decrement
    fn parse_postfix_increment(&mut self, exp: Exp) -> Result<Exp, String> {
        match self.tokens.peek()? {
            Token::DoublePlus => {
                self.tokens.take_token()?;
                return Ok(Exp::PostfixIncrement(Box::new(exp)));
            }
            Token::DoubleHyphen => {
                self.tokens.take_token()?;
                return Ok(Exp::PostfixDecrement(Box::new(exp)));
            }
            _ => return Ok(exp),
        }
    }

    // Helper function to parse the middle of a conditional expression:
    // "?" <exp> ":"
    fn parse_conditional_middle(&mut self) -> Result<Exp, String> {
        self.expect(Token::QuestionMark)?;
        let e = self.parse_exp(0)?;
        self.expect(Token::Colon)?;
        Ok(e)
    }

    // <factor> ::= <const> | <identifier>
    //              | "(" { <type-specifier> }+ ")" <factor>
    //              | <unop> <factor> | "(" <exp> ")"
    //              | <identifier> "(" [ <argument-list ] ")"
    fn parse_factor(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // constant
            Token::ConstInt(_) | Token::ConstLong(_) => self.parse_const(),
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

                self.parse_postfix_increment(exp)
            }
            // unary expression
            Token::Hyphen | Token::Tilde | Token::Bang => {
                let operator = self.parse_unop()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Unary(operator, Box::new(inner_exp)))
            }
            // prefix increment
            Token::DoublePlus => {
                self.tokens.take_token()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::PrefixIncrement(Box::new(inner_exp)))
            }
            // prefix decrement
            Token::DoubleHyphen => {
                self.tokens.take_token()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::PrefixDecrement(Box::new(inner_exp)))
            }
            // cast or parenthesized expression
            Token::OpenParen => {
                // Consume open paren
                let _ = self.tokens.take_token();
                if is_type_specifier(self.tokens.peek()?) {
                    // It's a cast expression
                    let type_specifiers = self.parse_type_specifier_list()?;
                    let target_type = self.parse_type(type_specifiers)?;
                    self.expect(Token::CloseParen)?;
                    let inner_exp = self.parse_factor()?;
                    Ok(Exp::Cast {
                        target_type,
                        e: Box::new(inner_exp),
                    })
                } else {
                    // It's parenthesized
                    if let Token::Identifier(_) = self.tokens.peek()? {
                        if self.tokens.peek_nth(1)? == &Token::CloseParen {
                            let id = self.parse_id()?;
                            let _ = self.tokens.take_token();
                            return self.parse_postfix_increment(Exp::Var(id));
                        }
                    }
                    let e = self.parse_exp(0);
                    let _ = self.expect(Token::CloseParen);
                    e
                }
            }
            t => Err(format!("Expected a factor, found {:?}", t)),
        }
    }

    // <argument-list> ::= <exp> { "," <exp> }
    fn parse_argument_list(&mut self) -> Result<Vec<Exp>, String> {
        let mut args = vec![];

        args.push(self.parse_exp(0)?);

        while self.tokens.peek()? == &Token::Comma {
            self.tokens.take_token()?;
            args.push(self.parse_exp(0)?);
        }

        Ok(args)
    }

    fn parse_exp(&mut self, min_prec: i8) -> Result<Exp, String> {
        let mut left = self.parse_factor()?;
        let mut next_token = self.tokens.peek()?;

        while let Some(prec) = get_precedence(&next_token) {
            if prec < min_prec {
                break;
            }

            match next_token {
                Token::EqualSign => {
                    self.tokens.take_token()?;
                    let right = self.parse_exp(prec)?;
                    left = Exp::Assignment(Box::new(left), Box::new(right));
                }
                Token::PlusEqual
                | Token::MinusEqual
                | Token::StarEqual
                | Token::SlashEqual
                | Token::PercentEqual
                | Token::AmpersandEqual
                | Token::PipeEqual
                | Token::CaretEqual
                | Token::LeftShiftEqual
                | Token::RightShiftEqual => {
                    let operator = self.parse_compound_assign_op()?;
                    let right = self.parse_exp(prec)?;
                    left = Exp::CompoundAssign(operator, Box::new(left), Box::new(right));
                }
                Token::QuestionMark => {
                    let middle = self.parse_conditional_middle()?;
                    let right = self.parse_exp(prec)?;
                    left = Exp::Conditional {
                        condition: left.into(),
                        then_result: middle.into(),
                        else_result: right.into(),
                    };
                }
                _ => {
                    let operator = self.parse_binop()?;
                    let right = self.parse_exp(prec + 1)?;
                    left = Exp::Binary(operator, Box::new(left), Box::new(right));
                }
            }

            next_token = self.tokens.peek()?;
        }

        Ok(left)
    }

    fn parse_optional_exp(&mut self, delim: Token) -> Result<Option<Exp>, String> {
        if self.tokens.peek()? == &delim {
            self.tokens.take_token()?;
            return Ok(None);
        } else {
            let e = self.parse_exp(0)?;
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
                    Some(self.parse_exp(0)?)
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
    //              | "goto" <identifier> ";"
    //              | <identifier> ":" <statement>
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
        match self.tokens.peek() {
            // "return" <exp> ";"
            Ok(Token::KWReturn) => {
                // consume return keyword
                self.tokens.take_token()?;
                let exp = self.parse_exp(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Return(exp))
            }
            // "if" "(" <exp> ")" <statement> [ "else" <statement> ]
            Ok(Token::KWIf) => {
                // if statement - consume if keyword
                self.tokens.take_token()?;
                self.expect(Token::OpenParen)?;
                let condition = self.parse_exp(0)?;
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
            // "switch" "(" <exp> ")" <statement>
            Ok(Token::KWSwitch) => {
                // consume switch keyword
                self.tokens.take_token()?;
                self.expect(Token::OpenParen)?;
                let condition = self.parse_exp(0)?;
                self.expect(Token::CloseParen)?;
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::Switch {
                    condition,
                    body,
                    cases: HashSet::new(),
                    id: "".to_string(),
                })
            }
            // labelled statement
            Ok(Token::Identifier(_)) => {
                if self.tokens.peek_nth(1)? == &Token::Colon {
                    let label = self.parse_id()?;
                    self.expect(Token::Colon)?;
                    let statement = self.parse_statement()?;
                    Ok(Statement::Labelled {
                        label: label.clone(),
                        statement: Box::new(statement),
                    })
                } else {
                    let exp = self.parse_exp(0)?;
                    self.expect(Token::Semicolon)?;
                    Ok(Statement::Expression(exp))
                }
            }
            // "case" <int> ":" <statement>
            Ok(Token::KWCase) => {
                self.tokens.take_token()?;
                let condition = match self.tokens.take_token()? {
                    Token::ConstInt(c) | Token::ConstLong(c) => c.to_i64().unwrap(),
                    other => {
                        return Err(format!("Expected a constant, found {:?}", other));
                    }
                };
                self.expect(Token::Colon)?;
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::Case {
                    condition,
                    body,
                    switch_label: "".to_string(),
                })
            }
            // "default" ":"
            Ok(Token::KWDefault) => {
                self.tokens.take_token()?;
                self.expect(Token::Colon)?;
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::Default {
                    body,
                    switch_label: "".to_string(),
                })
            }
            // "goto" <identifier> ";"
            Ok(Token::KWGoto) => {
                self.tokens.take_token()?;
                let label = self.parse_id()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Goto(label))
            }
            Ok(Token::OpenBrace) => Ok(Statement::Compound(self.parse_block()?)),
            // "break" ";"
            Ok(Token::KWBreak) => {
                // consume break keyword
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Break("".to_string()))
            }
            // "continue" ";"
            Ok(Token::KWContinue) => {
                // consume continue keyword
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Continue("".to_string()))
            }
            // "while" "(" <exp> ")" <statement>
            Ok(Token::KWWhile) => {
                // consume while keyword
                self.tokens.take_token()?;
                self.expect(Token::OpenParen)?;
                let condition = self.parse_exp(0)?;
                self.expect(Token::CloseParen)?;
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::While {
                    condition,
                    body,
                    id: "".to_string(),
                })
            }
            // "do" <statement> "while" "(" <exp> ")" ";"
            Ok(Token::KWDo) => {
                // consume do keyword
                self.tokens.take_token()?;
                let body = Box::new(self.parse_statement()?);
                self.expect(Token::KWWhile)?;
                self.expect(Token::OpenParen)?;
                let condition = self.parse_exp(0)?;
                self.expect(Token::CloseParen)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::DoWhile {
                    body,
                    condition,
                    id: "".to_string(),
                })
            }
            // "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
            Ok(Token::KWFor) => {
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
            // <exp> ";" | ";"
            Ok(_) => {
                let exp = self.parse_optional_exp(Token::Semicolon)?;
                if let Some(e) = exp {
                    Ok(Statement::Expression(e))
                } else {
                    Ok(Statement::Null)
                }
            }
            _ => {
                // No tokens left
                return Err("No tokens left".to_string());
            }
        }
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
    use super::*;

    #[test]
    fn expression() {
        let mut parser = Parser::new(vec![Token::ConstInt(BigInt::from(100)), Token::Semicolon]);
        assert_eq!(
            parser.parse_exp(40).unwrap(),
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
            Statement::Labelled {
                label: "label".to_string(),
                statement: Box::new(Statement::Return(Exp::Constant(T::ConstInt(42)))),
            }
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
