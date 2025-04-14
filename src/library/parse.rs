use super::{
    ast::{
        BinaryOperator, BlockItem, CompoundAssignOperator, Declaration, Exp, FunctionDefinition,
        Program, Statement, UnaryOperator,
    },
    tok_stream::TokenStream,
    tokens::Token,
};

pub struct Parser {
    tokens: TokenStream,
}

fn get_precedence(token: &Token) -> Option<i32> {
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

    // Expressions

    // <int> ::= ? A constant token ?
    fn parse_int(&mut self) -> Result<Exp, String> {
        match self.tokens.take_token() {
            Ok(Token::Constant(c)) => Ok(Exp::Constant(c)),
            other => Err(format!("Expected constant, found {:?}", other.unwrap())),
        }
    }

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

    fn parse_compound_assgin_op(&mut self) -> Result<CompoundAssignOperator, String> {
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
    fn parse_postfix_increment(&mut self, id: String) -> Result<Exp, String> {
        match self.tokens.peek()? {
            Token::DoublePlus => {
                self.tokens.take_token()?;
                return Ok(Exp::PostfixIncrement(Box::new(Exp::Var(id))));
            }
            Token::DoubleHyphen => {
                self.tokens.take_token()?;
                return Ok(Exp::PostfixDecrement(Box::new(Exp::Var(id))));
            }
            _ => return Ok(Exp::Var(id)),
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

    // <factor> ::= <int> | <identifier> |<unop> <factor> | "(" <exp> ")"
    fn parse_factor(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // constant
            Token::Constant(_) => self.parse_int(),
            // identifier
            Token::Identifier(_) => {
                let id = self.parse_id()?;
                self.parse_postfix_increment(id)
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
            Token::OpenParen => {
                let _ = self.tokens.take_token();
                if let Token::Identifier(_) = self.tokens.peek()? {
                    if self.tokens.peek_nth(1)? == &Token::CloseParen {
                        let id = self.parse_id()?;
                        let _ = self.tokens.take_token();
                        return self.parse_postfix_increment(id);
                    }
                }
                let e = self.parse_exp(0);
                let _ = self.expect(Token::CloseParen);
                e
            }
            t => Err(format!("Expected a factor, found {:?}", t)),
        }
    }

    fn parse_exp(&mut self, min_prec: i32) -> Result<Exp, String> {
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
                    let operator = self.parse_compound_assgin_op()?;
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

    // <declaration> ::= "int" <identifier> [ "=" <exp> ] ";"
    fn parse_declaration(&mut self) -> Result<Declaration, String> {
        self.expect(Token::KWInt)?;
        let var_name = self.parse_id()?;

        let init = match self.tokens.take_token()? {
            // No initializer
            Token::Semicolon => None,
            // With initializer
            Token::EqualSign => {
                let init_exp = self.parse_exp(0)?;
                self.expect(Token::Semicolon)?;
                Some(init_exp)
            }
            // Unexpected token
            other => {
                return Err(format!(
                    "Expected an initializer or semicolon, found {:?}",
                    other
                ));
            }
        };

        Ok(Declaration {
            name: var_name,
            init,
        })
    }

    // <statement> ::= "return" <exp> ";"
    //              | "if" "(" <exp> ")" <statement> [ "else" <statement> ]
    //              | <exp> ";"
    //              | ";"
    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.tokens.peek() {
            // "return" <exp> ";"
            Ok(Token::KWReturn) => {
                // consume return keyword
                self.tokens.take_token()?;
                let exp = self.parse_exp(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Return(exp))
            }
            // ";"
            Ok(Token::Semicolon) => {
                // consume semicolon
                self.tokens.take_token()?;
                Ok(Statement::Null)
            }
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
            // <exp> ";"
            _ => {
                let exp = self.parse_exp(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Expression(exp))
            }
        }
    }

    // <block_item> ::= <statement> | <declaration>
    fn parse_block_item(&mut self) -> Result<BlockItem, String> {
        if self.tokens.peek() == Ok(&Token::KWInt) {
            Ok(BlockItem::D(self.parse_declaration()?))
        } else {
            Ok(BlockItem::S(self.parse_statement()?))
        }
    }

    // Top level

    // <function> ::= "int" <identifier> "(" "void" ")" "{" { <block-item> } "}"
    fn parse_function_definition(&mut self) -> Result<FunctionDefinition, String> {
        self.expect(Token::KWInt)?;
        let fun_name = self.parse_id()?;
        self.expect(Token::OpenParen)?;
        self.expect(Token::KWVoid)?;
        self.expect(Token::CloseParen)?;
        self.expect(Token::OpenBrace)?;

        let mut body = Vec::new();
        while self.tokens.peek()? != &Token::CloseBrace {
            let next_block_item = self.parse_block_item()?;
            body.push(next_block_item);
        }

        self.expect(Token::CloseBrace)?;
        Ok(FunctionDefinition {
            name: fun_name,
            body: body,
        })
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let fun_def = self.parse_function_definition()?;
        if self.tokens.is_empty() {
            Ok(Program { function: fun_def })
        } else {
            Err("Unexpected tokens after function definition".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression() {
        let mut parser = Parser::new(vec![Token::Constant(100), Token::Semicolon]);
        assert_eq!(parser.parse_exp(40).unwrap(), Exp::Constant(100));
    }

    #[test]
    fn statement() {
        let mut parser = Parser::new(vec![Token::KWReturn, Token::Constant(4), Token::Semicolon]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::Return(Exp::Constant(4))
        );
    }

    #[test]
    #[should_panic]
    fn error() {
        let mut parser = Parser::new(vec![Token::KWInt]);
        let _ = parser.parse_program();
    }

    #[test]
    fn empty() {
        let mut parser = Parser::new(vec![]);
        let result = parser.parse_program();
        assert!(result.is_err());
    }
}
