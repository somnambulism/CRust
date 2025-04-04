use super::{
    ast::{BinaryOperator, Exp, FunctionDefinition, Program, Statement, UnaryOperator},
    tok_stream::TokenStream,
    tokens::Token,
};

pub struct Parser {
    tokens: TokenStream,
}

fn get_precedence(token: &Token) -> Option<i32> {
    match token {
        Token::Star | Token::Slash | Token::Percent => Some(50),
        Token::Plus | Token::Hyphen => Some(45),
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

    // <factor> ::= <int> | <unop> <factor> | "(" <exp> ")"
    fn parse_factor(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // constant
            Token::Constant(_) => self.parse_int(),
            // unary expression
            Token::Hyphen | Token::Tilde | Token::Bang => {
                let operator = self.parse_unop()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Unary(operator, Box::new(inner_exp)))
            }
            Token::OpenParen => {
                let _ = self.tokens.take_token();
                let e = self.parse_exp(0);
                let _ = self.expect(Token::CloseParen);
                e
            }
            t => Err(format!("Expected a factor, found {:?}", t)),
        }
    }

    fn parse_exp(&mut self, min_prec: i32) -> Result<Exp, String> {
        let mut left = self.parse_factor()?;

        while let Ok(next_token) = self.tokens.peek() {
            if let Some(prec) = get_precedence(&next_token) {
                if prec < min_prec {
                    break;
                }

                // Consume operator
                let operator = self.parse_binop()?;
                let right = self.parse_exp(prec + 1)?;

                left = Exp::Binary(operator, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::KWReturn)?;
        let exp = self.parse_exp(0)?;
        self.expect(Token::Semicolon)?;
        Ok(Statement::Return(exp))
    }

    fn parse_function_definition(&mut self) -> Result<FunctionDefinition, String> {
        self.expect(Token::KWInt)?;
        let fun_name = self.parse_id()?;
        self.expect(Token::OpenParen)?;
        self.expect(Token::KWVoid)?;
        self.expect(Token::CloseParen)?;
        self.expect(Token::OpenBrace)?;
        let statement = self.parse_statement()?;
        self.expect(Token::CloseBrace)?;
        Ok(FunctionDefinition {
            name: fun_name,
            body: statement,
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
