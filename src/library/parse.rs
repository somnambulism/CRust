use super::{
    ast::{Exp, FunctionDefinition, Program, Statement, UnaryOperator},
    tok_stream::TokenStream,
    tokens::Token,
};

pub struct Parser {
    tokens: TokenStream,
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

    fn parse_unop(&mut self) -> Result<UnaryOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::Tilde) => Ok(UnaryOperator::Complement),
            Ok(Token::Hyphen) => Ok(UnaryOperator::Negate),
            other => Err(format!(
                "Expected a unary operator, found {:?}",
                other.unwrap()
            )),
        }
    }

    fn parse_exp(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            Token::Constant(_) => self.parse_int(),
            Token::Hyphen | Token::Tilde => {
                let operator = self.parse_unop()?;
                let inner_exp = self.parse_exp()?;
                Ok(Exp::Unary(operator, Box::new(inner_exp)))
            }
            Token::OpenParen => {
                let _ = self.tokens.take_token();
                let e = self.parse_exp();
                let _ = self.expect(Token::CloseParen);
                e
            }
            t => Err(format!("Expected an expression, found {:?}", t)),
        }
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::KWReturn)?;
        let exp = self.parse_exp()?;
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
        let mut parser = Parser::new(vec![Token::Constant(100)]);
        assert_eq!(parser.parse_exp().unwrap(), Exp::Constant(100));
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
