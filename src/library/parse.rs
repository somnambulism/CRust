use super::{
    ast::{Exp, FunctionDefinition, Program, Statement},
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

    fn parse_exp(&mut self) -> Result<Exp, String> {
        self.parse_int()
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
