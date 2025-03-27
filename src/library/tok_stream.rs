use std::vec::IntoIter;

use super::tokens::Token;

#[derive(Debug)]
pub struct TokenStream {
    tokens: IntoIter<Token>,
}

impl TokenStream {
    pub fn new(tokens: Vec<Token>) -> Self {
        TokenStream {
            tokens: tokens.into_iter(),
        }
    }

    pub fn take_token(&mut self) -> Result<Token, String> {
        self.tokens
            .next()
            .ok_or_else(|| "End of stream".to_string())
    }

    pub fn peek(&self) -> Result<&Token, String> {
        self.tokens
            .as_slice()
            .first()
            .ok_or_else(|| "End of steram".to_string())
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.clone().count() == 0
    }
}
