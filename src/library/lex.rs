use super::tokens::Token;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct TokenDef {
    re: Regex,
    converter: fn(&str) -> Token,
}

#[derive(Debug)]
pub struct MatchDef {
    matched_substring: String,
    matching_token: TokenDef,
}

pub struct Lexer {
    token_defs: Vec<TokenDef>,
}

impl Lexer {
    pub fn new() -> Self {
        let token_defs = vec![
            TokenDef {
                re: Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*\b").unwrap(),
                converter: Lexer::convert_identifier,
            },
            TokenDef {
                re: Regex::new(r"^[0-9]+\b").unwrap(),
                converter: Lexer::convert_int,
            },
            TokenDef {
                re: Regex::new(r"^\(").unwrap(),
                converter: |_s| Token::OpenParen,
            },
            TokenDef {
                re: Regex::new(r"^\)").unwrap(),
                converter: |_s| Token::CloseParen,
            },
            TokenDef {
                re: Regex::new(r"^\{").unwrap(),
                converter: |_s| Token::OpenBrace,
            },
            TokenDef {
                re: Regex::new(r"^\}").unwrap(),
                converter: |_s| Token::CloseBrace,
            },
            TokenDef {
                re: Regex::new(r"^;").unwrap(),
                converter: |_s| Token::Semicolon,
            },
        ];
        Lexer { token_defs }
    }

    pub fn lex(&self, input: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let mut input = input.to_string();

        while !input.is_empty() {
            if let Some(ws_count) = count_leading_ws(&input) {
                input = input[ws_count..].to_string();
            }
            if input.is_empty() {
                break;
            }
            match self.find_match(&input) {
                Some(match_def) => {
                    let token = (match_def.matching_token.converter)(&match_def.matched_substring);
                    tokens.push(token);
                    input = input[match_def.matched_substring.len()..].to_string();
                }
                None => return Err(format!("Lexical error at: {}", input)),
            }
        }

        Ok(tokens)
    }

    fn find_match(&self, input: &str) -> Option<MatchDef> {
        for token_def in &self.token_defs {
            if let Some(m) = token_def.re.find(input) {
                return Some(MatchDef {
                    matched_substring: m.as_str().to_string(),
                    matching_token: token_def.clone(),
                });
            }
        }
        None
    }

    fn convert_identifier(s: &str) -> Token {
        match s {
            "int" => Token::KWInt,
            "return" => Token::KWReturn,
            "void" => Token::KWVoid,
            _ => Token::Identifier(s.to_string()),
        }
    }

    fn convert_int(s: &str) -> Token {
        Token::Constant(s.parse::<i32>().unwrap())
    }
}

fn count_leading_ws(s: &str) -> Option<usize> {
    let re = regex::Regex::new(r"^\s+").unwrap();
    if let Some(mat) = re.find(s) {
        Some(mat.end())
    } else {
        None
    }
}
