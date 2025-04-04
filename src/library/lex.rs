use super::tokens::Token;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct TokenDef {
    re: Regex,
    converter: fn(&str) -> Token,
}

#[derive(Clone, Debug)]
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
            TokenDef {
                re: Regex::new("^--").unwrap(),
                converter: |_s| Token::DoubleHyphen,
            },
            TokenDef {
                re: Regex::new("^<<").unwrap(),
                converter: |_s| Token::LeftShift,
            },
            TokenDef {
                re: Regex::new("^>>").unwrap(),
                converter: |_s| Token::RightShift,
            },
            TokenDef {
                re: Regex::new("^&&").unwrap(),
                converter: |_s| Token::LogicalAnd,
            },
            TokenDef {
                re: Regex::new(r"^\|\|").unwrap(),
                converter: |_s| Token::LogicalOr,
            },
            TokenDef {
                re: Regex::new("^==").unwrap(),
                converter: |_s| Token::DoubleEqual,
            },
            TokenDef {
                re: Regex::new("^!=").unwrap(),
                converter: |_s| Token::NotEqual,
            },
            TokenDef {
                re: Regex::new("^<=").unwrap(),
                converter: |_s| Token::LessOrEqual,
            },
            TokenDef {
                re: Regex::new("^>=").unwrap(),
                converter: |_s| Token::GreaterOrEqual,
            },
            TokenDef {
                re: Regex::new("^-").unwrap(),
                converter: |_s| Token::Hyphen,
            },
            TokenDef {
                re: Regex::new("^~").unwrap(),
                converter: |_s| Token::Tilde,
            },
            TokenDef {
                re: Regex::new(r"^\+").unwrap(),
                converter: |_s| Token::Plus,
            },
            TokenDef {
                re: Regex::new(r"^\*").unwrap(),
                converter: |_s| Token::Star,
            },
            TokenDef {
                re: Regex::new("^/").unwrap(),
                converter: |_s| Token::Slash,
            },
            TokenDef {
                re: Regex::new("^%").unwrap(),
                converter: |_s| Token::Percent,
            },
            TokenDef {
                re: Regex::new("^&").unwrap(),
                converter: |_s| Token::Ampersand,
            },
            TokenDef {
                re: Regex::new(r"^\|").unwrap(),
                converter: |_s| Token::Pipe,
            },
            TokenDef {
                re: Regex::new(r"^\^").unwrap(),
                converter: |_s| Token::Caret,
            },
            TokenDef {
                re: Regex::new("^!").unwrap(),
                converter: |_s| Token::Bang,
            },
            TokenDef {
                re: Regex::new("^<").unwrap(),
                converter: |_s| Token::LessThan,
            },
            TokenDef {
                re: Regex::new("^>").unwrap(),
                converter: |_s| Token::GreaterThan,
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
    let ws_matcher = regex::Regex::new(r"^\s+").unwrap();
    ws_matcher.find(s).map(|m| m.end())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_whitespace() {
        let lexer = Lexer::new();
        assert_eq!(lexer.lex("  return").unwrap(), vec![Token::KWReturn]);
    }

    #[test]
    fn trailing_whitespace() {
        let lexer = Lexer::new();
        assert_eq!(
            lexer.lex("0;\t\n").unwrap(),
            vec![Token::Constant(0), Token::Semicolon]
        );
    }

    #[test]
    fn a_full_program() {
        let lexer = Lexer::new();
        assert_eq!(
            lexer.lex("int main(void){return 0;}").unwrap(),
            vec![
                Token::KWInt,
                Token::Identifier("main".to_string()),
                Token::OpenParen,
                Token::KWVoid,
                Token::CloseParen,
                Token::OpenBrace,
                Token::KWReturn,
                Token::Constant(0),
                Token::Semicolon,
                Token::CloseBrace,
            ]
        )
    }

    #[test]
    fn two_hyphens() {
        let lexer = Lexer::new();
        assert_eq!(
            lexer.lex("- -").unwrap(),
            vec![Token::Hyphen, Token::Hyphen]
        );
    }

    #[test]
    fn double_hyphen() {
        let lexer = Lexer::new();
        assert_eq!(lexer.lex("--").unwrap(), vec![Token::DoubleHyphen]);
    }

    #[test]
    fn two_tildes() {
        let lexer = Lexer::new();
        assert_eq!(lexer.lex("~~").unwrap(), vec![Token::Tilde, Token::Tilde]);
    }
}
