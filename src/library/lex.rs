use std::str::FromStr;

use crate::library::util::utils::string_util;

use super::tokens::Token;
use num_bigint::BigInt;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct TokenDef {
    re: Regex, // regular expression to recognize a token
    group: usize,
    /*
     * The match group within the regular expression that matches just the
     * token itself (and not any subsequent tokens that we need to consider
     * to recognize the match). In the common case where we don't need to
     * look at subsequent tokens, this is group 0, which matches the whole
     * regex. For constants, this is group 1.
     */
    converter: fn(&str) -> Token,
    // A function to convert the matched substring into a token
}

#[derive(Clone, Debug)]
pub struct MatchDef {
    matched_substring: String,
    // Substring matched the capture group specified in TokenDef
    matching_token: TokenDef, // Which token is matched
}

pub struct Lexer {
    token_defs: Vec<TokenDef>,
}

impl Lexer {
    pub fn new() -> Self {
        // List of token definitions
        let token_defs = vec![
            // all identifiers, including keywords
            TokenDef {
                re: Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*\b").unwrap(),
                group: 0,
                converter: Lexer::convert_identifier,
            },
            // constants
            TokenDef {
                re: Regex::new(r"^([0-9]+)[^\w.]").unwrap(),
                group: 1,
                converter: Lexer::convert_int,
            },
            TokenDef {
                re: Regex::new(r"^([0-9]+[lL])[^\w.]").unwrap(),
                group: 1,
                converter: Lexer::convert_long,
            },
            TokenDef {
                re: Regex::new(r"^([0-9]+[uU])[^\w.]").unwrap(),
                group: 1,
                converter: Lexer::convert_uint,
            },
            TokenDef {
                re: Regex::new(r"^([0-9]+([lL][uU]|[uU][lL]))[^\w.]").unwrap(),
                group: 1,
                converter: Lexer::convert_ulong,
            },
            TokenDef {
                re: Regex::new(
                    r"^(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)[^\w.]",
                )
                .unwrap(),
                group: 1,
                converter: Lexer::convert_double,
            },
            // punctuation
            TokenDef {
                re: Regex::new(r"^\(").unwrap(),
                group: 0,
                converter: |_s| Token::OpenParen,
            },
            TokenDef {
                re: Regex::new(r"^\)").unwrap(),
                group: 0,
                converter: |_s| Token::CloseParen,
            },
            TokenDef {
                re: Regex::new(r"^\{").unwrap(),
                group: 0,
                converter: |_s| Token::OpenBrace,
            },
            TokenDef {
                re: Regex::new(r"^\}").unwrap(),
                group: 0,
                converter: |_s| Token::CloseBrace,
            },
            TokenDef {
                re: Regex::new(r"^;").unwrap(),
                group: 0,
                converter: |_s| Token::Semicolon,
            },
            TokenDef {
                re: Regex::new("^--").unwrap(),
                group: 0,
                converter: |_s| Token::DoubleHyphen,
            },
            TokenDef {
                re: Regex::new(r"^\+\+").unwrap(),
                group: 0,
                converter: |_s| Token::DoublePlus,
            },
            TokenDef {
                re: Regex::new("^<<=").unwrap(),
                group: 0,
                converter: |_s| Token::LeftShiftEqual,
            },
            TokenDef {
                re: Regex::new("^>>=").unwrap(),
                group: 0,
                converter: |_s| Token::RightShiftEqual,
            },
            TokenDef {
                re: Regex::new("^<<").unwrap(),
                group: 0,
                converter: |_s| Token::LeftShift,
            },
            TokenDef {
                re: Regex::new("^>>").unwrap(),
                group: 0,
                converter: |_s| Token::RightShift,
            },
            TokenDef {
                re: Regex::new("^&&").unwrap(),
                group: 0,
                converter: |_s| Token::LogicalAnd,
            },
            TokenDef {
                re: Regex::new(r"^\|\|").unwrap(),
                group: 0,
                converter: |_s| Token::LogicalOr,
            },
            TokenDef {
                re: Regex::new("^==").unwrap(),
                group: 0,
                converter: |_s| Token::DoubleEqual,
            },
            TokenDef {
                re: Regex::new("^!=").unwrap(),
                group: 0,
                converter: |_s| Token::NotEqual,
            },
            TokenDef {
                re: Regex::new("^<=").unwrap(),
                group: 0,
                converter: |_s| Token::LessOrEqual,
            },
            TokenDef {
                re: Regex::new("^>=").unwrap(),
                group: 0,
                converter: |_s| Token::GreaterOrEqual,
            },
            TokenDef {
                re: Regex::new(r"^\+=").unwrap(),
                group: 0,
                converter: |_s| Token::PlusEqual,
            },
            TokenDef {
                re: Regex::new("^-=").unwrap(),
                group: 0,
                converter: |_s| Token::MinusEqual,
            },
            TokenDef {
                re: Regex::new(r"^\*=").unwrap(),
                group: 0,
                converter: |_s| Token::StarEqual,
            },
            TokenDef {
                re: Regex::new("^/=").unwrap(),
                group: 0,
                converter: |_s| Token::SlashEqual,
            },
            TokenDef {
                re: Regex::new("^%=").unwrap(),
                group: 0,
                converter: |_s| Token::PercentEqual,
            },
            TokenDef {
                re: Regex::new("^&=").unwrap(),
                group: 0,
                converter: |_s| Token::AmpersandEqual,
            },
            TokenDef {
                re: Regex::new(r"^\|=").unwrap(),
                group: 0,
                converter: |_s| Token::PipeEqual,
            },
            TokenDef {
                re: Regex::new(r"^\^=").unwrap(),
                group: 0,
                converter: |_s| Token::CaretEqual,
            },
            TokenDef {
                re: Regex::new("^-").unwrap(),
                group: 0,
                converter: |_s| Token::Hyphen,
            },
            TokenDef {
                re: Regex::new("^~").unwrap(),
                group: 0,
                converter: |_s| Token::Tilde,
            },
            TokenDef {
                re: Regex::new(r"^\+").unwrap(),
                group: 0,
                converter: |_s| Token::Plus,
            },
            TokenDef {
                re: Regex::new(r"^\*").unwrap(),
                group: 0,
                converter: |_s| Token::Star,
            },
            TokenDef {
                re: Regex::new("^/").unwrap(),
                group: 0,
                converter: |_s| Token::Slash,
            },
            TokenDef {
                re: Regex::new("^%").unwrap(),
                group: 0,
                converter: |_s| Token::Percent,
            },
            TokenDef {
                re: Regex::new("^&").unwrap(),
                group: 0,
                converter: |_s| Token::Ampersand,
            },
            TokenDef {
                re: Regex::new(r"^\|").unwrap(),
                group: 0,
                converter: |_s| Token::Pipe,
            },
            TokenDef {
                re: Regex::new(r"^\^").unwrap(),
                group: 0,
                converter: |_s| Token::Caret,
            },
            TokenDef {
                re: Regex::new("^!").unwrap(),
                group: 0,
                converter: |_s| Token::Bang,
            },
            TokenDef {
                re: Regex::new("^<").unwrap(),
                group: 0,
                converter: |_s| Token::LessThan,
            },
            TokenDef {
                re: Regex::new("^>").unwrap(),
                group: 0,
                converter: |_s| Token::GreaterThan,
            },
            TokenDef {
                re: Regex::new("^=").unwrap(),
                group: 0,
                converter: |_s| Token::EqualSign,
            },
            TokenDef {
                re: Regex::new(r"^\?").unwrap(),
                group: 0,
                converter: |_s| Token::QuestionMark,
            },
            TokenDef {
                re: Regex::new("^:").unwrap(),
                group: 0,
                converter: |_s| Token::Colon,
            },
            TokenDef {
                re: Regex::new("^,").unwrap(),
                group: 0,
                converter: |_s| Token::Comma,
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
            if let Some(m) = token_def.re.captures(input) {
                // It matched! Now extract the matching substring.
                return Some(MatchDef {
                    matched_substring: m.get(token_def.group).unwrap().as_str().to_string(),
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
            "if" => Token::KWIf,
            "else" => Token::KWElse,
            "do" => Token::KWDo,
            "while" => Token::KWWhile,
            "for" => Token::KWFor,
            "break" => Token::KWBreak,
            "continue" => Token::KWContinue,
            "static" => Token::KWStatic,
            "extern" => Token::KWExtern,
            "long" => Token::KWLong,
            "unsigned" => Token::KWUnsigned,
            "signed" => Token::KWSigned,
            "double" => Token::KWDouble,
            "goto" => Token::KWGoto,
            "switch" => Token::KWSwitch,
            "case" => Token::KWCase,
            "default" => Token::KWDefault,
            _ => Token::Identifier(s.to_string()),
        }
    }

    fn convert_int(s: &str) -> Token {
        Token::ConstInt(BigInt::from_str(s).unwrap())
    }

    fn convert_long(s: &str) -> Token {
        // drop "l" suffix
        let const_str = string_util::chop_suffix(s, 1);
        Token::ConstLong(BigInt::from_str(const_str).unwrap())
    }

    fn convert_uint(s: &str) -> Token {
        // drop "u" suffix
        let const_str = string_util::chop_suffix(s, 1);
        Token::ConstUInt(BigInt::from_str(const_str).unwrap())
    }

    fn convert_ulong(s: &str) -> Token {
        // remove ul/lu suffix
        let const_str = string_util::chop_suffix(s, 2);
        Token::ConstULong(BigInt::from_str(const_str).unwrap())
    }

    fn convert_double(s: &str) -> Token {
        Token::ConstDouble(f64::from_str(s).unwrap())
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
            vec![Token::ConstInt(BigInt::ZERO), Token::Semicolon]
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
                Token::ConstInt(BigInt::ZERO),
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
