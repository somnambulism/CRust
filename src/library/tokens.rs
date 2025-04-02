#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // Tokens with contents
    Identifier(String),
    Constant(i32),

    // Keywords
    KWInt,
    KWReturn,
    KWVoid,

    // Punctuation
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    Semicolon,
    Hyphen,
    DoubleHyphen,
    Tilde,
    Plus,
    Star,
    Slash,
    Percent,
    Ampersand,
    Pipe,
    Caret,
    LeftShift,
    RightShift,
}
