use std::cmp::Ordering;

use num_bigint::BigInt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Tokens with contents
    Identifier(String),
    StringLiteral(String),
    ConstChar(String),
    ConstInt(BigInt),
    ConstLong(BigInt),
    ConstUInt(BigInt),
    ConstULong(BigInt),
    ConstDouble(f64),

    // Keywords
    KWInt,
    KWLong,
    KWChar,
    KWSigned,
    KWUnsigned,
    KWDouble,
    KWReturn,
    KWVoid,
    KWIf,
    KWElse,
    KWDo,
    KWWhile,
    KWFor,
    KWBreak,
    KWContinue,
    KWGoto,
    KWSwitch,
    KWCase,
    KWDefault,
    KWStatic,
    KWExtern,
    KWSizeOf,

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
    DoublePlus,
    Star,
    Slash,
    Percent,
    Ampersand,   // &
    Pipe,        // |
    Caret,       // ^
    LeftShift,   // <<
    RightShift,  // >>
    Bang,        // !
    LogicalAnd,  // &&
    LogicalOr,   // ||
    DoubleEqual, // ==
    NotEqual,    // !=
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
    EqualSign,               // =
    PlusEqual,               // +=
    HyphenEqual,             // -=
    StarEqual,               // *=
    SlashEqual,              // /=
    PercentEqual,            // %=
    AmpersandEqual,          // &=
    PipeEqual,               // |=
    CaretEqual,              // ^=
    DoubleLeftBracketEqual,  // <<=
    DoubleRightBracketEqual, // >>=
    QuestionMark,
    Colon,
    Comma,
    OpenBracket,
    CloseBracket,
}

impl Eq for Token {}

impl PartialOrd for Token {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Token {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        fn rank(t: &Token) -> u8 {
            match t {
                Token::KWInt => 8,
                Token::KWLong => 9,
                Token::KWChar => 10,
                Token::KWSigned => 11,
                Token::KWUnsigned => 12,
                Token::KWDouble => 13,
                Token::KWStatic => 14,
                Token::KWExtern => 15,
                _ => 0,
            }
        }

        let r1 = rank(self);
        let r2 = rank(other);

        match r1.cmp(&r2) {
            Ordering::Equal => Ordering::Equal,
            other => other,
        }
    }
}
