use num_bigint::BigInt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Tokens with contents
    Identifier(String),
    ConstInt(BigInt),
    ConstLong(BigInt),
    ConstUInt(BigInt),
    ConstULong(BigInt),
    ConstDouble(f64),

    // Keywords
    KWInt,
    KWLong,
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
