use num_bigint::BigInt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    // Tokens with contents
    Identifier(String),
    ConstInt(BigInt),
    ConstLong(BigInt),
    ConstUInt(BigInt),
    ConstULong(BigInt),

    // Keywords
    KWInt,
    KWLong,
    KWSigned,
    KWUnsigned,
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
    EqualSign,       // =
    PlusEqual,       // +=
    MinusEqual,      // -=
    StarEqual,       // *=
    SlashEqual,      // /=
    PercentEqual,    // %=
    AmpersandEqual,  // &=
    PipeEqual,       // |=
    CaretEqual,      // ^=
    LeftShiftEqual,  // <<=
    RightShiftEqual, // >>=
    QuestionMark,
    Colon,
    Comma,
}
