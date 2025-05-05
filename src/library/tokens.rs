#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // Tokens with contents
    Identifier(String),
    Constant(i32),

    // Keywords
    KWInt,
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
}
