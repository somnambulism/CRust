use num_traits::ToPrimitive;

use crate::library::{
    ast::{storage_class::StorageClass, untyped_exp::Initializer},
    r#const::T,
    types::Type,
};

use super::{
    ast::block_items::{Block as BlockStruct, Program as ProgramStruct},
    ast::block_items::{
        BlockItem, Declaration, ForInit, FunctionDeclaration, Statement, VariableDeclaration,
    },
    ast::ops::{BinaryOperator, UnaryOperator},
    ast::untyped_exp::Exp,
    tok_stream::TokenStream,
    tokens::Token,
};

type Block = BlockStruct<Initializer, Exp>;
type Program = ProgramStruct<Initializer, Exp>;

// first parse declarators of this type, then convert to AST
#[derive(Clone, Debug)]
enum Declarator {
    Ident(String),
    PointerDeclarator(Box<Declarator>),
    ArrayDeclarator(Box<Declarator>, Exp),
    FunDeclarator(Vec<ParamInfo>, Box<Declarator>),
}

#[derive(Clone, Debug)]
struct ParamInfo(Type, Declarator);

impl Declarator {
    pub fn process(&self, base_type: Type) -> (String, Type, Vec<String>) {
        match self {
            Declarator::Ident(s) => (s.to_string(), base_type, vec![]),
            Declarator::PointerDeclarator(d) => {
                let derived_type = Type::Pointer(Box::new(base_type));
                d.process(derived_type)
            }
            Declarator::ArrayDeclarator(inner, cnst) => {
                let size = const_to_dim(cnst);
                let derived_type = Type::Array {
                    elem_type: Box::new(base_type),
                    size,
                };
                inner.process(derived_type)
            }
            Declarator::FunDeclarator(params, declarator) => match &**declarator {
                Declarator::Ident(s) => {
                    let mut param_names = vec![];
                    let mut param_types = vec![];

                    for param in params {
                        let ParamInfo(p_base_type, p_decl) = param;

                        let (param_name, param_t, _) = p_decl.process(p_base_type.clone());

                        if let Type::FunType { .. } = param_t {
                            panic!("Function pointers in parameters are not supported");
                        }

                        param_names.push(param_name);
                        param_types.push(param_t);
                    }

                    let fun_type = Type::FunType {
                        param_types,
                        ret_type: Box::new(base_type),
                    };

                    (s.clone(), fun_type, param_names)
                }

                _ => panic!("Can't appy additional type deprevations to a function declarator"),
            },
        }
    }
}

#[derive(Debug)]
enum AbstractDeclarator {
    AbstractPointer(Box<AbstractDeclarator>),
    AbstractArray(Box<AbstractDeclarator>, Exp),
    AbstractBase,
}

impl AbstractDeclarator {
    pub fn process(&self, base_type: Type) -> Type {
        match self {
            AbstractDeclarator::AbstractBase => base_type,
            AbstractDeclarator::AbstractArray(inner, cnst) => {
                let dim = const_to_dim(cnst);
                let derived_type = Type::Array {
                    elem_type: Box::new(base_type),
                    size: dim,
                };
                inner.process(derived_type)
            }
            AbstractDeclarator::AbstractPointer(inner) => {
                let derived_type = Type::Pointer(Box::new(base_type));
                inner.process(derived_type)
            }
        }
    }
}

pub struct Parser {
    tokens: TokenStream,
}

fn get_precedence(token: &Token) -> Option<i8> {
    match token {
        Token::Star | Token::Slash | Token::Percent => Some(50),
        Token::Plus | Token::Hyphen | Token::DoublePlus | Token::DoubleHyphen => Some(45),
        Token::LeftShift | Token::RightShift => Some(40),
        Token::LessThan | Token::LessOrEqual | Token::GreaterThan | Token::GreaterOrEqual => {
            Some(35)
        }
        Token::DoubleEqual | Token::NotEqual => Some(30),
        Token::Ampersand => Some(25),
        Token::Caret => Some(20),
        Token::Pipe => Some(15),
        Token::LogicalAnd => Some(10),
        Token::LogicalOr => Some(5),
        Token::QuestionMark => Some(3),
        Token::EqualSign
        | Token::PlusEqual
        | Token::HyphenEqual
        | Token::StarEqual
        | Token::SlashEqual
        | Token::PercentEqual
        | Token::AmpersandEqual
        | Token::PipeEqual
        | Token::CaretEqual
        | Token::DoubleLeftBracketEqual
        | Token::DoubleRightBracketEqual => Some(1),
        _ => None,
    }
}

fn get_compound_operator(token: &Token) -> Option<BinaryOperator> {
    match token {
        Token::EqualSign => None,
        Token::PlusEqual => Some(BinaryOperator::Add),
        Token::HyphenEqual => Some(BinaryOperator::Subtract),
        Token::SlashEqual => Some(BinaryOperator::Divide),
        Token::PercentEqual => Some(BinaryOperator::Mod),
        Token::StarEqual => Some(BinaryOperator::Multiply),
        Token::AmpersandEqual => Some(BinaryOperator::BitwiseAnd),
        Token::PipeEqual => Some(BinaryOperator::BitwiseOr),
        Token::CaretEqual => Some(BinaryOperator::BitwiseXor),
        Token::DoubleLeftBracketEqual => Some(BinaryOperator::BitshiftLeft),
        Token::DoubleRightBracketEqual => Some(BinaryOperator::BitshiftRight),
        t => panic!(
            "Internal error: not a compound assignment operator: {:?}",
            t
        ),
    }
}

fn is_assignment(token: &Token) -> bool {
    matches!(
        token,
        Token::EqualSign
            | Token::PlusEqual
            | Token::HyphenEqual
            | Token::StarEqual
            | Token::SlashEqual
            | Token::PercentEqual
            | Token::AmpersandEqual
            | Token::CaretEqual
            | Token::PipeEqual
            | Token::DoubleLeftBracketEqual
            | Token::DoubleRightBracketEqual
    )
}

fn unescape(s: &str) -> Result<String, String> {
    let mut chars = s.chars().peekable();
    let mut out = String::new();

    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }

        // after backslash
        let esc = chars.next().ok_or("Dangling backslash")?;

        let decoded = match esc {
            '\'' => '\'',
            '"' => '"',
            '?' => '?',
            '\\' => '\\',
            'a' => '\x07',
            'b' => '\x08',
            'f' => '\x0C',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'v' => '\x0B',
            _ => return Err(format!("Invalid escape: \\{}", esc)),
        };

        out.push(decoded);
    }

    Ok(out)
}

fn is_type_specifier(token: &Token) -> bool {
    matches!(
        token,
        Token::KWInt
            | Token::KWLong
            | Token::KWSigned
            | Token::KWUnsigned
            | Token::KWDouble
            | Token::KWChar
    )
}

fn is_specifier(token: &Token) -> bool {
    match token {
        Token::KWStatic | Token::KWExtern => true,
        other => is_type_specifier(other),
    }
}

/* Convert constant to int and check that it's a valid array dimension: must be integer > 0 */
fn const_to_dim(c: &Exp) -> i64 {
    let i = match c {
        Exp::Constant(T::ConstInt(i)) => i64::from(*i),
        Exp::Constant(T::ConstLong(l)) => i64::from(*l),
        Exp::Constant(T::ConstUInt(u)) => i64::from(*u),
        Exp::Constant(T::ConstULong(ul)) => *ul as i64,
        Exp::Constant(T::ConstDouble(_)) => panic!("Array dimensions must have integer type"),
        Exp::Constant(T::ConstChar(_) | T::ConstUChar(_)) => {
            panic!("Internal error, we're not using char constants for array dimensions")
        }
        _ => panic!("Array dimensions must be constant expressions"),
    };
    if i > 0 {
        i
    } else {
        panic!("Array dimension must be greater than zero");
    }
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

    /* { "[" <const> "]" }+ */
    fn parse_array_dimensions(&mut self) -> Result<Vec<Exp>, String> {
        let mut dims = Vec::new();

        while let Token::OpenBracket = self.tokens.peek()? {
            self.tokens.take_token()?;

            let dim = self.parse_constant()?;
            self.expect(Token::CloseBracket)?;

            dims.push(dim);
        }

        Ok(dims)
    }

    /* getting a list of specifiers */

    fn parse_type_specifier_list(&mut self) -> Result<Vec<Token>, String> {
        let mut specs = vec![];

        let first = self.parse_type_specifier()?;
        specs.push(first);

        while let Ok(tok) = self.tokens.peek() {
            if is_type_specifier(tok) {
                let spec = self.parse_type_specifier()?;
                specs.push(spec);
            } else {
                break;
            }
        }

        Ok(specs)
    }

    // <type-specifier> ::= "int" | "long" | "unsigned" | "signed"
    fn parse_type_specifier(&mut self) -> Result<Token, String> {
        let spec = self.tokens.take_token()?;
        if is_type_specifier(&spec) {
            Ok(spec)
        } else {
            Err(format!("Expected a type specifier, found {:?}", spec))
        }
    }

    fn parse_specifier_list(&mut self) -> Result<Vec<Token>, String> {
        let spec = self.parse_specifier()?;
        let mut specifiers = vec![spec];
        while let Ok(token) = self.tokens.peek() {
            if is_specifier(token) {
                specifiers.push(self.parse_specifier()?);
            } else {
                break;
            }
        }
        Ok(specifiers)
    }

    // <specifier> ::= <type-specifier> | "static" | "extern"
    fn parse_specifier(&mut self) -> Result<Token, String> {
        let spec = self.tokens.take_token()?;
        if is_specifier(&spec) {
            Ok(spec)
        } else {
            Err(format!(
                "Expected a type or storage-class specifier, found {:?}",
                spec
            ))
        }
    }

    // Convert a single token to a storage class
    fn parse_storage_class(&mut self, token: &Token) -> Result<StorageClass, String> {
        match token {
            Token::KWExtern => Ok(StorageClass::Extern),
            Token::KWStatic => Ok(StorageClass::Static),
            other => Err(format!(
                "Expected a storage class specifier, found {:?}",
                other
            )),
        }
    }

    // Convert list of specifiers to a type
    fn parse_type(&mut self, mut specifier_list: Vec<Token>) -> Result<Type, String> {
        specifier_list.sort();

        match specifier_list.as_slice() {
            [Token::KWDouble] => return Ok(Type::Double),
            [Token::KWChar] => return Ok(Type::Char),
            [Token::KWChar, Token::KWSigned] => return Ok(Type::SChar),
            [Token::KWChar, Token::KWUnsigned] => return Ok(Type::UChar),
            _ => {
                for i in 0..specifier_list.len() - 1 {
                    if specifier_list[i] == specifier_list[i + 1]
                        && specifier_list[i] != Token::KWLong
                    {
                        return Err(format!("Duplicate type specifier: {:?}", specifier_list[i]));
                    }
                }

                let long_count = specifier_list
                    .iter()
                    .filter(|t| matches!(t, Token::KWLong))
                    .count();
                if long_count > 2 {
                    return Err(
                        "Invalid type specifier: 'long' appears more than twice".to_string()
                    );
                }

                if specifier_list.is_empty()
                    || specifier_list.contains(&Token::KWDouble)
                    || specifier_list.contains(&Token::KWChar)
                    || specifier_list.contains(&Token::KWSigned)
                        && specifier_list.contains(&Token::KWUnsigned)
                {
                    return Err("Invalid type specifier".to_string());
                }

                if specifier_list.contains(&Token::KWUnsigned) && long_count > 0 {
                    return Ok(Type::ULong);
                } else if specifier_list.contains(&Token::KWUnsigned) {
                    return Ok(Type::UInt);
                } else if long_count > 0 {
                    return Ok(Type::Long);
                } else {
                    return Ok(Type::Int);
                }
            }
        }
    }

    // Convert list of specifiers to type and storage class
    fn parse_type_and_storage_class(
        &mut self,
        specifier_list: Vec<Token>,
    ) -> Result<(Type, Option<StorageClass>), String> {
        let (types, storage_class): (Vec<_>, Vec<_>) = specifier_list
            .into_iter()
            .partition(|tok| is_type_specifier(tok));
        let typ = self.parse_type(types)?;
        let storage_class = match storage_class.as_slice() {
            [] => None,
            [sc] => Some(self.parse_storage_class(sc)?),
            _ => {
                return Err("Internal error - not a storage class".to_string());
            }
        };
        Ok((typ, storage_class))
    }

    /* parsing grammar symbols */

    fn parse_id(&mut self) -> Result<String, String> {
        match self.tokens.take_token() {
            Ok(Token::Identifier(x)) => Ok(x),
            other => Err(format!("Expected identifier, found {:?}", other.unwrap())),
        }
    }

    /* Parsing constants */

    /**
     * <int> ::= ? A constant token ?
     * <char> ::= ? A char token ?
     * <long> ::= ? An int or long token ?
     * <uint> ::= ? An unsigned int token ?
     * <ulong> ::= ? An unsigned int or unsigned long token ?
     * <double> ::= ? A floating-point constant token ?
     */
    fn parse_constant(&mut self) -> Result<Exp, String> {
        match self.tokens.take_token()? {
            Token::ConstChar(s) => {
                let unescaped = unescape(&s)?;
                if unescaped.len() == 1 {
                    Ok(Exp::Constant(T::ConstInt(
                        unescaped.bytes().nth(0).unwrap() as i32,
                    )))
                } else {
                    Err(format!(
                        "Internal error: Character token {} contains multiple characters",
                        s
                    ))
                }
            }

            Token::ConstDouble(d) => Ok(Exp::Constant(T::ConstDouble(d))),

            Token::ConstInt(c) => {
                if let Some(i) = c.to_i32() {
                    Ok(Exp::Constant(T::ConstInt(i)))
                } else if let Some(l) = c.to_i64() {
                    Ok(Exp::Constant(T::ConstLong(l)))
                } else {
                    Err(format!("Constant {} too large to fit into i64", c))
                }
            }

            Token::ConstLong(c) => c
                .to_i64()
                .map(T::ConstLong)
                .map(Exp::Constant)
                .ok_or(format!("Constant {} too large to fit into i64", c)),

            Token::ConstUInt(c) => {
                if let Some(u) = c.to_u32() {
                    Ok(Exp::Constant(T::ConstUInt(u)))
                } else if let Some(ul) = c.to_u64() {
                    Ok(Exp::Constant(T::ConstULong(ul)))
                } else {
                    Err(format!("Constant {} too large to fit into u64", c))
                }
            }

            Token::ConstULong(c) => c
                .to_u64()
                .map(T::ConstULong)
                .map(Exp::Constant)
                .ok_or(format!("Constant {} too large to fit into i64", c)),

            // we only call this when we know the next token is a constant
            _ => Err("Internal error when parsing constant".to_string()),
        }
    }

    /* Parsing declarators */

    // <simple-declarator> ::= <identifier> | "(" <declarator> ")"
    fn parse_simple_declarator(&mut self) -> Result<Declarator, String> {
        let next_tok = self.tokens.take_token()?;
        match next_tok {
            Token::OpenParen => {
                let decl = self.parse_declarator()?;
                self.expect(Token::CloseParen)?;
                Ok(decl)
            }
            Token::Identifier(id) => Ok(Declarator::Ident(id.to_string())),
            other => Err(format!("Expected a simple declarator, found {:?}", other)),
        }
    }

    // <declarator> ::= "*" <declarator> | <direct-declarator>
    fn parse_declarator(&mut self) -> Result<Declarator, String> {
        if let Token::Star = self.tokens.peek()? {
            self.tokens.take_token()?;
            let inner = self.parse_declarator()?;
            Ok(Declarator::PointerDeclarator(Box::new(inner)))
        } else {
            self.parse_direct_declarator()
        }
    }

    /* <direct-declarator> ::= <simple-declarator> [ <declarator-suffix> ]
     * <declarator-suffix> ::= <param-list> | { "[" <const> "]" }
     */
    fn parse_direct_declarator(&mut self) -> Result<Declarator, String> {
        let simple_dec = self.parse_simple_declarator()?;
        match self.tokens.peek()? {
            Token::OpenBracket => {
                let array_dimensions = self.parse_array_dimensions()?;

                let mut decl = simple_dec;

                for dim in array_dimensions {
                    decl = Declarator::ArrayDeclarator(Box::new(decl), dim)
                }

                Ok(decl)
            }
            Token::OpenParen => {
                let params = self.parse_param_list()?;
                Ok(Declarator::FunDeclarator(params, Box::new(simple_dec)))
            }
            _ => Ok(simple_dec),
        }
    }

    // <param-list> ::= "(" <param> { "," <param> } ")" | "(" "void" ")"
    fn parse_param_list(&mut self) -> Result<Vec<ParamInfo>, String> {
        self.expect(Token::OpenParen)?;

        let params = if let Token::KWVoid = self.tokens.peek()? {
            self.tokens.take_token()?;
            vec![]
        } else {
            self.param_loop()?
        };

        self.expect(Token::CloseParen)?;
        Ok(params)
    }

    fn param_loop(&mut self) -> Result<Vec<ParamInfo>, String> {
        let p = self.parse_param()?;

        if let Token::Comma = self.tokens.peek()? {
            // parse the rest of the param list
            self.tokens.take_token()?;
            let mut rest = self.param_loop()?;
            let mut result = Vec::with_capacity(1 + rest.len());
            result.push(p);
            result.append(&mut rest);
            Ok(result)
        } else {
            Ok(vec![p])
        }
    }

    fn parse_param(&mut self) -> Result<ParamInfo, String> {
        let specifiers = self.parse_type_specifier_list()?;
        let param_type = self.parse_type(specifiers)?;
        let param_decl = self.parse_declarator()?;

        Ok(ParamInfo(param_type, param_decl))
    }

    /* Abstract declarators */

    /*
     * <abstract-declarator> ::= "*" [ <abstract-declarator> ]
     *                           | <direct-abstract-declarator>
     */
    fn parse_abstract_declarator(&mut self) -> Result<AbstractDeclarator, String> {
        if let Token::Star = self.tokens.peek()? {
            // it's a pointer declarator
            self.tokens.take_token()?;
            let inner = match self.tokens.peek()? {
                Token::Star | Token::OpenParen | Token::OpenBracket => {
                    // it's an inner declarator
                    self.parse_abstract_declarator()?
                }
                Token::CloseParen => AbstractDeclarator::AbstractBase,
                other => panic!("Expected an abstract declarator, found {:?}", other),
            };
            Ok(AbstractDeclarator::AbstractPointer(Box::new(inner)))
        } else {
            self.parse_direct_abstract_declarator()
        }
    }

    /* <direct-abstract-declarator> ::= "(" <abstract-declarator> ")" { "[" <const> "]" }
     *                                | { "[" <const> "]" }+
     */
    fn parse_direct_abstract_declarator(&mut self) -> Result<AbstractDeclarator, String> {
        match self.tokens.peek()? {
            Token::OpenParen => {
                self.tokens.take_token()?;
                let abstr_decl = self.parse_abstract_declarator()?;
                self.expect(Token::CloseParen)?;
                // inner declarator is followed by possibly-empty list of array dimensions
                let array_dimensions = self.parse_array_dimensions()?;

                let mut decl = abstr_decl;

                for dim in array_dimensions {
                    decl = AbstractDeclarator::AbstractArray(Box::new(decl), dim);
                }

                Ok(decl)
            }
            Token::OpenBracket => {
                let array_dimensions = self.parse_array_dimensions()?;

                let mut decl = AbstractDeclarator::AbstractBase;

                for dim in array_dimensions {
                    decl = AbstractDeclarator::AbstractArray(Box::new(decl), dim);
                }

                Ok(decl)
            }
            other => panic!("Expected an abstract direct declarator, found {:?}", other),
        }

        // self.expect(Token::OpenParen)?;
        // let decl = self.parse_abstract_declarator()?;
        // self.expect(Token::CloseParen)?;
        // Ok(decl)
    }

    // <unop> ::= "-" | "~" | "!" | "++" | "--"
    fn parse_unop(&mut self) -> Result<UnaryOperator, String> {
        match self.tokens.take_token() {
            Ok(Token::Tilde) => Ok(UnaryOperator::Complement),
            Ok(Token::Hyphen) => Ok(UnaryOperator::Negate),
            Ok(Token::Bang) => Ok(UnaryOperator::Not),
            Ok(Token::DoublePlus) => Ok(UnaryOperator::Incr),
            Ok(Token::DoubleHyphen) => Ok(UnaryOperator::Decr),
            // we only call this when we know the next token is unop
            other => Err(format!(
                "Expected a unary operator, found {:?}",
                other.unwrap()
            )),
        }
    }

    // <binop> ::= "-" | "+" | "*" | "/" | "%" | "&" | "|"
    //          | "<<" | ">>" | "&&" | "||"
    //          | "==" | "!=" | "<" | "<=" | ">" | ">="
    fn parse_binop(&mut self) -> Result<BinaryOperator, String> {
        match self.tokens.take_token()? {
            Token::Plus => Ok(BinaryOperator::Add),
            Token::Hyphen => Ok(BinaryOperator::Subtract),
            Token::Star => Ok(BinaryOperator::Multiply),
            Token::Slash => Ok(BinaryOperator::Divide),
            Token::Percent => Ok(BinaryOperator::Mod),
            Token::Ampersand => Ok(BinaryOperator::BitwiseAnd),
            Token::Pipe => Ok(BinaryOperator::BitwiseOr),
            Token::Caret => Ok(BinaryOperator::BitwiseXor),
            Token::LeftShift => Ok(BinaryOperator::BitshiftLeft),
            Token::RightShift => Ok(BinaryOperator::BitshiftRight),
            Token::LogicalAnd => Ok(BinaryOperator::And),
            Token::LogicalOr => Ok(BinaryOperator::Or),
            Token::DoubleEqual => Ok(BinaryOperator::Equal),
            Token::NotEqual => Ok(BinaryOperator::NotEqual),
            Token::LessThan => Ok(BinaryOperator::LessThan),
            Token::LessOrEqual => Ok(BinaryOperator::LessOrEqual),
            Token::GreaterThan => Ok(BinaryOperator::GreaterThan),
            Token::GreaterOrEqual => Ok(BinaryOperator::GreaterOrEqual),
            other => Err(format!("Expected a binary operator, found {:?}", other)),
        }
    }

    fn parse_string_literals(&mut self) -> Result<String, String> {
        let mut result = String::new();

        while let Token::StringLiteral(s) = self.tokens.peek()? {
            let unescaped = unescape(s)?;
            result.push_str(&unescaped);
            self.tokens.take_token()?;
        }

        Ok(result)
    }

    // <primary-exp> ::= <const> | <identifier> | <identifier> "(" [ <argument-list ] ")" | "(" <exp> ")"
    fn parse_primary_expression(&mut self) -> Result<Exp, String> {
        let next_token = self.tokens.peek()?;
        match next_token {
            // constant
            Token::ConstChar(_)
            | Token::ConstInt(_)
            | Token::ConstLong(_)
            | Token::ConstUInt(_)
            | Token::ConstULong(_)
            | Token::ConstDouble(_) => self.parse_constant(),
            // variable or function call
            Token::Identifier(_) => {
                let id = self.parse_id()?;
                // look at next token to figure out whether this is a variable or function call
                let exp = if self.tokens.peek()? == &Token::OpenParen {
                    let args = self.parse_optional_arg_list()?;
                    Exp::FunCall { f: id, args }
                } else {
                    // It's a variable
                    Exp::Var(id)
                };

                Ok(exp)
            }
            Token::StringLiteral(_) => {
                let string_exp = self.parse_string_literals()?;
                Ok(Exp::String(string_exp))
            }
            // cast or parenthesized expression
            Token::OpenParen => {
                // Consume open paren
                self.tokens.take_token()?;
                let e = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                Ok(e)
            }
            t => Err(format!("Expected a factor, found {:?}", t)),
        }
    }

    fn parse_postfix_exp(&mut self) -> Result<Exp, String> {
        let primary = self.parse_primary_expression()?;
        self.postfix_helper(primary)
    }

    fn postfix_helper(&mut self, primary: Exp) -> Result<Exp, String> {
        match self.tokens.peek()? {
            Token::DoubleHyphen => {
                self.tokens.take_token()?;
                let decr_exp = Exp::PostfixDecr(primary.into());
                self.postfix_helper(decr_exp)
            }
            Token::DoublePlus => {
                self.tokens.take_token()?;
                let incr_exp = Exp::PostfixIncr(primary.into());
                self.postfix_helper(incr_exp)
            }
            Token::OpenBracket => {
                self.tokens.take_token()?;
                let index = self.parse_expression(0)?;
                self.expect(Token::CloseBracket)?;
                let subscript_exp = Exp::Subscript {
                    ptr: Box::new(primary),
                    index: Box::new(index),
                };
                self.postfix_helper(subscript_exp)
            }
            _ => Ok(primary),
        }
    }

    // <factor> ::= <unop> <factor> | "(" { <type-specifier> }+ ")" | factor | <postfix-exp>
    fn parse_factor(&mut self) -> Result<Exp, String> {
        let next_tokens = self.tokens.npeek(2);
        match next_tokens {
            // unary expression
            [
                Token::Hyphen
                | Token::Tilde
                | Token::Bang
                | Token::DoublePlus
                | Token::DoubleHyphen,
                _,
            ] => {
                let operator = self.parse_unop()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Unary(operator, Box::new(inner_exp)))
            }
            [Token::Star, _] => {
                self.tokens.take_token()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Dereference(Box::new(inner_exp)))
            }
            [Token::Ampersand, _] => {
                self.tokens.take_token()?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::AddrOf(Box::new(inner_exp)))
            }
            [Token::OpenParen, t] if is_type_specifier(t) => {
                // It's a cast - consume open paren, then parse type specifiers
                self.tokens.take_token()?;
                let type_specifiers = self.parse_type_specifier_list()?;
                let base_type = self.parse_type(type_specifiers)?;
                // check for optional abstract declarator
                let target_type = if let Token::CloseParen = self.tokens.peek()? {
                    base_type
                } else {
                    let abstract_decl = self.parse_abstract_declarator()?;
                    abstract_decl.process(base_type)
                };
                self.expect(Token::CloseParen)?;
                let inner_exp = self.parse_factor()?;
                Ok(Exp::Cast {
                    target_type,
                    e: Box::new(inner_exp),
                })
            }
            _ => self.parse_postfix_exp(),
        }
    }

    // "(" [ <argument-list> ] ")"
    fn parse_optional_arg_list(&mut self) -> Result<Vec<Exp>, String> {
        self.expect(Token::OpenParen)?;
        let args = if self.tokens.peek()? == &Token::CloseParen {
            vec![]
        } else {
            self.parse_arg_list()?
        };
        self.expect(Token::CloseParen)?;
        Ok(args)
    }

    // <argument-list> ::= <exp> { "," <exp> }
    fn parse_arg_list(&mut self) -> Result<Vec<Exp>, String> {
        let mut args = vec![];

        args.push(self.parse_expression(0)?);

        while self.tokens.peek()? == &Token::Comma {
            self.tokens.take_token()?;
            args.push(self.parse_expression(0)?);
        }

        Ok(args)
    }

    // "?" <exp> ":"
    fn parse_conditional_middle(&mut self) -> Result<Exp, String> {
        self.expect(Token::QuestionMark)?;
        let e = self.parse_expression(0)?;
        self.expect(Token::Colon)?;
        Ok(e)
    }

    // <exp> ::= <factor> | <exp> <binop> <exp> | <exp> "?" <exp> ":" <exp>
    fn parse_expression(&mut self, min_prec: i8) -> Result<Exp, String> {
        let mut left = self.parse_factor()?;
        let mut next_token = self.tokens.peek()?.clone();

        while let Some(prec) = get_precedence(&next_token) {
            if prec < min_prec {
                break;
            }

            if is_assignment(&next_token) {
                self.tokens.take_token()?;
                let right = self.parse_expression(prec)?;
                left = match get_compound_operator(&next_token) {
                    None => Exp::Assignment(left.into(), right.into()),
                    Some(op) => Exp::CompoundAssign(op, left.into(), right.into()),
                };
            } else if next_token == Token::QuestionMark {
                let middle = self.parse_conditional_middle()?;
                let right = self.parse_expression(prec)?;
                left = Exp::Conditional {
                    condition: left.into(),
                    then_result: middle.into(),
                    else_result: right.into(),
                };
            } else {
                let operator = self.parse_binop()?;
                let right = self.parse_expression(prec + 1)?;
                left = Exp::Binary(operator, Box::new(left), Box::new(right));
            }

            next_token = self.tokens.peek()?.clone();
        }

        Ok(left)
    }

    fn parse_optional_expression(&mut self, delim: Token) -> Result<Option<Exp>, String> {
        if self.tokens.peek()? == &delim {
            self.tokens.take_token()?;
            return Ok(None);
        } else {
            let e = self.parse_expression(0)?;
            self.expect(delim)?;
            Ok(Some(e))
        }
    }

    // <initializer> ::= <exp> | "{" <initializer> { "," <initializer> } [ "," ] "}"
    fn parse_initializer(&mut self) -> Result<Initializer, String> {
        if self.tokens.peek()? == &Token::OpenBrace {
            self.tokens.take_token()?;
            let init_list = self.parse_init_list()?;
            self.expect(Token::CloseBrace)?;
            Ok(Initializer::CompoundInit(init_list))
        } else {
            let e = self.parse_expression(0)?;
            Ok(Initializer::SingleInit(e))
        }
    }

    fn parse_init_list(&mut self) -> Result<Vec<Box<Initializer>>, String> {
        let next_init = self.parse_initializer()?;

        match self.tokens.npeek(2) {
            // trailing comma - consume it and return
            [Token::Comma, Token::CloseBrace] => {
                self.tokens.take_token()?;
                Ok(vec![Box::new(next_init)])
            }
            // comma that isn't followed by a brace means there's one more element
            [Token::Comma, _] => {
                self.tokens.take_token()?;
                let mut rest = self.parse_init_list()?;
                let mut res = Vec::with_capacity(1 + rest.len());
                res.push(Box::new(next_init));
                res.append(&mut rest);
                Ok(res)
            }
            _ => Ok(vec![Box::new(next_init)]),
        }
    }

    // <statement> ::= "return" <exp> ";"
    //              | "if" "(" <exp> ")" <statement> [ "else" <statement> ]
    //              | <label> ":" <statement> ";"
    //              | "goto" <label> ";"
    //              | <block>
    //              | <break> ";"
    //              | <continue> ";"
    //              | "while" "(" <exp> ")" <statement>
    //              | "do" <statement> "while" "(" <exp> ")" ";"
    //              | "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
    //              | "switch" "(" <exp> ")" <statement>
    //              | "case" <int> ":" <statement>
    //              | "default" ":" <statement>
    //              | <exp> ";"
    //              | ";"
    fn parse_statement(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        match self.tokens.npeek(2) {
            [Token::KWIf, _] => self.parse_if_statement(),
            [Token::OpenBrace, _] => Ok(Statement::Compound(self.parse_block()?)),
            [Token::KWDo, _] => self.parse_do_loop(),
            [Token::KWWhile, _] => self.parse_while_loop(),
            [Token::KWFor, _] => self.parse_for_loop(),
            [Token::KWBreak, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Break("".to_string()))
            }
            [Token::KWContinue, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Continue("".to_string()))
            }
            [Token::KWReturn, _] => {
                // consume return keyword
                self.tokens.take_token()?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Return(exp))
            }
            [Token::KWGoto, _] => {
                self.tokens.take_token()?;
                let lbl = self.parse_id()?;
                self.expect(Token::Semicolon)?;
                Ok(Statement::Goto(lbl))
            }
            [Token::KWSwitch, _] => self.parse_switch_statement(),
            [Token::KWCase, _] => {
                self.tokens.take_token()?;
                let case_val = self.parse_expression(0)?;
                self.expect(Token::Colon)?;
                let stmt = Box::new(self.parse_statement()?);
                Ok(Statement::Case(case_val, stmt, "".to_string()))
            }
            [Token::KWDefault, _] => {
                self.tokens.take_token()?;
                self.expect(Token::Colon)?;
                let stmt = Box::new(self.parse_statement()?);
                Ok(Statement::Default(stmt, "".to_string()))
            }
            [Token::Identifier(_), Token::Colon] => {
                // consume label
                let tok = self.tokens.take_token()?;
                let lbl = match tok {
                    Token::Identifier(name) => name,
                    other => {
                        return Err(format!("Expected identifier, found {:?}", other));
                    }
                };
                self.tokens.take_token()?; // consume colon
                let stmt = self.parse_statement()?;
                Ok(Statement::LabelledStatement(lbl, Box::new(stmt)))
            }
            _ => {
                let exp = self.parse_optional_expression(Token::Semicolon)?;
                if let Some(e) = exp {
                    Ok(Statement::Expression(e))
                } else {
                    Ok(Statement::Null)
                }
            }
        }
    }

    // "if" "(" <exp> ")" <statement> [ "else" <statement> ]
    fn parse_if_statement(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        self.expect(Token::KWIf)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let then_clause = self.parse_statement()?;

        let else_clause = if let Ok(Token::KWElse) = self.tokens.peek() {
            // there is an else clause - consume the else keyword
            self.tokens.take_token()?;
            Some(Box::new(self.parse_statement()?))
        } else {
            // there's no else clause
            None
        };

        Ok(Statement::If {
            condition,
            then_clause: then_clause.into(),
            else_clause,
        })
    }

    // "do" <statement> "while" "(" <exp> ")" ";"
    fn parse_do_loop(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        self.expect(Token::KWDo)?;
        let body = Box::new(self.parse_statement()?);
        self.expect(Token::KWWhile)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        self.expect(Token::Semicolon)?;
        Ok(Statement::DoWhile {
            body,
            condition,
            id: "".to_string(),
        })
    }

    // "while" "(" <exp> ")" <statement>
    fn parse_while_loop(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        self.expect(Token::KWWhile)?;
        self.expect(Token::OpenParen)?;
        let condition = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::While {
            condition,
            body,
            id: "".to_string(),
        })
    }

    // "for" "(" <for-init> [ <exp> ] ";" [ <exp> ] ")" <statement>
    fn parse_for_loop(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        self.expect(Token::KWFor)?;
        self.expect(Token::OpenParen)?;
        let init = self.parse_for_init()?;
        let condition = self.parse_optional_expression(Token::Semicolon)?;
        let post = self.parse_optional_expression(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::For {
            init,
            condition,
            post,
            body,
            id: "".to_string(),
        })
    }

    // "switch" "(" <exp> ")" <statement>
    fn parse_switch_statement(&mut self) -> Result<Statement<Initializer, Exp>, String> {
        self.expect(Token::KWSwitch)?;
        self.expect(Token::OpenParen)?;
        let control = self.parse_expression(0)?;
        self.expect(Token::CloseParen)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::Switch {
            control,
            body,
            cases: Vec::new(),
            id: "".to_string(),
        })
    }

    // <block_item> ::= <statement> | <declaration>
    fn parse_block_item(&mut self) -> Result<BlockItem<Initializer, Exp>, String> {
        if is_specifier(self.tokens.peek()?) {
            Ok(BlockItem::D(self.parse_declaration()?))
        } else {
            Ok(BlockItem::S(self.parse_statement()?))
        }
    }

    fn parse_block(&mut self) -> Result<Block, String> {
        self.expect(Token::OpenBrace)?;
        let mut block = Vec::new();
        while self.tokens.peek()? != &Token::CloseBrace {
            let block_item = self.parse_block_item()?;
            block.push(block_item);
        }
        self.expect(Token::CloseBrace)?;
        Ok(BlockStruct(block))
    }

    // <function-declaration> ::= { <specifier> }+ <declarator> ( <block> | ";" )
    // we've already parsed { <specifier> }+ <declarator>
    fn finish_parsing_function_declaration(
        &mut self,
        fun_type: Type,
        storage_class: Option<StorageClass>,
        name: String,
        params: Vec<String>,
    ) -> Result<FunctionDeclaration<Initializer, Exp>, String> {
        let body = match self.tokens.peek()? {
            Token::OpenBrace => Some(self.parse_block()?),
            Token::Semicolon => {
                self.tokens.take_token()?;
                None
            }
            other => {
                return Err(format!(
                    "Expected function body or semicolor, found {:?}",
                    other
                ));
            }
        };

        Ok(FunctionDeclaration {
            name,
            fun_type,
            storage_class,
            params,
            body,
        })
    }

    // <variable-declaration> ::= { <specifier> }+ <declarator> [ "=" <exp> ] ";"
    // we've already parsed { <specifier> }+ <declarator>
    fn finish_parsing_variable_declaration(
        &mut self,
        var_type: Type,
        storage_class: Option<StorageClass>,
        name: String,
    ) -> Result<VariableDeclaration<Initializer>, String> {
        match self.tokens.take_token()? {
            Token::Semicolon => Ok(VariableDeclaration {
                name,
                var_type,
                storage_class,
                init: None,
            }),
            Token::EqualSign => {
                let init = self.parse_initializer()?;
                self.expect(Token::Semicolon)?;
                Ok(VariableDeclaration {
                    name,
                    var_type,
                    storage_class,
                    init: Some(init),
                })
            }
            other => Err(format!(
                "Expected an initializer or semicolon, found {:?}",
                other
            )),
        }
    }

    // <declaration> ::= <variable-declaration> | <function-declaration>
    // parse until declarator, then call appropriate function to finish parsing
    fn parse_declaration(&mut self) -> Result<Declaration<Initializer, Exp>, String> {
        let specifiers = self.parse_specifier_list()?;
        let (base_typ, storage_class) = self.parse_type_and_storage_class(specifiers)?;
        let declarator = self.parse_declarator()?;
        let (name, typ, params) = declarator.process(base_typ);

        match typ {
            Type::FunType { .. } => Ok(Declaration::FunDecl(
                self.finish_parsing_function_declaration(typ, storage_class, name, params)?,
            )),
            _ => {
                if params.len() == 0 {
                    Ok(Declaration::VarDecl(
                        self.finish_parsing_variable_declaration(typ, storage_class, name)?,
                    ))
                } else {
                    Err("Internal error: declarator has parameters but object type".to_string())
                }
            }
        }
    }

    fn parse_variable_declaration(&mut self) -> Result<VariableDeclaration<Initializer>, String> {
        match self.parse_declaration()? {
            Declaration::VarDecl(vd) => Ok(vd),
            Declaration::FunDecl(_) => {
                Err("Expected variable declaration but found function declaration".to_string())
            }
        }
    }

    // <for-init> ::= <variable-declaration> | [ <exp> ] ";"
    fn parse_for_init(&mut self) -> Result<ForInit<Initializer, Exp>, String> {
        if is_specifier(self.tokens.peek()?) {
            Ok(ForInit::InitDecl(self.parse_variable_declaration()?))
        } else {
            let opt_e = self.parse_optional_expression(Token::Semicolon)?;
            Ok(ForInit::InitExp(opt_e))
        }
    }

    // <program> ::= { <declaration> }*
    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut fun_decls = vec![];

        while !self.tokens.is_empty() {
            let next_decl = self.parse_declaration()?;
            fun_decls.push(next_decl);
        }

        Ok(ProgramStruct(fun_decls))
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::FromPrimitive;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn signed_long_constant() {
        let tok_list = vec![Token::ConstLong(
            BigInt::from_i64(4611686018427387904).unwrap(),
        )];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_constant().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstLong(4611686018427387904)));
    }

    #[test]
    fn unsigned_int_constant() {
        let tok_list = vec![Token::ConstUInt(BigInt::from_str("4294967291").unwrap())];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_constant().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstUInt(4294967291)));
    }

    #[test]
    fn unsigned_long_constant() {
        let tok_list = vec![Token::ConstULong(
            BigInt::from_str("18446744073709551611").unwrap(),
        )];
        let mut parser = Parser::new(tok_list);
        let c = parser.parse_constant().unwrap();
        assert_eq!(c, Exp::Constant(T::ConstULong(18446744073709551611)));
    }

    #[test]
    fn expression() {
        let mut parser = Parser::new(vec![Token::ConstInt(BigInt::from(100)), Token::Semicolon]);
        assert_eq!(
            parser.parse_expression(40).unwrap(),
            Exp::Constant(T::ConstInt(100))
        );
    }

    #[test]
    fn statement() {
        let mut parser = Parser::new(vec![
            Token::KWReturn,
            Token::ConstInt(BigInt::from(4)),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::Return(Exp::Constant(T::ConstInt(4)))
        );
    }

    #[test]
    #[should_panic]
    fn error() {
        let mut parser = Parser::new(vec![Token::KWInt]);
        let _ = parser.parse_program();
    }

    #[test]
    fn labelled_statement() {
        let mut parser = Parser::new(vec![
            Token::Identifier("label".to_string()),
            Token::Colon,
            Token::KWReturn,
            Token::ConstInt(BigInt::from(42)),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::LabelledStatement(
                "label".to_string(),
                Box::new(Statement::Return(Exp::Constant(T::ConstInt(42)))),
            )
        );
    }

    #[test]
    fn goto_statement() {
        let mut parser = Parser::new(vec![
            Token::KWGoto,
            Token::Identifier("label".to_string()),
            Token::Semicolon,
        ]);
        assert_eq!(
            parser.parse_statement().unwrap(),
            Statement::Goto("label".to_string())
        );
    }
}
