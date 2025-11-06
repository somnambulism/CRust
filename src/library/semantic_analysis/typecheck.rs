use core::panic;
use std::{collections::HashSet, iter::zip, ops::Rem};

use crate::library::{
    ast::{
        block_items::{
            Block, BlockItem, Declaration, ForInit, FunctionDeclaration, Program, Statement,
            VariableDeclaration,
        },
        ops::{BinaryOperator, UnaryOperator},
        storage_class::StorageClass,
        typed_exp::{InnerExp, TypedExp},
        untyped_exp::Exp,
    },
    r#const::{T, type_of_const},
    const_convert::const_convert,
    initializers::{StaticInit, zero},
    symbols::{Entry, IdentifierAttrs, InitialValue, SymbolTable},
    types::Type,
};

fn convert_to(e: TypedExp, target_type: Type) -> TypedExp {
    let cast = InnerExp::Cast {
        target_type: target_type.clone(),
        e: Box::new(e),
    };
    TypedExp::set_type(cast, target_type)
}

fn get_common_type(t1: &Type, t2: &Type) -> Type {
    if t1 == t2 {
        t1.clone()
    } else if t1 == &Type::Double || t2 == &Type::Double {
        Type::Double
    } else if t1.get_size() == t2.get_size() {
        if t1.is_signed() {
            t2.clone()
        } else {
            t1.clone()
        }
    } else if t1.get_size() > t2.get_size() {
        t1.clone()
    } else {
        t2.clone()
    }
}

pub struct TypeChecker {
    pub symbol_table: SymbolTable,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            symbol_table: SymbolTable::new(),
        }
    }

    fn typecheck_var(&self, v: &str) -> TypedExp {
        let v_type = self.symbol_table.get(v).t.clone();
        let e = InnerExp::Var(v.to_string());
        match v_type {
            Type::FunType { .. } => {
                panic!("Tried to use function name {} as variable", v);
            }
            _ => TypedExp::set_type(e, v_type),
        }
    }

    fn typecheck_const(&self, c: &T) -> TypedExp {
        let e = InnerExp::Constant(c.clone());
        TypedExp::set_type(e, type_of_const(c))
    }

    fn typecheck_exp(&self, exp: &Exp) -> TypedExp {
        match exp {
            Exp::Var(v) => self.typecheck_var(v),
            Exp::Constant(c) => self.typecheck_const(c),
            Exp::Cast {
                target_type,
                e: inner,
            } => {
                let cast_exp = InnerExp::Cast {
                    target_type: target_type.clone(),
                    e: self.typecheck_exp(inner).into(),
                };
                TypedExp::set_type(cast_exp, target_type.clone())
            }
            Exp::Unary(op, inner) => self.typecheck_unary(&op, &inner),
            Exp::Binary(op, e1, e2) => self.typecheck_binary(op, e1, e2),
            Exp::PostfixDecr(inner) => {
                self.typecheck_increment_decrement(inner, InnerExp::PostfixDecrement)
            }
            Exp::PostfixIncr(inner) => {
                self.typecheck_increment_decrement(inner, InnerExp::PostfixIncrement)
            }
            Exp::Assignment(lhs, rhs) => self.typecheck_assignment(lhs, rhs),
            Exp::CompoundAssign(op, lhs, rhs) => self.typecheck_compound_assignment(op, lhs, rhs),
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => self.typecheck_conditional(condition, then_result, else_result),
            Exp::FunCall { f, args } => self.typecheck_fun_call(f, args),
        }
    }

    fn typecheck_unary(&self, op: &UnaryOperator, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);
        let unary_exp = InnerExp::Unary(op.clone(), Box::new(typed_inner.clone()));
        match op {
            UnaryOperator::Not => TypedExp::set_type(unary_exp, Type::Int),
            UnaryOperator::Complement if typed_inner.get_type() == Type::Double => {
                panic!("Can't apply bitwise complement to double");
            }
            _ => TypedExp::set_type(unary_exp, typed_inner.get_type()),
        }
    }

    fn typecheck_binary(&self, op: &BinaryOperator, e1: &Exp, e2: &Exp) -> TypedExp {
        let typed_e1 = self.typecheck_exp(e1);
        let typed_e2 = self.typecheck_exp(e2);
        match op {
            BinaryOperator::BitshiftLeft | BinaryOperator::BitshiftRight => {
                if typed_e1.get_type() == Type::Double || typed_e2.get_type() == Type::Double {
                    panic!("Both operands of bitshift must be integer type");
                }
                // Don't perform usual arithmetic conversions; result has type of left operand
                let typed_binexp =
                    InnerExp::Binary(op.clone(), typed_e1.clone().into(), typed_e2.into());
                TypedExp::set_type(typed_binexp, typed_e1.get_type())
            }
            BinaryOperator::And | BinaryOperator::Or => {
                let typed_binexp = InnerExp::Binary(op.clone(), typed_e1.into(), typed_e2.into());
                TypedExp::set_type(typed_binexp, Type::Int)
            }
            _ => {
                let t1 = typed_e1.get_type();
                let t2 = typed_e2.get_type();
                let common_type = get_common_type(&t1, &t2);
                let converted_e1 = convert_to(typed_e1, common_type.clone());
                let converted_e2 = convert_to(typed_e2, common_type.clone());
                let binary_exp =
                    InnerExp::Binary(op.clone(), converted_e1.into(), converted_e2.into());
                match op {
                    BinaryOperator::Mod
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseXor
                        if common_type == Type::Double =>
                    {
                        panic!("Can't apply % or bitwise operations to double");
                    }
                    BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Mod
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOr
                    | BinaryOperator::BitwiseXor => TypedExp::set_type(binary_exp, common_type),
                    _ => TypedExp::set_type(binary_exp, Type::Int),
                }
            }
        }
    }

    fn typecheck_increment_decrement<F>(&self, inner: &Exp, ctor: F) -> TypedExp
    where
        F: Fn(Box<TypedExp>) -> InnerExp,
    {
        let typed_inner = self.typecheck_exp(inner);
        let inner_exp = ctor(Box::new(typed_inner.clone()));
        TypedExp::set_type(inner_exp, typed_inner.get_type())
    }

    fn typecheck_assignment(&self, lhs: &Exp, rhs: &Exp) -> TypedExp {
        let typed_lhs = self.typecheck_exp(lhs);
        let lhs_type = typed_lhs.get_type();
        let typed_rhs = self.typecheck_exp(rhs);
        let converted_rhs = convert_to(typed_rhs, lhs_type.clone());
        let assign_exp = InnerExp::Assignment(typed_lhs.into(), converted_rhs.into());
        TypedExp::set_type(assign_exp, lhs_type)
    }

    fn typecheck_compound_assignment(&self, op: &BinaryOperator, lhs: &Exp, rhs: &Exp) -> TypedExp {
        let typed_lhs = self.typecheck_exp(lhs);
        let lhs_type = typed_lhs.get_type();
        let typed_rhs = self.typecheck_exp(rhs);
        let rhs_type = typed_rhs.get_type();
        if matches!(op, BinaryOperator::Mod |
            BinaryOperator::BitwiseAnd |
            BinaryOperator::BitwiseOr |
            BinaryOperator::BitwiseXor |
            BinaryOperator::BitshiftLeft |
            BinaryOperator::BitshiftRight if lhs_type == Type::Double || rhs_type == Type::Double)
        {
            panic!("Operator {:?} does not support double operands", op);
        }
        let (result_t, converted_rhs) =
            if op == &BinaryOperator::BitshiftLeft || op == &BinaryOperator::BitshiftRight {
                (lhs_type.clone(), typed_rhs)
            } else {
                // We perform usual arithmetic conversions for every compound assignment operator
                // EXCEPT left/right bitshift
                let common_type = get_common_type(&lhs_type, &rhs_type);
                (common_type.clone(), convert_to(typed_rhs, common_type))
            };

        let compound_assign_exp = InnerExp::CompoundAssignment {
            op: op.clone(),
            lhs: typed_lhs.into(),
            rhs: converted_rhs.into(),
            result_t,
        };
        TypedExp::set_type(compound_assign_exp, lhs_type)
    }

    fn typecheck_conditional(
        &self,
        condition: &Exp,
        then_result: &Exp,
        else_result: &Exp,
    ) -> TypedExp {
        let typed_condition = self.typecheck_exp(condition);
        let typed_then = self.typecheck_exp(then_result);
        let typed_else = self.typecheck_exp(else_result);
        let common_type = get_common_type(&typed_then.get_type(), &typed_else.get_type());
        let converted_then = convert_to(typed_then, common_type.clone());
        let converted_else = convert_to(typed_else, common_type.clone());
        let conditional_exp = InnerExp::Conditional {
            condition: typed_condition.into(),
            then_result: converted_then.into(),
            else_result: converted_else.into(),
        };
        TypedExp::set_type(conditional_exp, common_type)
    }

    fn typecheck_fun_call(&self, f: &str, args: &[Exp]) -> TypedExp {
        let f_type = self.symbol_table.get(f).t.clone();

        match f_type {
            Type::FunType {
                param_types,
                ret_type,
            } => {
                if param_types.len() != args.len() {
                    panic!("Function {} called with wrong number of arguments", f);
                }
                let converted_args =
                    args.iter()
                        .zip(param_types.iter())
                        .map(|(arg, param_type)| {
                            convert_to(self.typecheck_exp(arg), param_type.clone())
                        });
                let call_exp = InnerExp::FunCall {
                    f: f.to_string(),
                    args: converted_args.collect(),
                };
                TypedExp::set_type(call_exp, *ret_type)
            }
            _ => {
                panic!("Tried to use variable {} as function name", f);
            }
        }
    }

    // convert a constant to a static initializer, performing type conversion if
    // needed
    fn to_static_init(&self, exp: &Exp, var_type: &Type) -> InitialValue {
        match exp {
            Exp::Constant(c) => {
                let init_val = match const_convert(var_type, &c) {
                    T::ConstInt(i) => StaticInit::IntInit(i),
                    T::ConstLong(l) => StaticInit::LongInit(l),
                    T::ConstUInt(u) => StaticInit::UIntInit(u),
                    T::ConstULong(ul) => StaticInit::ULongInit(ul),
                    T::ConstDouble(d) => StaticInit::DoubleInit(d),
                };
                InitialValue::Initial(init_val)
            }
            _ => {
                panic!("Non-constant initializer on static variable");
            }
        }
    }

    fn typecheck_block(&mut self, ret_type: &Type, b: &Block<Exp>) -> Block<TypedExp> {
        Block(
            b.0.iter()
                .map(|item| self.typecheck_block_item(ret_type, item))
                .collect(),
        )
    }

    fn typecheck_block_item(
        &mut self,
        ret_type: &Type,
        block_item: &BlockItem<Exp>,
    ) -> BlockItem<TypedExp> {
        match block_item {
            BlockItem::S(s) => BlockItem::S(self.typecheck_statement(ret_type, s)),
            BlockItem::D(d) => BlockItem::D(self.typecheck_local_decl(d)),
        }
    }

    fn typecheck_statement(
        &mut self,
        ret_type: &Type,
        statement: &Statement<Exp>,
    ) -> Statement<TypedExp> {
        match statement {
            Statement::Return(e) => {
                let typed_e = self.typecheck_exp(e);
                Statement::Return(convert_to(typed_e, ret_type.clone()))
            }
            Statement::Expression(e) => Statement::Expression(self.typecheck_exp(e)),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition: self.typecheck_exp(condition),
                then_clause: self.typecheck_statement(&ret_type, &then_clause).into(),
                else_clause: else_clause
                    .as_ref()
                    .map(|e| self.typecheck_statement(&ret_type, e).into()),
            },
            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => {
                let typed_condition = self.typecheck_exp(condition);
                let switch_type = typed_condition.get_type();

                // Check and convert all case-constants to type of the switch
                let mut converted_cases = HashSet::new();
                for case in cases {
                    if let Some(value) = case {
                        // Convert value to type of the switch expression
                        let converted_value = match switch_type {
                            Type::Int | Type::UInt => (*value).rem(0x100000000) as i64, // 2^32
                            Type::Long | Type::ULong => *value,
                            _ => panic!("Switch condition must be integer type"),
                        };

                        // Check for duplicates after convert
                        if converted_cases.contains(&Some(converted_value)) {
                            panic!(
                                "Duplicate case value {} after type conversion",
                                converted_value
                            );
                        }

                        converted_cases.insert(Some(converted_value));
                    } else {
                        // default case
                        converted_cases.insert(None);
                    }
                }

                Statement::Switch {
                    condition: typed_condition,
                    body: self.typecheck_statement(&ret_type, body).into(),
                    cases: cases.clone(),
                    id: id.clone(),
                }
            }
            Statement::Case {
                condition,
                body,
                switch_label,
            } => Statement::Case {
                condition: *condition,
                body: self.typecheck_statement(&ret_type, body).into(),
                switch_label: switch_label.clone(),
            },
            Statement::Default { body, switch_label } => Statement::Default {
                body: self.typecheck_statement(&ret_type, body).into(),
                switch_label: switch_label.clone(),
            },
            Statement::Compound(block) => {
                Statement::Compound(self.typecheck_block(&ret_type, block))
            }
            Statement::While {
                condition,
                body,
                id,
            } => Statement::While {
                condition: self.typecheck_exp(condition),
                body: self.typecheck_statement(&ret_type, body).into(),
                id: id.clone(),
            },
            Statement::DoWhile {
                body,
                condition,
                id,
            } => Statement::DoWhile {
                body: self.typecheck_statement(&ret_type, body).into(),
                condition: self.typecheck_exp(condition),
                id: id.clone(),
            },
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => {
                let typechecked_for_init = match init {
                    ForInit::InitDecl(VariableDeclaration {
                        storage_class: Some(_),
                        ..
                    }) => {
                        panic!("Storage class not permitted on declaration in for loop header");
                    }
                    ForInit::InitDecl(d) => ForInit::InitDecl(self.typecheck_local_var_decl(d)),
                    ForInit::InitExp(e) => {
                        ForInit::InitExp(Option::map(e.as_ref(), |e| self.typecheck_exp(e)))
                    }
                };
                Statement::For {
                    init: typechecked_for_init,
                    condition: Option::map(condition.as_ref(), |c| self.typecheck_exp(c)),
                    post: Option::map(post.as_ref(), |p| self.typecheck_exp(p)),
                    body: self.typecheck_statement(&ret_type, body).into(),
                    id: id.clone(),
                }
            }
            Statement::LabelledStatement(lbl, stmt, ..) => Statement::LabelledStatement(
                lbl.clone(),
                self.typecheck_statement(ret_type, &stmt).into(),
            ),
            Statement::Null => Statement::Null,
            Statement::Break(s) => Statement::Break(s.to_string()),
            Statement::Continue(s) => Statement::Continue(s.to_string()),
            Statement::Goto(s) => Statement::Goto(s.to_string()),
        }
    }

    fn typecheck_local_decl(&mut self, decl: &Declaration<Exp>) -> Declaration<TypedExp> {
        match decl {
            Declaration::VarDecl(vd) => Declaration::VarDecl(self.typecheck_local_var_decl(vd)),
            Declaration::FunDecl(fd) => Declaration::FunDecl(self.typecheck_fn_decl(fd)),
        }
    }

    fn typecheck_local_var_decl(
        &mut self,
        var_decl: &VariableDeclaration<Exp>,
    ) -> VariableDeclaration<TypedExp> {
        let VariableDeclaration {
            name,
            var_type,
            init,
            storage_class,
        } = var_decl;
        match storage_class {
            Some(StorageClass::Extern) => {
                if init.is_some() {
                    panic!("initializer on local extern declaration");
                }
                match self.symbol_table.get_opt(name.as_str()) {
                    Some(Entry { t, .. }) => {
                        // If an external local var is already in the symbol table, don't need
                        // to add it
                        if t != var_type {
                            panic!("Variable {} redeclared with different type", name);
                        }
                    }
                    None => {
                        self.symbol_table.add_static_var(
                            name,
                            var_type.clone(),
                            true,
                            InitialValue::NoInitializer,
                        );
                    }
                }
                VariableDeclaration {
                    name: name.clone(),
                    init: None,
                    storage_class: *storage_class,
                    var_type: var_type.clone(),
                }
            }
            Some(StorageClass::Static) => {
                let zero_init = InitialValue::Initial(zero(var_type));
                let static_init = match init {
                    Some(i) => self.to_static_init(i, var_type),
                    None => zero_init.clone(),
                };
                self.symbol_table
                    .add_static_var(name, var_type.clone(), false, static_init);

                // NOTE: we won't actually use init in subsequent passes so we can drop it
                VariableDeclaration {
                    name: name.clone(),
                    init: None,
                    storage_class: *storage_class,
                    var_type: var_type.clone(),
                }
            }
            None => {
                self.symbol_table.add_automatic_var(name, var_type.clone());
                VariableDeclaration {
                    name: name.clone(),
                    var_type: var_type.clone(),
                    storage_class: *storage_class,
                    init: Option::map(init.as_ref(), |e| {
                        convert_to(self.typecheck_exp(e), var_type.clone())
                    }),
                }
            }
        }
    }

    fn typecheck_fn_decl(
        &mut self,
        func_decl: &FunctionDeclaration<Exp>,
    ) -> FunctionDeclaration<TypedExp> {
        let FunctionDeclaration {
            name,
            fun_type,
            params,
            body,
            storage_class,
        } = func_decl;

        let has_body = body.is_some();

        let global = *storage_class != Some(StorageClass::Static);
        // helper function to reconcile current and previous declarations
        let check_against_previous = |entry: &Entry| {
            let Entry { t: prev_t, attrs } = entry;
            if prev_t != fun_type {
                panic!("Redeclared function {} with a different type", name);
            } else {
                match attrs {
                    IdentifierAttrs::FunAttr {
                        global: prev_global,
                        defined: prev_defined,
                        ..
                    } => {
                        if *prev_defined && has_body {
                            panic!("Defined body of function {} twice", name);
                        } else if *prev_global && *storage_class == Some(StorageClass::Static) {
                            panic!("Static function declaration follows non-static");
                        } else {
                            let defined = has_body || *prev_defined;
                            (defined, *prev_global)
                        }
                    }
                    _ => {
                        panic!(
                            "Internal error: symbol has function type but not function attributes"
                        );
                    }
                }
            }
        };

        let old_decl = self.symbol_table.get_opt(name.as_str());

        let (defined, global) = if let Some(old_d) = old_decl {
            check_against_previous(old_d)
        } else {
            (has_body, global)
        };

        self.symbol_table
            .add_fun(name.as_str(), fun_type.clone(), global, defined);
        let (params_ts, return_t) = match &fun_type {
            Type::FunType {
                param_types,
                ret_type,
            } => (param_types, ret_type),
            _ => panic!("Internal error, function has non-function type"),
        };

        if has_body {
            for (p, t) in zip(params, params_ts) {
                self.symbol_table.add_automatic_var(&p, t.clone());
            }
        }
        let body = Option::map(body.as_ref(), |b| self.typecheck_block(return_t, b));

        FunctionDeclaration {
            name: name.to_string(),
            fun_type: fun_type.clone(),
            params: params.to_vec(),
            body,
            storage_class: *storage_class,
        }
    }

    fn typecheck_file_scope_var_decl(
        &mut self,
        var_decl: &VariableDeclaration<Exp>,
    ) -> VariableDeclaration<TypedExp> {
        let VariableDeclaration {
            name,
            var_type,
            init,
            storage_class,
        } = var_decl;

        let default_init = if let Some(StorageClass::Extern) = *storage_class {
            InitialValue::NoInitializer
        } else {
            InitialValue::Tentative
        };
        let static_init = match init {
            Some(i) => self.to_static_init(i, var_type),
            None => default_init,
        };

        let current_global = *storage_class != Some(StorageClass::Static);
        let old_decl = self.symbol_table.get_opt(name.as_str());
        let (global, init) = if let Some(old_d) = old_decl {
            let Entry { t, attrs } = old_d;
            if t != var_type {
                panic!("Variable {} redeclared with different type", name);
            } else {
                if let IdentifierAttrs::StaticAttr {
                    init: prev_init,
                    global: prev_global,
                } = attrs
                {
                    let global = if storage_class == &Some(StorageClass::Extern) {
                        *prev_global
                    } else if current_global == *prev_global {
                        current_global
                    } else {
                        panic!("Conflicting variable linkage");
                    };
                    let init = match (prev_init, &static_init) {
                        (InitialValue::Initial(_), InitialValue::Initial(_)) => {
                            panic!("Conflicting global variable definition");
                        }
                        (InitialValue::Initial(_), _) => prev_init.clone(),
                        (
                            InitialValue::Tentative,
                            InitialValue::Tentative | InitialValue::NoInitializer,
                        ) => InitialValue::Tentative,
                        (_, InitialValue::Initial(_)) | (InitialValue::NoInitializer, _) => {
                            static_init
                        }
                    };
                    (global, init)
                } else {
                    panic!(
                        "Internal error, file-scope variable previously declared as local variable or function"
                    );
                }
            }
        } else {
            (current_global, static_init)
        };
        self.symbol_table
            .add_static_var(name, var_type.clone(), global, init);
        // Okay to drop initializer b/c it's never used after this pass
        VariableDeclaration {
            name: name.to_string(),
            var_type: var_type.clone(),
            init: None,
            storage_class: *storage_class,
        }
    }

    fn typecheck_global_decl(&mut self, decl: &Declaration<Exp>) -> Declaration<TypedExp> {
        match decl {
            Declaration::FunDecl(fd) => Declaration::FunDecl(self.typecheck_fn_decl(fd)),
            Declaration::VarDecl(vd) => {
                Declaration::VarDecl(self.typecheck_file_scope_var_decl(vd))
            }
        }
    }

    pub fn typecheck(&mut self, program: &Program<Exp>) -> Program<TypedExp> {
        Program(
            program
                .0
                .iter()
                .map(|decl| self.typecheck_global_decl(decl))
                .collect(),
        )
    }
}
