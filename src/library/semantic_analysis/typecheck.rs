use core::panic;
use std::iter::zip;

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

fn is_pointer(t: &Type) -> bool {
    matches!(t, Type::Pointer(_))
}

fn is_arithmetic(t: &Type) -> bool {
    match t {
        Type::Int | Type::UInt | Type::Long | Type::ULong | Type::Double => true,
        Type::FunType { .. } | Type::Pointer(_) => false,
    }
}

fn is_integer(t: &Type) -> bool {
    match t {
        Type::Int | Type::Long | Type::UInt | Type::ULong => true,
        Type::Double | Type::FunType { .. } | Type::Pointer(_) => false,
    }
}

fn is_lvalue(e: &Exp) -> bool {
    matches!(e, Exp::Var(_) | Exp::Dereference(_))
}

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

fn is_null_pointer_constant(exp: &InnerExp) -> bool {
    matches!(
        exp,
        InnerExp::Constant(T::ConstInt(0))
            | InnerExp::Constant(T::ConstLong(0))
            | InnerExp::Constant(T::ConstUInt(0))
            | InnerExp::Constant(T::ConstULong(0))
    )
}

fn get_common_pointer_type(e1: &TypedExp, e2: &TypedExp) -> Type {
    if e1.t == e2.t {
        e1.t.clone()
    } else if is_null_pointer_constant(&e1.e) {
        e2.t.clone()
    } else if is_null_pointer_constant(&e2.e) {
        e1.t.clone()
    } else {
        panic!("Expressions have incompatible types: {:?}, {:?}", e1, e2)
    }
}

fn convert_by_assignment(e: TypedExp, target_type: Type) -> TypedExp {
    if e.t == target_type {
        e
    } else if is_arithmetic(&e.t) && is_arithmetic(&target_type) {
        convert_to(e, target_type)
    } else if is_null_pointer_constant(&e.e) && is_pointer(&target_type) {
        convert_to(e, target_type)
    } else {
        panic!(
            "Cannot convert type {:?} for assignment: {:?}",
            target_type, e
        )
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
            } => self.typecheck_cast(target_type, inner),
            Exp::Unary(UnaryOperator::Not, inner) => self.typecheck_not(&inner),
            Exp::Unary(UnaryOperator::Complement, inner) => self.typecheck_complement(&inner),
            Exp::Unary(UnaryOperator::Negate, inner) => self.typecheck_negate(&inner),
            Exp::Unary(op, inner) => self.typecheck_incr(&op, &inner),
            Exp::Binary(op, e1, e2) => match op {
                BinaryOperator::And | BinaryOperator::Or => self.typecheck_logical(op, e1, e2),
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Mod
                | BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseOr
                | BinaryOperator::BitwiseXor => self.typecheck_arithmetic(op, e1, e2),
                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::GreaterThan
                | BinaryOperator::GreaterOrEqual
                | BinaryOperator::LessThan
                | BinaryOperator::LessOrEqual => self.typecheck_comparison(op, e1, e2),
                BinaryOperator::BitshiftLeft | BinaryOperator::BitshiftRight => {
                    self.typecheck_bitshift(op, e1, e2)
                }
            },
            Exp::PostfixDecr(inner) => self.typecheck_postfix_decr(inner),
            Exp::PostfixIncr(inner) => self.typecheck_postfix_incr(inner),
            Exp::Assignment(lhs, rhs) => self.typecheck_assignment(lhs, rhs),
            Exp::CompoundAssign(op, lhs, rhs) => self.typecheck_compound_assignment(op, lhs, rhs),
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => self.typecheck_conditional(condition, then_result, else_result),
            Exp::FunCall { f, args } => self.typecheck_fun_call(f, args),
            Exp::Dereference(inner) => self.typecheck_dereference(inner),
            Exp::AddrOf(inner) => self.typecheck_addr_of(inner),
        }
    }

    fn typecheck_cast(&self, target_type: &Type, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);
        match (target_type, typed_inner.t) {
            (Type::Double, Type::Pointer(_)) | (Type::Pointer(_), Type::Double) => {
                panic!("Cannot cast between pointer and double")
            }
            _ => {
                let cast_exp = InnerExp::Cast {
                    target_type: target_type.clone(),
                    e: self.typecheck_exp(inner).into(),
                };
                TypedExp::set_type(cast_exp, target_type.clone())
            }
        }
    }

    fn typecheck_not(&self, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);
        let not_exp = InnerExp::Unary(UnaryOperator::Not, typed_inner.into());
        TypedExp::set_type(not_exp, Type::Int)
    }

    fn typecheck_complement(&self, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);

        if typed_inner.t == Type::Double || is_pointer(&typed_inner.t) {
            panic!("Bitwise complement only valid for integer types");
        } else {
            let complement_exp =
                InnerExp::Unary(UnaryOperator::Complement, typed_inner.clone().into());
            TypedExp::set_type(complement_exp, typed_inner.t)
        }
    }

    fn typecheck_negate(&self, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);
        if let Type::Pointer(_) = typed_inner.t {
            panic!("Can't negate a pointer {:?}", inner);
        } else {
            let negate_exp = InnerExp::Unary(UnaryOperator::Negate, typed_inner.clone().into());
            TypedExp::set_type(negate_exp, typed_inner.t)
        }
    }

    fn typecheck_incr(&self, op: &UnaryOperator, inner: &Exp) -> TypedExp {
        if is_lvalue(inner) {
            let typed_inner = self.typecheck_exp(inner);
            let typed_exp = InnerExp::Unary(op.clone(), typed_inner.clone().into());
            TypedExp::set_type(typed_exp, typed_inner.t)
        } else {
            panic!("Operand of ++/-- must be an lvalue");
        }
    }

    fn typecheck_postfix_decr(&self, e: &Exp) -> TypedExp {
        if is_lvalue(e) {
            // Result has same value as e; no conversions required.
            let typed_e = self.typecheck_exp(e);
            let result_type = typed_e.get_type();
            TypedExp::set_type(InnerExp::PostfixDecr(typed_e.into()), result_type)
        } else {
            panic!("Operand of postfix -- must be an lvalue");
        }
    }

    fn typecheck_postfix_incr(&self, e: &Exp) -> TypedExp {
        if is_lvalue(e) {
            // Result has same value as e; no conversions required.
            let typed_e = self.typecheck_exp(e);
            let result_type = typed_e.get_type();
            TypedExp::set_type(InnerExp::PostfixIncr(typed_e.into()), result_type)
        } else {
            panic!("Operand of postfix ++ must be an lvalue");
        }
    }

    fn typecheck_logical(&self, op: &BinaryOperator, e1: &Exp, e2: &Exp) -> TypedExp {
        let typed_e1 = self.typecheck_exp(e1);
        let typed_e2 = self.typecheck_exp(e2);
        let typed_binexp = InnerExp::Binary(op.clone(), typed_e1.clone().into(), typed_e2.into());
        TypedExp::set_type(typed_binexp, Type::Int)
    }

    // handle ^, & and | here because they follow same typing rules as arithmetic ops
    fn typecheck_arithmetic(&self, op: &BinaryOperator, e1: &Exp, e2: &Exp) -> TypedExp {
        let typed_e1 = self.typecheck_exp(e1);
        let typed_e2 = self.typecheck_exp(e2);

        if is_pointer(&typed_e1.t) || is_pointer(&typed_e2.t) {
            panic!("arithmetic operations not permitted on pointers");
        } else {
            let common_type = get_common_type(&typed_e1.t, &typed_e2.t);
            let converted_e1 = convert_to(typed_e1, common_type.clone());
            let converted_e2 = convert_to(typed_e2, common_type.clone());
            let binary_exp = InnerExp::Binary(op.clone(), converted_e1.into(), converted_e2.into());

            match op {
                BinaryOperator::Mod
                | BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseOr
                | BinaryOperator::BitwiseXor
                    if common_type == Type::Double =>
                {
                    panic!("Can't apply %, &, | or ^ to double");
                }
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Mod
                | BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseOr
                | BinaryOperator::BitwiseXor => TypedExp::set_type(binary_exp, common_type),
                op => panic!("Internal error: {:?} should be typechecked elsewhere.", op),
            }
        }
    }

    fn typecheck_comparison(&self, op: &BinaryOperator, e1: &Exp, e2: &Exp) -> TypedExp {
        let typed_e1 = self.typecheck_exp(e1);
        let typed_e2 = self.typecheck_exp(e2);

        let common_type = if is_pointer(&typed_e1.t) || is_pointer(&typed_e2.t) {
            get_common_pointer_type(&typed_e1, &typed_e2)
        } else {
            get_common_type(&typed_e1.t, &typed_e2.t)
        };

        let converted_e1 = convert_to(typed_e1, common_type.clone());
        let converted_e2 = convert_to(typed_e2, common_type);
        let binary_exp =
            InnerExp::Binary(op.clone(), Box::new(converted_e1), Box::new(converted_e2));
        TypedExp::set_type(binary_exp, Type::Int)
    }

    fn typecheck_bitshift(&self, op: &BinaryOperator, e1: &Exp, e2: &Exp) -> TypedExp {
        let typed_e1 = self.typecheck_exp(e1);
        let typed_e2 = self.typecheck_exp(e2);

        if !(is_integer(&typed_e1.get_type()) && is_integer(&typed_e2.get_type())) {
            panic!("Both operands of bit shift operation must be integers");
        } else {
            // Don't perform usual arithmetic conversions; results has type of left operand
            let typed_binop =
                InnerExp::Binary(op.clone(), typed_e1.clone().into(), typed_e2.into());
            TypedExp::set_type(typed_binop, typed_e1.get_type())
        }
    }

    fn typecheck_assignment(&self, lhs: &Exp, rhs: &Exp) -> TypedExp {
        if is_lvalue(lhs) {
            let typed_lhs = self.typecheck_exp(lhs);
            let lhs_type = typed_lhs.get_type();
            let typed_rhs = self.typecheck_exp(rhs);
            let converted_rhs = convert_by_assignment(typed_rhs, lhs_type.clone());
            let assign_exp = InnerExp::Assignment(typed_lhs.into(), converted_rhs.into());
            TypedExp::set_type(assign_exp, lhs_type)
        } else {
            panic!("left hand side of assignment is invalid lvalue");
        }
    }

    fn typecheck_compound_assignment(&self, op: &BinaryOperator, lhs: &Exp, rhs: &Exp) -> TypedExp {
        if is_lvalue(lhs) {
            let typed_lhs = self.typecheck_exp(lhs);
            let lhs_type = typed_lhs.get_type();
            let typed_rhs = self.typecheck_exp(rhs);
            let rhs_type = typed_rhs.get_type();

            match op {
                BinaryOperator::Mod
                | BinaryOperator::BitwiseAnd
                | BinaryOperator::BitwiseOr
                | BinaryOperator::BitwiseXor
                | BinaryOperator::BitshiftLeft
                | BinaryOperator::BitshiftRight
                    if !is_integer(&lhs_type) || !is_integer(&rhs_type) =>
                {
                    panic!("Operator {:?} does only supports integer operands", op)
                }
                BinaryOperator::Multiply | BinaryOperator::Divide
                    if is_pointer(&lhs_type) || is_pointer(&rhs_type) =>
                {
                    panic!("Operator {:?} does not support pointer operands", op)
                }
                _ => (),
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
        } else {
            panic!("Left-hand side of compound assignment must be an lvalue");
        }
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

        let common_type = if is_pointer(&typed_then.t) || is_pointer(&typed_else.t) {
            get_common_pointer_type(&typed_then, &typed_else)
        } else {
            get_common_type(&typed_then.t, &typed_else.t)
        };

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
                            convert_by_assignment(self.typecheck_exp(arg), param_type.clone())
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

    fn typecheck_dereference(&self, inner: &Exp) -> TypedExp {
        let typed_inner = self.typecheck_exp(inner);
        if let Type::Pointer(referenced_t) = typed_inner.get_type() {
            let deref_exp = InnerExp::Dereference(typed_inner.into());
            TypedExp::set_type(deref_exp, *referenced_t)
        } else {
            panic!("Tried to dereference non-ponter {:?}", inner)
        }
    }

    fn typecheck_addr_of(&self, inner: &Exp) -> TypedExp {
        match inner {
            Exp::Dereference(_) | Exp::Var(_) => {
                let typed_inner = self.typecheck_exp(inner);
                let inner_t = typed_inner.get_type();
                let addr_exp = InnerExp::AddrOf(typed_inner.into());
                TypedExp::set_type(addr_exp, Type::Pointer(inner_t.into()))
            }
            _ => panic!("Cannot take address of non-lvalue {:?}", inner),
        }
    }

    fn to_static_init(&self, exp: &Exp, var_type: &Type) -> InitialValue {
        match exp {
            Exp::Constant(c) => {
                let init_val = if is_pointer(var_type) {
                    if is_null_pointer_constant(&InnerExp::Constant(c.clone())) {
                        StaticInit::ULongInit(0)
                    } else {
                        panic!(
                            "Static pointers can only be initialized with null pointer constants"
                        );
                    }
                } else {
                    match const_convert(var_type, &c) {
                        T::ConstInt(i) => StaticInit::IntInit(i),
                        T::ConstLong(l) => StaticInit::LongInit(l),
                        T::ConstUInt(u) => StaticInit::UIntInit(u),
                        T::ConstULong(ul) => StaticInit::ULongInit(ul),
                        T::ConstDouble(d) => StaticInit::DoubleInit(d),
                    }
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
                Statement::Return(convert_by_assignment(typed_e, ret_type.clone()))
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
            Statement::LabelledStatement(lbl, stmt, ..) => Statement::LabelledStatement(
                lbl.clone(),
                self.typecheck_statement(ret_type, &stmt).into(),
            ),
            Statement::Case(e, s, id) => {
                let typed_e = self.typecheck_exp(&e);
                if typed_e.get_type() == Type::Double {
                    panic!("Case expression cannot be double");
                } else {
                    Statement::Case(
                        typed_e,
                        self.typecheck_statement(ret_type, s).into(),
                        id.clone(),
                    )
                }
            }
            Statement::Default(s, id) => {
                Statement::Default(self.typecheck_statement(&ret_type, s).into(), id.clone())
            }
            Statement::Switch {
                control,
                body,
                cases,
                id,
            } => {
                let typed_control = self.typecheck_exp(control);
                if !is_integer(&typed_control.get_type()) {
                    panic!("Switch control expression must have integer type");
                } else {
                    Statement::Switch {
                        control: typed_control,
                        body: self.typecheck_statement(ret_type, &body).into(),
                        id: id.clone(),
                        cases: cases.clone(),
                    }
                }
            }
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
                        convert_by_assignment(self.typecheck_exp(e), var_type.clone())
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
