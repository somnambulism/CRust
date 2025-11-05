use std::iter::once;

use crate::library::{
    ast::{
        block_items::{
            Block, BlockItem, Declaration, ForInit, FunctionDeclaration as AstFunction,
            Program as AstProgram, Statement, SwitchCases, VariableDeclaration,
        },
        ops::{BinaryOperator as AstBinaryOperator, UnaryOperator as AstUnaryOperator},
        typed_exp::{InnerExp, TypedExp},
    },
    r#const::{INT_ONE, INT_ZERO, T},
    const_convert::const_convert,
    initializers::zero,
    symbols::{IdentifierAttrs, InitialValue, SymbolTable},
    tacky::{BinaryOperator, Instruction, Program, TackyVal, TopLevel, UnaryOperator},
    types::Type,
    util::unique_ids::{make_label, make_temporary},
};

pub struct TackyGen {
    pub symbol_table: SymbolTable,
}

fn break_label(label: String) -> String {
    format!("break.{}", label)
}

fn continue_label(label: String) -> String {
    format!("continue.{}", label)
}

fn case_label(condition: i64, switch_label: String) -> String {
    format!("switch.{}.case.{}", switch_label, condition)
}

fn default_label(switch_label: String) -> String {
    format!("switch.{}.default", switch_label)
}

fn mk_const(t: &Type, i: i64) -> T {
    let as_int = T::ConstInt(i as i32);
    const_convert(t, &as_int)
}

fn mk_ast_const(t: &Type, i: i64) -> TypedExp {
    TypedExp {
        e: InnerExp::Constant(mk_const(t, i)),
        t: t.clone(),
    }
}

fn convert_op(op: AstUnaryOperator) -> UnaryOperator {
    match op {
        AstUnaryOperator::Complement => UnaryOperator::Complement,
        AstUnaryOperator::Negate => UnaryOperator::Negate,
        AstUnaryOperator::Not => UnaryOperator::Not,
        AstUnaryOperator::Incr | AstUnaryOperator::Decr => {
            panic!("Internal error: should not convert ++/-- directly to TACKY unary ops");
        }
    }
}

fn convert_binop(op: AstBinaryOperator) -> BinaryOperator {
    match op {
        AstBinaryOperator::Add => BinaryOperator::Add,
        AstBinaryOperator::Subtract => BinaryOperator::Subtract,
        AstBinaryOperator::Multiply => BinaryOperator::Multiply,
        AstBinaryOperator::Divide => BinaryOperator::Divide,
        AstBinaryOperator::Mod => BinaryOperator::Mod,
        AstBinaryOperator::BitwiseAnd => BinaryOperator::BitwiseAnd,
        AstBinaryOperator::BitwiseOr => BinaryOperator::BitwiseOr,
        AstBinaryOperator::BitwiseXor => BinaryOperator::Xor,
        AstBinaryOperator::BitshiftLeft => BinaryOperator::LeftShift,
        AstBinaryOperator::BitshiftRight => BinaryOperator::RightShift,
        AstBinaryOperator::Equal => BinaryOperator::Equal,
        AstBinaryOperator::NotEqual => BinaryOperator::NotEqual,
        AstBinaryOperator::LessThan => BinaryOperator::LessThan,
        AstBinaryOperator::LessOrEqual => BinaryOperator::LessOrEqual,
        AstBinaryOperator::GreaterThan => BinaryOperator::GreaterThan,
        AstBinaryOperator::GreaterOrEqual => BinaryOperator::GreaterOrEqual,
        AstBinaryOperator::And | AstBinaryOperator::Or => {
            panic!("Internal error, cannot convert these directly to TACKY binops");
        }
    }
}

impl TackyGen {
    pub fn new(symbols: SymbolTable) -> Self {
        TackyGen {
            symbol_table: symbols,
        }
    }

    fn create_tmp(&mut self, t: Type) -> String {
        let name = make_temporary();
        self.symbol_table.add_automatic_var(&name, t);
        name
    }

    fn emit_decr(
        &mut self,
        op: AstBinaryOperator,
        inner: TypedExp,
    ) -> (Vec<Instruction>, TackyVal) {
        let v = match inner.e {
            InnerExp::Var(v) => v,
            _ => panic!("Invalid lvalue for postfix increment/decrement"),
        };
        let dst = TackyVal::Var(self.create_tmp(inner.t.clone()));
        let instrs = vec![
            Instruction::Copy {
                src: TackyVal::Var(v.clone()),
                dst: dst.clone(),
            },
            Instruction::Binary {
                op: convert_binop(op),
                src1: TackyVal::Var(v.clone()),
                src2: TackyVal::Constant(mk_const(&inner.t, 1)),
                dst: TackyVal::Var(v),
            },
        ];
        (instrs, dst)
    }

    fn emit_tacky_for_exp(&mut self, exp: TypedExp) -> (Vec<Instruction>, TackyVal) {
        let TypedExp { e, t } = exp;
        match e {
            InnerExp::Constant(c) => (vec![], TackyVal::Constant(c)),
            InnerExp::Var(v) => (vec![], TackyVal::Var(v)),
            InnerExp::Cast { target_type, e } => self.emit_cast_expression(target_type, *e),
            InnerExp::Unary(AstUnaryOperator::Incr, v) => {
                self.emit_compound_expression(AstBinaryOperator::Add, *v, mk_ast_const(&t, 1), t)
            }
            InnerExp::Unary(AstUnaryOperator::Decr, v) => self.emit_compound_expression(
                AstBinaryOperator::Subtract,
                *v,
                mk_ast_const(&t, 1),
                t,
            ),
            InnerExp::Unary(op, inner) => self.emit_unary_expression(t, op, *inner),
            InnerExp::Binary(AstBinaryOperator::And, e1, e2) => self.emit_and_expression(*e1, *e2),
            InnerExp::Binary(AstBinaryOperator::Or, e1, e2) => self.emit_or_expression(*e1, *e2),
            InnerExp::Binary(op, e1, e2) => self.emit_binary_expression(t, op, *e1, *e2),
            InnerExp::CompoundAssignment {
                op,
                lhs,
                rhs,
                result_t,
            } => self.emit_compound_expression(op, *lhs, *rhs, result_t),
            InnerExp::PostfixIncrement(inner) => self.emit_decr(AstBinaryOperator::Add, *inner),
            InnerExp::PostfixDecrement(inner) => {
                self.emit_decr(AstBinaryOperator::Subtract, *inner)
            }
            InnerExp::Assignment(lhs, rhs) => {
                if let TypedExp {
                    e: InnerExp::Var(v),
                    ..
                } = *lhs
                {
                    self.emit_assignment(&v, *rhs)
                } else {
                    panic!("Internal error: bad lvalue");
                }
            }
            InnerExp::Conditional {
                condition,
                then_result,
                else_result,
            } => self.emit_conditional_expression(t, *condition, *then_result, *else_result),
            InnerExp::FunCall { f, args } => self.emit_fun_call(t, f.as_str(), args),
        }
    }

    fn emit_unary_expression(
        &mut self,
        t: Type,
        op: AstUnaryOperator,
        inner: TypedExp,
    ) -> (Vec<Instruction>, TackyVal) {
        let (mut eval_inner, v) = self.emit_tacky_for_exp(inner);
        // define a temporary variable to hold result of this expression
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let tacky_op = convert_op(op);
        eval_inner.push(Instruction::Unary {
            op: tacky_op,
            src: v,
            dst: dst.clone(),
        });
        (eval_inner, dst)
    }

    fn emit_cast_expression(
        &mut self,
        target_type: Type,
        inner: TypedExp,
    ) -> (Vec<Instruction>, TackyVal) {
        let (mut eval_inner, result) = self.emit_tacky_for_exp(inner.clone());
        let src_type = inner.get_type();

        if src_type == target_type {
            (eval_inner, result)
        } else {
            let dst_name = self.create_tmp(target_type.clone());
            let dst = TackyVal::Var(dst_name);
            let cast_instruction =
                self.get_cast_instruction(result, dst.clone(), src_type, target_type);

            eval_inner.push(cast_instruction);
            (eval_inner, dst)
        }
    }

    fn get_cast_instruction(
        &self,
        src: TackyVal,
        dst: TackyVal,
        src_t: Type,
        dst_t: Type,
    ) -> Instruction {
        match (dst_t.clone(), src_t.clone()) {
            (Type::Double, _) => {
                if src_t.is_signed() {
                    Instruction::IntToDouble {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                } else {
                    Instruction::UIntToDouble {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                }
            }
            (_, Type::Double) => {
                if dst_t.is_signed() {
                    Instruction::DoubleToInt {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                } else {
                    Instruction::DoubleToUInt {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                }
            }
            (_, _) => {
                // cast between int types
                if dst_t.get_size() == src_t.get_size() {
                    Instruction::Copy {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                } else if dst_t.get_size() < src_t.get_size() {
                    Instruction::Truncate {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                } else if src_t.is_signed() {
                    Instruction::SignExtend {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                } else {
                    Instruction::ZeroExtend {
                        src: src.clone(),
                        dst: dst.clone(),
                    }
                }
            }
        }
    }

    fn emit_binary_expression(
        &mut self,
        t: Type,
        op: AstBinaryOperator,
        e1: TypedExp,
        e2: TypedExp,
    ) -> (Vec<Instruction>, TackyVal) {
        let (eval_v1, v1) = self.emit_tacky_for_exp(e1);
        let (eval_v2, v2) = self.emit_tacky_for_exp(e2);
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let tacky_op = convert_binop(op);
        let mut instructions = eval_v1;
        instructions.extend(eval_v2);
        instructions.push(Instruction::Binary {
            op: tacky_op,
            src1: v1,
            src2: v2,
            dst: dst.clone(),
        });
        (instructions, dst)
    }

    fn emit_and_expression(&mut self, e1: TypedExp, e2: TypedExp) -> (Vec<Instruction>, TackyVal) {
        let (eval_v1, v1) = self.emit_tacky_for_exp(e1);
        let (eval_v2, v2) = self.emit_tacky_for_exp(e2);
        let false_label = make_label("and_false");
        let end_label = make_label("and_end");
        let dst_name = self.create_tmp(Type::Int);
        let dst = TackyVal::Var(dst_name);
        let mut instructions = eval_v1;
        instructions.push(Instruction::JumpIfZero(v1, false_label.clone()));
        instructions.extend(eval_v2);
        instructions.extend(vec![
            Instruction::JumpIfZero(v2, false_label.clone()),
            Instruction::Copy {
                src: TackyVal::Constant(INT_ONE),
                dst: dst.clone(),
            },
            Instruction::Jump(end_label.clone()),
            Instruction::Label(false_label),
            Instruction::Copy {
                src: TackyVal::Constant(INT_ZERO),
                dst: dst.clone(),
            },
            Instruction::Label(end_label),
        ]);
        (instructions, dst)
    }

    fn emit_or_expression(&mut self, e1: TypedExp, e2: TypedExp) -> (Vec<Instruction>, TackyVal) {
        let (eval_v1, v1) = self.emit_tacky_for_exp(e1);
        let (eval_v2, v2) = self.emit_tacky_for_exp(e2);
        let true_label = make_label("or_true");
        let end_label = make_label("or_end");
        let dst_name = self.create_tmp(Type::Int);
        let dst = TackyVal::Var(dst_name);
        let mut instructions = eval_v1;
        instructions.push(Instruction::JumpIfNotZero(v1, true_label.clone()));
        instructions.extend(eval_v2);
        instructions.extend(vec![
            Instruction::JumpIfNotZero(v2, true_label.clone()),
            Instruction::Copy {
                src: TackyVal::Constant(INT_ZERO),
                dst: dst.clone(),
            },
            Instruction::Jump(end_label.clone()),
            Instruction::Label(true_label),
            Instruction::Copy {
                src: TackyVal::Constant(INT_ONE),
                dst: dst.clone(),
            },
            Instruction::Label(end_label),
        ]);
        (instructions, dst)
    }

    fn emit_assignment(&mut self, v: &str, rhs: TypedExp) -> (Vec<Instruction>, TackyVal) {
        let (mut rhs_instructions, rhs_result) = self.emit_tacky_for_exp(rhs);
        rhs_instructions.push(Instruction::Copy {
            src: rhs_result,
            dst: TackyVal::Var(v.to_string()),
        });
        (rhs_instructions, TackyVal::Var(v.to_string()))
    }

    fn emit_compound_expression(
        &mut self,
        op: AstBinaryOperator,
        lhs: TypedExp,
        rhs: TypedExp,
        result_t: Type,
    ) -> (Vec<Instruction>, TackyVal) {
        // make sure it's an lvalue
        let v = match lhs.e {
            InnerExp::Var(v) => v,
            _ => panic!("bad lvalue in compound assignment or prefix incr/decr"),
        };
        // evaluate RHS - type checker already added conversion to common type if one is needed
        let (mut instructions, rhs) = self.emit_tacky_for_exp(rhs);
        let dst = TackyVal::Var(v.to_string());
        let tacky_op = convert_binop(op);

        let operation_and_assignment = if result_t == lhs.t {
            // result of binary operation already has correct destination type
            vec![Instruction::Binary {
                op: tacky_op,
                src1: dst.clone(),
                src2: rhs,
                dst: dst.clone(),
            }]
        } else {
            /*
             * must convert LHS to op type, then convert result back, so we'll have
             * tmp = <cast v to result_type>
             * tmp = tmp op rhs
             * lhs = <cast tmp to lhs.type>
             */
            let tmp = TackyVal::Var(self.create_tmp(result_t.clone()));
            let cast_lhs_to_tmp = self.get_cast_instruction(
                dst.clone(),
                tmp.clone(),
                lhs.t.clone(),
                result_t.clone(),
            );
            let binary_instr = Instruction::Binary {
                op: tacky_op,
                src1: tmp.clone(),
                src2: rhs,
                dst: tmp.clone(),
            };

            let cast_tmp_to_lhs = self.get_cast_instruction(tmp, dst.clone(), result_t, lhs.t);
            vec![cast_lhs_to_tmp, binary_instr, cast_tmp_to_lhs]
        };

        instructions.extend(operation_and_assignment);
        (instructions, dst)
    }

    fn emit_conditional_expression(
        &mut self,
        t: Type,
        condition: TypedExp,
        e1: TypedExp,
        e2: TypedExp,
    ) -> (Vec<Instruction>, TackyVal) {
        let (eval_cond, c) = self.emit_tacky_for_exp(condition);
        let (eval_v1, v1) = self.emit_tacky_for_exp(e1);
        let (eval_v2, v2) = self.emit_tacky_for_exp(e2);
        let e2_label = make_label("conditional_else");
        let end_label = make_label("conditional_end");
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let mut instructions = eval_cond;
        instructions.push(Instruction::JumpIfZero(c, e2_label.clone()));
        instructions.extend(eval_v1);
        instructions.extend(vec![
            Instruction::Copy {
                src: v1,
                dst: dst.clone(),
            },
            Instruction::Jump(end_label.clone()),
            Instruction::Label(e2_label),
        ]);
        instructions.extend(eval_v2);
        instructions.extend(vec![
            Instruction::Copy {
                src: v2,
                dst: dst.clone(),
            },
            Instruction::Label(end_label),
        ]);
        (instructions, dst)
    }

    fn emit_fun_call(
        &mut self,
        t: Type,
        f: &str,
        args: Vec<TypedExp>,
    ) -> (Vec<Instruction>, TackyVal) {
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let (arg_instructions, arg_vals): (Vec<Vec<Instruction>>, Vec<TackyVal>) = args
            .into_iter()
            .map(|arg| self.emit_tacky_for_exp(arg))
            .unzip();
        let mut instructions: Vec<Instruction> = arg_instructions.into_iter().flatten().collect();
        instructions.push(Instruction::FunCall {
            f: f.to_string(),
            args: arg_vals,
            dst: dst.clone(),
        });
        (instructions, dst)
    }

    fn emit_tacky_for_statement(&mut self, stmt: Statement<TypedExp>) -> Vec<Instruction> {
        match stmt {
            Statement::Return(e) => {
                let (mut eval_exp, v) = self.emit_tacky_for_exp(e);
                eval_exp.push(Instruction::Return(v));
                eval_exp
            }
            Statement::Expression(e) => {
                // evaluate expression but ignore the result
                let (eval_exp, _exp_result) = self.emit_tacky_for_exp(e);
                eval_exp
            }
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => self.emit_tacky_for_if_statement(condition, then_clause, else_clause),
            Statement::Compound(block) => {
                let Block(items) = block;
                items
                    .into_iter()
                    .flat_map(|item| self.emit_tacky_for_block_item(item))
                    .collect()
            }
            Statement::Break(id) => vec![Instruction::Jump(break_label(id))],
            Statement::Continue(id) => vec![Instruction::Jump(continue_label(id))],
            Statement::DoWhile {
                body,
                condition,
                id,
            } => self.emit_tacky_for_do_loop(*body, condition, id),
            Statement::While {
                condition,
                body,
                id,
            } => self.emit_tacky_for_while_loop(condition, *body, id),
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => self.emit_tacky_for_for_loop(init, condition, post, *body, id),
            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => self.emit_tacky_for_switch(condition, *body, cases, id),
            Statement::Case {
                condition,
                body,
                switch_label,
            } => {
                let mut instructions = self.emit_tacky_for_statement(*body);
                instructions.insert(0, Instruction::Label(case_label(condition, switch_label)));
                instructions
            }
            Statement::Default { body, switch_label } => {
                let mut instructions = self.emit_tacky_for_statement(*body);
                instructions.insert(0, Instruction::Label(default_label(switch_label)));
                instructions
            }
            Statement::Null => vec![],
            Statement::Labelled { label, statement } => {
                let mut instructions = self.emit_tacky_for_statement(*statement);
                instructions.insert(0, Instruction::Label(label));
                instructions
            }
            Statement::Goto(label) => {
                vec![Instruction::Jump(label)]
            }
        }
    }

    fn emit_tacky_for_block_item(&mut self, item: BlockItem<TypedExp>) -> Vec<Instruction> {
        match item {
            BlockItem::S(s) => self.emit_tacky_for_statement(s),
            BlockItem::D(d) => self.emit_local_declaration(d),
        }
    }

    fn emit_local_declaration(&mut self, d: Declaration<TypedExp>) -> Vec<Instruction> {
        match d {
            Declaration::VarDecl(VariableDeclaration {
                storage_class: Some(_),
                ..
            }) => vec![],
            Declaration::VarDecl(vd) => self.emit_var_declaration(vd),
            Declaration::FunDecl(_) => vec![],
        }
    }

    fn emit_var_declaration(&mut self, d: VariableDeclaration<TypedExp>) -> Vec<Instruction> {
        match d {
            VariableDeclaration {
                name,
                init: Some(e),
                ..
            } => {
                // treat declaration with initializer like an assignment expression
                let (eval_assignment, _assignment_result) = self.emit_assignment(&name, e);
                eval_assignment
            }
            VariableDeclaration {
                name: _,
                init: None,
                ..
            } => {
                // don't generate instructions for declaration without initializer
                vec![]
            }
        }
    }

    fn emit_tacky_for_if_statement(
        &mut self,
        condition: TypedExp,
        then_clause: Box<Statement<TypedExp>>,
        else_clause: Option<Box<Statement<TypedExp>>>,
    ) -> Vec<Instruction> {
        if let None = else_clause {
            // no else clause
            let end_label = make_label("if_end");
            let (mut eval_condition, c) = self.emit_tacky_for_exp(condition);
            eval_condition.push(Instruction::JumpIfZero(c, end_label.clone()));
            eval_condition.extend(self.emit_tacky_for_statement(*then_clause));
            eval_condition.push(Instruction::Label(end_label));
            eval_condition
        } else {
            let else_label = make_label("else");
            let end_label = make_label("if_end");
            let (mut eval_condition, c) = self.emit_tacky_for_exp(condition);
            eval_condition.push(Instruction::JumpIfZero(c, else_label.clone()));
            eval_condition.extend(self.emit_tacky_for_statement(*then_clause));
            eval_condition.extend(vec![
                Instruction::Jump(end_label.clone()),
                Instruction::Label(else_label),
            ]);
            eval_condition.extend(self.emit_tacky_for_statement(*else_clause.unwrap()));
            eval_condition.push(Instruction::Label(end_label));
            eval_condition
        }
    }

    fn emit_tacky_for_do_loop(
        &mut self,
        body: Statement<TypedExp>,
        condition: TypedExp,
        id: String,
    ) -> Vec<Instruction> {
        let start_label = make_label("do_loop_start");
        let cont_label = continue_label(id.clone());
        let br_label = break_label(id);
        let (eval_condition, c) = self.emit_tacky_for_exp(condition);
        let mut instructions = vec![Instruction::Label(start_label.clone())];
        instructions.extend(self.emit_tacky_for_statement(body));
        instructions.push(Instruction::Label(cont_label));
        instructions.extend(eval_condition);
        instructions.extend(vec![
            Instruction::JumpIfNotZero(c, start_label),
            Instruction::Label(br_label),
        ]);
        instructions
    }

    fn emit_tacky_for_while_loop(
        &mut self,
        condition: TypedExp,
        body: Statement<TypedExp>,
        id: String,
    ) -> Vec<Instruction> {
        let cont_label = continue_label(id.clone());
        let br_label = break_label(id);
        let (eval_condition, c) = self.emit_tacky_for_exp(condition);
        let mut instructions = vec![Instruction::Label(cont_label.clone())];
        instructions.extend(eval_condition);
        instructions.push(Instruction::JumpIfZero(c, br_label.clone()));
        instructions.extend(self.emit_tacky_for_statement(body));
        instructions.extend(vec![
            Instruction::Jump(cont_label),
            Instruction::Label(br_label),
        ]);
        instructions
    }

    fn emit_tacky_for_for_loop(
        &mut self,
        init: ForInit<TypedExp>,
        condition: Option<TypedExp>,
        post: Option<TypedExp>,
        body: Statement<TypedExp>,
        id: String,
    ) -> Vec<Instruction> {
        // generate some labels
        let start_label = make_label("for_start");
        let cont_label = continue_label(id.clone());
        let br_label = break_label(id);
        let mut for_init_instructions = match init {
            ForInit::InitDecl(d) => self.emit_var_declaration(d),
            ForInit::InitExp(e) => match e.map(|e| self.emit_tacky_for_exp(e)) {
                Some((instrs, _)) => instrs,
                None => vec![],
            },
        };
        let test_condition = match condition.map(|e| self.emit_tacky_for_exp(e)) {
            Some((instrs, v)) => instrs
                .into_iter()
                .chain(once(Instruction::JumpIfZero(v, br_label.clone())))
                .collect(),
            None => vec![],
        };
        let post_instructions = match post.map(|e| self.emit_tacky_for_exp(e)) {
            Some((instrs, _post_result)) => instrs,
            None => vec![],
        };
        for_init_instructions.push(Instruction::Label(start_label.clone()));
        for_init_instructions.extend(test_condition);
        for_init_instructions.extend(self.emit_tacky_for_statement(body));
        for_init_instructions.push(Instruction::Label(cont_label));
        for_init_instructions.extend(post_instructions);
        for_init_instructions.extend(vec![
            Instruction::Jump(start_label),
            Instruction::Label(br_label),
        ]);
        for_init_instructions
    }

    fn emit_tacky_for_switch(
        &mut self,
        condition: TypedExp,
        body: Statement<TypedExp>,
        cases: SwitchCases,
        id: String,
    ) -> Vec<Instruction> {
        let mut instructions = vec![];
        let br_label = break_label(id.clone());
        let (eval_condition, c) = self.emit_tacky_for_exp(condition.clone());
        instructions.extend(eval_condition);

        for case in &cases {
            if let Some(value) = case {
                let temp_var_name = self.create_tmp(condition.get_type());
                let temp_var = TackyVal::Var(temp_var_name);
                let src2 = match condition.get_type() {
                    Type::Int => TackyVal::Constant(T::ConstInt(*value as i32)),
                    Type::Long => TackyVal::Constant(T::ConstLong(*value)),
                    Type::UInt => TackyVal::Constant(T::ConstUInt(*value as u32)),
                    Type::ULong => TackyVal::Constant(T::ConstULong(*value as u64)),
                    _ => panic!("switch condition should be int or long"),
                };
                instructions.push(Instruction::Binary {
                    op: BinaryOperator::Equal,
                    src1: c.clone(),
                    src2: src2,
                    dst: temp_var.clone(),
                });
                instructions.push(Instruction::JumpIfNotZero(
                    temp_var,
                    case_label(*value, id.clone()),
                ))
            }
        }

        if cases.contains(&None) {
            instructions.push(Instruction::Jump(default_label(id.clone())));
        } else {
            instructions.push(Instruction::Jump(br_label.clone()));
        }

        instructions.extend(self.emit_tacky_for_statement(body));
        instructions.push(Instruction::Label(br_label));

        instructions
    }

    fn emit_fun_declaration(&mut self, fun_decl: Declaration<TypedExp>) -> Option<TopLevel> {
        match fun_decl {
            Declaration::FunDecl(AstFunction {
                name,
                params,
                body: Some(block_items),
                ..
            }) => {
                let global = self.symbol_table.is_global(name.as_str());
                let mut body_instructions: Vec<Instruction> = block_items
                    .0
                    .into_iter()
                    .flat_map(|item| self.emit_tacky_for_block_item(item))
                    .collect();
                let extra_return = Instruction::Return(TackyVal::Constant(INT_ZERO));
                body_instructions.push(extra_return);
                Some(TopLevel::FunctionDefinition {
                    name,
                    global,
                    params,
                    body: body_instructions,
                })
            }
            _ => None,
        }
    }

    fn convert_symbols_to_tacky(&mut self) -> Vec<TopLevel> {
        self.symbol_table
            .bindings()
            .iter()
            .filter_map(|(name, entry)| match &entry.attrs {
                IdentifierAttrs::StaticAttr { init, global } => match init {
                    InitialValue::Initial(i) => Some(TopLevel::StaticVariable {
                        name: name.clone(),
                        t: entry.t.clone(),
                        global: *global,
                        init: *i,
                    }),
                    InitialValue::Tentative => Some(TopLevel::StaticVariable {
                        name: name.clone(),
                        t: entry.t.clone(),
                        global: *global,
                        init: zero(&entry.t),
                    }),
                    InitialValue::NoInitializer => None,
                },
                _ => None,
            })
            .collect()
    }

    pub fn generate(&mut self, ast: AstProgram<TypedExp>) -> Program {
        let AstProgram(fn_defs) = ast;
        let tacky_fn_defs: Vec<TopLevel> = fn_defs
            .into_iter()
            .filter_map(|fun_decl| self.emit_fun_declaration(fun_decl))
            .collect();
        let tacky_var_defs = self.convert_symbols_to_tacky();
        Program {
            top_levels: tacky_fn_defs.into_iter().chain(tacky_var_defs).collect(),
        }
    }
}
