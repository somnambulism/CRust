use std::iter::once;

use crate::library::{
    ast::{
        block_items::{
            Block, BlockItem, Declaration, ForInit, FunctionDeclaration as AstFunction,
            Program as AstProgram, Statement, VariableDeclaration,
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

// An expression result that may or may not be lvalue converted
#[derive(Debug, PartialEq, Clone)]
enum ExpResult {
    PlainOperand(TackyVal),
    DereferencedPointer(TackyVal),
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

    // return list of instructions to evaluate expressions and resulting exp_result value as a pair
    fn emit_tacky_for_exp(&mut self, exp: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let TypedExp { e, t } = exp;
        match e {
            InnerExp::Constant(c) => (vec![], ExpResult::PlainOperand(TackyVal::Constant(c))),
            InnerExp::Var(v) => (vec![], ExpResult::PlainOperand(TackyVal::Var(v))),
            InnerExp::Unary(AstUnaryOperator::Incr, v) => {
                self.emit_compound_expression(AstBinaryOperator::Add, *v, mk_ast_const(&t, 1), t)
            }
            InnerExp::Unary(AstUnaryOperator::Decr, v) => self.emit_compound_expression(
                AstBinaryOperator::Subtract,
                *v,
                mk_ast_const(&t, 1),
                t,
            ),
            InnerExp::Cast { target_type, e } => self.emit_cast_expression(target_type, *e),
            InnerExp::Unary(op, inner) => self.emit_unary_expression(t, op, *inner),
            InnerExp::Binary(AstBinaryOperator::And, e1, e2) => self.emit_and_expression(*e1, *e2),
            InnerExp::Binary(AstBinaryOperator::Or, e1, e2) => self.emit_or_expression(*e1, *e2),
            InnerExp::Binary(op, e1, e2) => self.emit_binary_expression(t, op, *e1, *e2),
            InnerExp::Assignment(lhs, rhs) => self.emit_assignment(*lhs, *rhs),
            InnerExp::CompoundAssignment {
                op,
                lhs,
                rhs,
                result_t,
            } => self.emit_compound_expression(op, *lhs, *rhs, result_t),
            InnerExp::PostfixIncr(inner) => self.emit_postfix(AstBinaryOperator::Add, *inner),
            InnerExp::PostfixDecr(inner) => self.emit_postfix(AstBinaryOperator::Subtract, *inner),
            InnerExp::Conditional {
                condition,
                then_result,
                else_result,
            } => self.emit_conditional_expression(t, *condition, *then_result, *else_result),
            InnerExp::FunCall { f, args } => self.emit_fun_call(t, f.as_str(), args),
            InnerExp::Dereference(inner) => self.emit_dereference(*inner),
            InnerExp::AddrOf(inner) => self.emit_addr_of(&t, *inner),
        }
    }

    /* Helper functions for individual expressions */

    fn emit_unary_expression(
        &mut self,
        t: Type,
        op: AstUnaryOperator,
        inner: TypedExp,
    ) -> (Vec<Instruction>, ExpResult) {
        let (mut eval_inner, v) = self.emit_tacky_and_convert(inner);
        // define a temporary variable to hold result of this expression
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let tacky_op = convert_op(op);
        eval_inner.push(Instruction::Unary {
            op: tacky_op,
            src: v,
            dst: dst.clone(),
        });
        (eval_inner, ExpResult::PlainOperand(dst))
    }

    fn emit_postfix(
        &mut self,
        op: AstBinaryOperator,
        inner: TypedExp,
    ) -> (Vec<Instruction>, ExpResult) {
        // define var for result - i.e. value of lval BEFORE incr or decr
        let dst = TackyVal::Var(self.create_tmp(inner.clone().t));
        // evaluate inner to get exp_result
        let (mut instrs, lval) = self.emit_tacky_for_exp(inner.clone());
        let tacky_op = convert_binop(op);
        let one = TackyVal::Constant(mk_const(&inner.t, 1));

        // copy result to dst and perform incr or decr
        let oper_instrs = match lval {
            ExpResult::PlainOperand(TackyVal::Var(v)) => {
                /* dst = v
                 * v = v + 1 // or v - 1
                 */
                vec![
                    Instruction::Copy {
                        src: TackyVal::Var(v.clone()),
                        dst: dst.clone(),
                    },
                    Instruction::Binary {
                        op: tacky_op,
                        src1: TackyVal::Var(v.clone()),
                        src2: one,
                        dst: TackyVal::Var(v),
                    },
                ]
            }
            ExpResult::DereferencedPointer(p) => {
                /* dst = Load(p)
                 * tmp = dst + 1 // or dst - 1
                 * Store(tmp, p)
                 */
                let tmp = TackyVal::Var(self.create_tmp(inner.t));
                vec![
                    Instruction::Load {
                        src_ptr: p.clone(),
                        dst: dst.clone(),
                    },
                    Instruction::Binary {
                        op: tacky_op,
                        src1: dst.clone(),
                        src2: one,
                        dst: tmp.clone(),
                    },
                    Instruction::Store {
                        src: tmp,
                        dst_ptr: p,
                    },
                ]
            }
            ExpResult::PlainOperand(_) => {
                panic!("Invalid lvalue in postfix incr/decr: {:?}", inner)
            }
        };

        instrs.extend(oper_instrs);
        (instrs, ExpResult::PlainOperand(dst))
    }

    fn emit_cast_expression(
        &mut self,
        target_type: Type,
        inner: TypedExp,
    ) -> (Vec<Instruction>, ExpResult) {
        let (mut eval_inner, result) = self.emit_tacky_and_convert(inner.clone());
        let src_type = inner.get_type();

        if src_type == target_type {
            (eval_inner, ExpResult::PlainOperand(result))
        } else {
            let dst_name = self.create_tmp(target_type.clone());
            let dst = TackyVal::Var(dst_name);
            let cast_instruction =
                self.get_cast_instruction(result, dst.clone(), src_type, target_type);

            eval_inner.push(cast_instruction);
            (eval_inner, ExpResult::PlainOperand(dst))
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
    ) -> (Vec<Instruction>, ExpResult) {
        let (eval_v1, v1) = self.emit_tacky_and_convert(e1);
        let (eval_v2, v2) = self.emit_tacky_and_convert(e2);
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
        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_compound_expression(
        &mut self,
        op: AstBinaryOperator,
        lhs: TypedExp,
        rhs: TypedExp,
        result_t: Type,
    ) -> (Vec<Instruction>, ExpResult) {
        /*
         * if LHS is var with same type as result:
         *   lhs = lhs <op> rval
         * if LHS is a var with different type:
         *   tmp = cast(lhs)
         *   tmp = tmp <op> rval
         *   lhs = cast(tmp)
         * if LHS is pointer with same type:
         *   tmp = load(lhs_ptr)
         *   tmp = tmp <op> rval
         *   store(tmp, lhs_ptr)
         * if LHS is pointer with different type:
         *   tmp = load(lhs_ptr)
         *   tmp2 = cast(tmp)
         *   tmp2 = tmp2 <op> rval
         *   tmp = cast(tmp2)
         *   store(tmp, rhs_ptr)
         */
        let lhs_t = lhs.clone().t;
        // evaluate LHS
        let (eval_lhs, lhs) = self.emit_tacky_for_exp(lhs);
        // evaluate RHS - type checker already added conversion to common type if one is needed
        let (eval_rhs, rhs) = self.emit_tacky_and_convert(rhs);

        /* If LHS is a variable, we can update it directly. If it's a dereferenced pointer,
         * we need to load it into a temporary variable, operate on that, and then store it
         */
        let (dst, load_instr, store_instr) = match lhs {
            ExpResult::PlainOperand(dst) => (dst, vec![], vec![]),
            ExpResult::DereferencedPointer(p) => {
                let dst = TackyVal::Var(self.create_tmp(lhs_t.clone()));
                (
                    dst.clone(),
                    vec![Instruction::Load {
                        src_ptr: p.clone(),
                        dst: dst.clone(),
                    }],
                    vec![Instruction::Store {
                        src: dst,
                        dst_ptr: p,
                    }],
                )
            }
        };

        /* If LHS type and result type are the same, we can operate on dst directly. Otherwise
         * we need to cast dst to correct type before operation, then cast result back and assign
         * to dst.
         */
        let (result_var, cast_to, cast_from) = if lhs_t == result_t {
            (dst.clone(), vec![], vec![])
        } else {
            let tmp = TackyVal::Var(self.create_tmp(result_t.clone()));
            let cast_lhs_to_tmp = self.get_cast_instruction(
                dst.clone(),
                tmp.clone(),
                lhs_t.clone(),
                result_t.clone(),
            );
            let cast_tmp_to_lhs =
                self.get_cast_instruction(tmp.clone(), dst.clone(), result_t, lhs_t);
            (tmp, vec![cast_lhs_to_tmp], vec![cast_tmp_to_lhs])
        };

        let binary_instr = Instruction::Binary {
            op: convert_binop(op),
            src1: result_var.clone(),
            src2: rhs,
            dst: result_var,
        };

        let mut instructions = eval_lhs;
        instructions.extend(eval_rhs);
        instructions.extend(load_instr);
        instructions.extend(cast_to);
        instructions.push(binary_instr);
        instructions.extend(cast_from);
        instructions.extend(store_instr);

        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_and_expression(&mut self, e1: TypedExp, e2: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let (eval_v1, v1) = self.emit_tacky_and_convert(e1);
        let (eval_v2, v2) = self.emit_tacky_and_convert(e2);
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
        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_or_expression(&mut self, e1: TypedExp, e2: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let (eval_v1, v1) = self.emit_tacky_and_convert(e1);
        let (eval_v2, v2) = self.emit_tacky_and_convert(e2);
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
        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_assignment(&mut self, lhs: TypedExp, rhs: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let (mut instructions, lval) = self.emit_tacky_for_exp(lhs);
        let (rhs_instructions, rval) = self.emit_tacky_and_convert(rhs);
        instructions.extend(rhs_instructions);

        match &lval {
            ExpResult::PlainOperand(o) => {
                instructions.push(Instruction::Copy {
                    src: rval,
                    dst: o.clone(),
                });
                (instructions, lval)
            }
            ExpResult::DereferencedPointer(ptr) => {
                instructions.push(Instruction::Store {
                    src: rval.clone(),
                    dst_ptr: ptr.clone(),
                });
                (instructions, ExpResult::PlainOperand(rval))
            }
        }
    }

    fn emit_conditional_expression(
        &mut self,
        t: Type,
        condition: TypedExp,
        e1: TypedExp,
        e2: TypedExp,
    ) -> (Vec<Instruction>, ExpResult) {
        let (eval_cond, c) = self.emit_tacky_and_convert(condition);
        let (eval_v1, v1) = self.emit_tacky_and_convert(e1);
        let (eval_v2, v2) = self.emit_tacky_and_convert(e2);
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
        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_fun_call(
        &mut self,
        t: Type,
        f: &str,
        args: Vec<TypedExp>,
    ) -> (Vec<Instruction>, ExpResult) {
        let dst_name = self.create_tmp(t);
        let dst = TackyVal::Var(dst_name);
        let (arg_instructions, arg_vals): (Vec<Vec<Instruction>>, Vec<TackyVal>) = args
            .into_iter()
            .map(|arg| self.emit_tacky_and_convert(arg))
            .unzip();
        let mut instructions: Vec<Instruction> = arg_instructions.into_iter().flatten().collect();
        instructions.push(Instruction::FunCall {
            f: f.to_string(),
            args: arg_vals,
            dst: dst.clone(),
        });
        (instructions, ExpResult::PlainOperand(dst))
    }

    fn emit_dereference(&mut self, inner: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let (instructions, result) = self.emit_tacky_and_convert(inner);
        (instructions, ExpResult::DereferencedPointer(result))
    }

    fn emit_addr_of(&mut self, t: &Type, inner: TypedExp) -> (Vec<Instruction>, ExpResult) {
        let (mut instructions, result) = self.emit_tacky_for_exp(inner);
        match result {
            ExpResult::PlainOperand(o) => {
                let dst = TackyVal::Var(self.create_tmp(t.clone()));
                instructions.push(Instruction::GetAddress {
                    src: o,
                    dst: dst.clone(),
                });
                (instructions, ExpResult::PlainOperand(dst))
            }
            ExpResult::DereferencedPointer(ptr) => (instructions, ExpResult::PlainOperand(ptr)),
        }
    }

    fn emit_tacky_and_convert(&mut self, e: TypedExp) -> (Vec<Instruction>, TackyVal) {
        let (mut instructions, result) = self.emit_tacky_for_exp(e.clone());
        match result {
            ExpResult::PlainOperand(o) => (instructions, o),
            ExpResult::DereferencedPointer(ptr) => {
                let dst = TackyVal::Var(self.create_tmp(e.t));
                instructions.push(Instruction::Load {
                    src_ptr: ptr,
                    dst: dst.clone(),
                });
                (instructions, dst)
            }
        }
    }

    fn emit_tacky_for_statement(&mut self, stmt: Statement<TypedExp>) -> Vec<Instruction> {
        match stmt {
            Statement::Return(e) => {
                let (mut eval_exp, v) = self.emit_tacky_and_convert(e);
                eval_exp.push(Instruction::Return(v));
                eval_exp
            }
            Statement::Expression(e) => {
                // evaluate expression but ignore the result
                let (eval_exp, _) = self.emit_tacky_for_exp(e);
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
            Statement::Case(_, stmt, id) => {
                let mut instructions = self.emit_tacky_for_statement(*stmt);
                instructions.insert(0, Instruction::Label(id));
                instructions
            }
            Statement::Default(stmt, id) => {
                let mut instructions = self.emit_tacky_for_statement(*stmt);
                instructions.insert(0, Instruction::Label(id));
                instructions
            }
            Statement::Switch {
                control,
                body,
                id,
                cases,
            } => self.emit_tacky_for_switch(control, *body, id, cases),
            Statement::Null => vec![],
            Statement::LabelledStatement(lbl, stmt) => {
                let mut instructions = self.emit_tacky_for_statement(*stmt);
                instructions.insert(0, Instruction::Label(lbl));
                instructions
            }
            Statement::Goto(lbl) => {
                vec![Instruction::Jump(lbl)]
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
                var_type,
                ..
            } => {
                // treat declaration with initializer like an assignment expression
                let (eval_assignment, _assign_result) = self.emit_assignment(
                    TypedExp {
                        e: InnerExp::Var(name),
                        t: var_type,
                    },
                    e,
                );
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
            let (mut eval_condition, c) = self.emit_tacky_and_convert(condition);
            eval_condition.push(Instruction::JumpIfZero(c, end_label.clone()));
            eval_condition.extend(self.emit_tacky_for_statement(*then_clause));
            eval_condition.push(Instruction::Label(end_label));
            eval_condition
        } else {
            let else_label = make_label("else");
            let end_label = make_label("if_end");
            let (mut eval_condition, c) = self.emit_tacky_and_convert(condition);
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
        let (eval_condition, c) = self.emit_tacky_and_convert(condition);
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
        let (eval_condition, c) = self.emit_tacky_and_convert(condition);
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

    fn emit_tacky_for_switch(
        &mut self,
        control: TypedExp,
        body: Statement<TypedExp>,
        id: String,
        cases: Vec<(Option<T>, String)>,
    ) -> Vec<Instruction> {
        let br_label = break_label(id.clone());
        let (eval_control, c) = self.emit_tacky_and_convert(control.clone());
        let cmp_result = TackyVal::Var(self.create_tmp(control.get_type()));

        let emit_tacky_for_case = |key: &Option<T>, id: &String| -> Vec<Instruction> {
            match key {
                Some(i) => vec![
                    Instruction::Binary {
                        op: BinaryOperator::Equal,
                        src1: TackyVal::Constant(i.clone()),
                        src2: c.clone(),
                        dst: cmp_result.clone(),
                    },
                    Instruction::JumpIfNotZero(cmp_result.clone(), id.clone()),
                ],
                None => vec![],
            }
        };

        let jump_to_cases: Vec<Instruction> = cases
            .iter()
            .flat_map(|(key, id)| emit_tacky_for_case(key, id))
            .collect();

        let default_tacky =
            if let Some((_, default_id)) = cases.iter().find(|(key, _)| key.is_none()) {
                vec![Instruction::Jump(default_id.clone())]
            } else {
                vec![]
            };

        eval_control
            .into_iter()
            .chain(jump_to_cases)
            .chain(default_tacky)
            .chain(vec![Instruction::Jump(br_label.clone())])
            .chain(self.emit_tacky_for_statement(body))
            .chain(vec![Instruction::Label(br_label)])
            .collect()
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
        let test_condition = match condition.map(|e| self.emit_tacky_and_convert(e)) {
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
