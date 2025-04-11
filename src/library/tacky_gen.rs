use super::{
    ast::{
        BinaryOperator as AstBinaryOperator, BlockItem, CompoundAssignOperator, Declaration, Exp,
        FunctionDefinition as AstFunction, Program as AstProgram, Statement,
        UnaryOperator as AstUnaryOperator,
    },
    tacky::{BinaryOperator, FunctionDefinition, Instruction, Program, TackyVal, UnaryOperator},
    unique_ids::{make_label, make_temporary},
};

fn convert_op(op: AstUnaryOperator) -> UnaryOperator {
    match op {
        AstUnaryOperator::Complement => UnaryOperator::Complement,
        AstUnaryOperator::Negate => UnaryOperator::Negate,
        AstUnaryOperator::Not => UnaryOperator::Not,
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
        AstBinaryOperator::Xor => BinaryOperator::Xor,
        AstBinaryOperator::LeftShift => BinaryOperator::LeftShift,
        AstBinaryOperator::RightShift => BinaryOperator::RightShift,
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

fn convert_compound_assignment_op(op: CompoundAssignOperator) -> BinaryOperator {
    match op {
        CompoundAssignOperator::PlusEqual => BinaryOperator::Add,
        CompoundAssignOperator::MinusEqual => BinaryOperator::Subtract,
        CompoundAssignOperator::StarEqual => BinaryOperator::Multiply,
        CompoundAssignOperator::SlashEqual => BinaryOperator::Divide,
        CompoundAssignOperator::PercentEqual => BinaryOperator::Mod,
        CompoundAssignOperator::AmpersandEqual => BinaryOperator::BitwiseAnd,
        CompoundAssignOperator::PipeEqual => BinaryOperator::BitwiseOr,
        CompoundAssignOperator::CaretEqual => BinaryOperator::Xor,
        CompoundAssignOperator::LeftShiftEqual => BinaryOperator::LeftShift,
        CompoundAssignOperator::RightShiftEqual => BinaryOperator::RightShift,
    }
}

fn emit_tacky_for_exp(exp: Exp) -> (Vec<Instruction>, TackyVal) {
    match exp {
        Exp::Constant(c) => (vec![], TackyVal::Constant(c)),
        Exp::Var(v) => (vec![], TackyVal::Var(v)),
        Exp::Unary(op, inner) => emit_unary_expression(op, *inner),
        Exp::Binary(AstBinaryOperator::And, e1, e2) => emit_and_expression(*e1, *e2),
        Exp::Binary(AstBinaryOperator::Or, e1, e2) => emit_or_expression(*e1, *e2),
        Exp::Binary(op, e1, e2) => emit_binary_expression(op, *e1, *e2),
        Exp::Assignment(lhs, rhs) => {
            if let Exp::Var(v) = *lhs {
                let (mut instructions, rhs_result) = emit_tacky_for_exp(*rhs);
                instructions.push(Instruction::Copy {
                    src: rhs_result,
                    dst: TackyVal::Var(v.clone()),
                });
                (instructions, TackyVal::Var(v))
            } else {
                panic!("Internal error: bad lvalue")
            }
        }
        Exp::CompoundAssign(op, lhs, rhs) => {
            if let Exp::Var(v) = *lhs {
                let (mut instructions, rhs_result) = emit_tacky_for_exp(*rhs);
                let tacky_op = convert_compound_assignment_op(op);
                let dst = TackyVal::Var(v.clone());
                instructions.push(Instruction::Binary {
                    op: tacky_op,
                    src1: TackyVal::Var(v.clone()),
                    src2: rhs_result,
                    dst: dst.clone(),
                });
                (instructions, TackyVal::Var(v))
            } else {
                panic!("Internal error: bad lvalue")
            }
        }
        Exp::PrefixIncrement(inner) => emit_inc_dec(*inner, true, false),
        Exp::PrefixDecrement(inner) => emit_inc_dec(*inner, false, false),
        Exp::PostfixIncrement(inner) => emit_inc_dec(*inner, true, true),
        Exp::PostfixDecrement(inner) => emit_inc_dec(*inner, false, true),
    }
}

fn emit_unary_expression(op: AstUnaryOperator, inner: Exp) -> (Vec<Instruction>, TackyVal) {
    let (mut eval_inner, v) = emit_tacky_for_exp(inner);
    let dst_name = make_temporary();
    let dst = TackyVal::Var(dst_name);
    let tacky_op = convert_op(op);
    eval_inner.push(Instruction::Unary {
        op: tacky_op,
        src: v,
        dst: dst.clone(),
    });
    (eval_inner, dst)
}

fn emit_binary_expression(op: AstBinaryOperator, e1: Exp, e2: Exp) -> (Vec<Instruction>, TackyVal) {
    let (eval_v1, v1) = emit_tacky_for_exp(e1);
    let (eval_v2, v2) = emit_tacky_for_exp(e2);
    let dst_name = make_temporary();
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

fn emit_and_expression(e1: Exp, e2: Exp) -> (Vec<Instruction>, TackyVal) {
    let (eval_v1, v1) = emit_tacky_for_exp(e1);
    let (eval_v2, v2) = emit_tacky_for_exp(e2);
    let false_label = make_label("and_false");
    let end_label = make_label("and_end");
    let dst_name = make_temporary();
    let dst = TackyVal::Var(dst_name);
    let mut instructions = eval_v1;
    instructions.push(Instruction::JumpIfZero(v1, false_label.clone()));
    instructions.extend(eval_v2);
    instructions.extend(vec![
        Instruction::JumpIfZero(v2, false_label.clone()),
        Instruction::Copy {
            src: TackyVal::Constant(1),
            dst: dst.clone(),
        },
        Instruction::Jump(end_label.clone()),
        Instruction::Label(false_label),
        Instruction::Copy {
            src: TackyVal::Constant(0),
            dst: dst.clone(),
        },
        Instruction::Label(end_label),
    ]);
    (instructions, dst)
}

fn emit_or_expression(e1: Exp, e2: Exp) -> (Vec<Instruction>, TackyVal) {
    let (eval_v1, v1) = emit_tacky_for_exp(e1);
    let (eval_v2, v2) = emit_tacky_for_exp(e2);
    let true_label = make_label("or_true");
    let end_label = make_label("or_end");
    let dst_name = make_temporary();
    let dst = TackyVal::Var(dst_name);
    let mut instructions = eval_v1;
    instructions.push(Instruction::JumpIfNotZero(v1, true_label.clone()));
    instructions.extend(eval_v2);
    instructions.extend(vec![
        Instruction::JumpIfNotZero(v2, true_label.clone()),
        Instruction::Copy {
            src: TackyVal::Constant(0),
            dst: dst.clone(),
        },
        Instruction::Jump(end_label.clone()),
        Instruction::Label(true_label),
        Instruction::Copy {
            src: TackyVal::Constant(1),
            dst: dst.clone(),
        },
        Instruction::Label(end_label),
    ]);
    (instructions, dst)
}

fn emit_inc_dec(inner: Exp, is_inc: bool, is_post: bool) -> (Vec<Instruction>, TackyVal) {
    if let Exp::Var(v) = inner {
        let dst = TackyVal::Var(v.clone());
        let tmp = TackyVal::Var(make_temporary());
        let op = if is_inc {
            BinaryOperator::Add
        } else {
            BinaryOperator::Subtract
        };

        let one = TackyVal::Constant(1);

        let mut instuctions = vec![];

        if is_post {
            // save the original value in a temporary variable
            instuctions.push(Instruction::Copy {
                src: dst.clone(),
                dst: tmp.clone(),
            });
        }

        instuctions.push(Instruction::Binary {
            op: op,
            src1: dst.clone(),
            src2: one,
            dst: dst.clone(),
        });

        let result = if is_post { tmp } else { dst };
        (instuctions, result)
    } else {
        panic!("Internal error: ++/-- can only be applied to variables");
    }
}

fn emit_tacky_for_statement(stmt: Statement) -> Vec<Instruction> {
    match stmt {
        Statement::Return(e) => {
            let (mut eval_exp, v) = emit_tacky_for_exp(e);
            eval_exp.push(Instruction::Return(v));
            eval_exp
        }
        Statement::Expression(e) => {
            // evaluate expression but ignore the result
            let (eval_exp, _exp_result) = emit_tacky_for_exp(e);
            eval_exp
        }
        Statement::Null => vec![],
    }
}

fn emit_tacky_for_block_item(item: BlockItem) -> Vec<Instruction> {
    match item {
        BlockItem::S(s) => emit_tacky_for_statement(s),
        BlockItem::D(Declaration {
            name,
            init: Some(e),
        }) => {
            // treat declaration with initializer like an assignment expression
            let (eval_assignment, _assignment_result) =
                emit_tacky_for_exp(Exp::Assignment(Exp::Var(name).into(), e.into()));
            eval_assignment
        }
        BlockItem::D(Declaration {
            name: _,
            init: None,
        }) => {
            // don't generate instructions for declaration without initializer
            vec![]
        }
    }
}

fn emit_tacky_for_function(AstFunction { name, body }: AstFunction) -> FunctionDefinition {
    let body_instructions: Vec<Instruction> = body
        .into_iter()
        .flat_map(emit_tacky_for_block_item)
        .collect();
    let extra_return = Instruction::Return(TackyVal::Constant(0));
    FunctionDefinition {
        name,
        body: body_instructions
            .into_iter()
            .chain(std::iter::once(extra_return))
            .collect(),
    }
}

pub fn generate(ast: AstProgram) -> Program {
    Program {
        function: emit_tacky_for_function(ast.function),
    }
}
