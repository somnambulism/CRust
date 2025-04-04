use super::{
    ast::{
        BinaryOperator as AstBinaryOperator, Exp, FunctionDefinition as AstFunction,
        Program as AstProgram, Statement, UnaryOperator as AstUnaryOperator,
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

fn emit_tacky_for_exp(exp: Exp) -> (Vec<Instruction>, TackyVal) {
    match exp {
        Exp::Constant(c) => (vec![], TackyVal::Constant(c)),
        Exp::Unary(op, inner) => emit_unary_expression(op, *inner),
        Exp::Binary(AstBinaryOperator::And, e1, e2) => emit_and_expression(*e1, *e2),
        Exp::Binary(AstBinaryOperator::Or, e1, e2) => emit_or_expression(*e1, *e2),
        Exp::Binary(op, e1, e2) => emit_binary_expression(op, *e1, *e2),
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

fn emit_tacky_for_statement(stmt: Statement) -> Vec<Instruction> {
    match stmt {
        Statement::Return(e) => {
            let (mut eval_exp, v) = emit_tacky_for_exp(e);
            eval_exp.push(Instruction::Return(v));
            eval_exp
        }
    }
}

fn emit_tacky_for_function(func: AstFunction) -> FunctionDefinition {
    FunctionDefinition {
        name: func.name,
        body: emit_tacky_for_statement(func.body),
    }
}

pub fn generate(ast: AstProgram) -> Program {
    Program {
        function: emit_tacky_for_function(ast.function),
    }
}
