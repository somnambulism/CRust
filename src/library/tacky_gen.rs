use super::{
    ast::{
        BinaryOperator as AstBinaryOperator, Exp, FunctionDefinition as AstFunction,
        Program as AstProgram, Statement, UnaryOperator as AstUnaryOperator,
    },
    tacky::{BinaryOperator, FunctionDefinition, Instruction, Program, TackyVal, UnaryOperator},
    unique_ids::make_temporary,
};

fn convert_op(op: AstUnaryOperator) -> UnaryOperator {
    match op {
        AstUnaryOperator::Complement => UnaryOperator::Complement,
        AstUnaryOperator::Negate => UnaryOperator::Negate,
    }
}

fn convert_binop(op: AstBinaryOperator) -> BinaryOperator {
    match op {
        AstBinaryOperator::Add => BinaryOperator::Add,
        AstBinaryOperator::Subtract => BinaryOperator::Subtract,
        AstBinaryOperator::Multiply => BinaryOperator::Multiply,
        AstBinaryOperator::Divide => BinaryOperator::Divide,
        AstBinaryOperator::Mod => BinaryOperator::Mod,
    }
}

fn emit_tacky_for_exp(exp: Exp) -> (Vec<Instruction>, TackyVal) {
    match exp {
        Exp::Constant(c) => (vec![], TackyVal::Constant(c)),
        Exp::Unary(op, inner) => emit_unary_expression(op, *inner),
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
