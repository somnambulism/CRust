use super::{
    ast::{
        Exp, FunctionDefinition as AstFunction, Program as AstProgram, Statement,
        UnaryOperator as AstUnaryOperator,
    },
    tacky::{FunctionDefinition, Instruction, Program, TackyVal, UnaryOperator},
    unique_ids::make_temporary,
};

fn convert_op(op: AstUnaryOperator) -> UnaryOperator {
    match op {
        AstUnaryOperator::Complement => UnaryOperator::Complement,
        AstUnaryOperator::Negate => UnaryOperator::Negate,
    }
}

fn emit_tacky_for_exp(exp: Exp) -> (Vec<Instruction>, TackyVal) {
    match exp {
        Exp::Constant(c) => (vec![], TackyVal::Constant(c)),
        Exp::Unary(op, inner) => {
            let (mut eval_inner, v) = emit_tacky_for_exp(*inner);
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
    }
}

fn emit_taky_for_statement(stmt: Statement) -> Vec<Instruction> {
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
        body: emit_taky_for_statement(func.body),
    }
}

pub fn generate(ast: AstProgram) -> Program {
    Program {
        function: emit_tacky_for_function(ast.function),
    }
}
