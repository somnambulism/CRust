use std::iter::once;

use super::{
    ast::{
        BinaryOperator as AstBinaryOperator, Block, BlockItem, CompoundAssignOperator, Declaration,
        Exp, ForInit, FunctionDeclaration as AstFunction, Program as AstProgram, Statement,
        SwitchCases, UnaryOperator as AstUnaryOperator, VariableDeclaration,
    },
    tacky::{BinaryOperator, FunctionDefinition, Instruction, Program, TackyVal, UnaryOperator},
    util::unique_ids::{make_label, make_temporary},
};

fn break_label(label: String) -> String {
    format!("break.{}", label)
}

fn continue_label(label: String) -> String {
    format!("continue.{}", label)
}

fn case_label(condition: i32, switch_label: String) -> String {
    format!("switch.{}.case.{}", switch_label, condition)
}

fn default_label(switch_label: String) -> String {
    format!("switch.{}.default", switch_label)
}

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
        Exp::Conditional {
            condition,
            then_result,
            else_result,
        } => emit_conditional_expression(*condition, *then_result, *else_result),
        Exp::FunCall { f, args } => emit_fun_call(f.as_str(), args),
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

        let mut instructions = vec![];

        if is_post {
            // save the original value in a temporary variable
            instructions.push(Instruction::Copy {
                src: dst.clone(),
                dst: tmp.clone(),
            });
        }

        instructions.push(Instruction::Binary {
            op: op,
            src1: dst.clone(),
            src2: one,
            dst: dst.clone(),
        });

        let result = if is_post { tmp } else { dst };
        (instructions, result)
    } else {
        panic!("Internal error: ++/-- can only be applied to variables");
    }
}

fn emit_conditional_expression(condition: Exp, e1: Exp, e2: Exp) -> (Vec<Instruction>, TackyVal) {
    let (eval_cond, c) = emit_tacky_for_exp(condition);
    let (eval_v1, v1) = emit_tacky_for_exp(e1);
    let (eval_v2, v2) = emit_tacky_for_exp(e2);
    let e2_label = make_label("conditional_else");
    let end_label = make_label("conditional_end");
    let dst_name = make_temporary();
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

fn emit_fun_call(f: &str, args: Vec<Exp>) -> (Vec<Instruction>, TackyVal) {
    let dst_name = make_temporary();
    let dst = TackyVal::Var(dst_name);
    let (arg_instructions, arg_vals): (Vec<Vec<Instruction>>, Vec<TackyVal>) =
        args.into_iter().map(|arg| emit_tacky_for_exp(arg)).unzip();
    let mut instructions: Vec<Instruction> = arg_instructions.into_iter().flatten().collect();
    instructions.push(Instruction::FunCall {
        f: f.to_string(),
        args: arg_vals,
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
        Statement::Expression(e) => {
            // evaluate expression but ignore the result
            let (eval_exp, _exp_result) = emit_tacky_for_exp(e);
            eval_exp
        }
        Statement::If {
            condition,
            then_clause,
            else_clause,
        } => emit_tacky_for_if_statement(condition, then_clause, else_clause),
        Statement::Compound(block) => {
            let Block(items) = block;
            items
                .into_iter()
                .flat_map(emit_tacky_for_block_item)
                .collect()
        }
        Statement::Break(id) => vec![Instruction::Jump(break_label(id))],
        Statement::Continue(id) => vec![Instruction::Jump(continue_label(id))],
        Statement::DoWhile {
            body,
            condition,
            id,
        } => emit_tacky_for_do_loop(*body, condition, id),
        Statement::While {
            condition,
            body,
            id,
        } => emit_tacky_for_while_loop(condition, *body, id),
        Statement::For {
            init,
            condition,
            post,
            body,
            id,
        } => emit_tacky_for_for_loop(init, condition, post, *body, id),
        Statement::Switch {
            condition,
            body,
            cases,
            id,
        } => emit_tacky_for_switch(condition, *body, cases, id),
        Statement::Case {
            condition,
            body,
            switch_label,
        } => {
            let mut instructions = emit_tacky_for_statement(*body);
            instructions.insert(0, Instruction::Label(case_label(condition, switch_label)));
            instructions
        }
        Statement::Default { body, switch_label } => {
            let mut instructions = emit_tacky_for_statement(*body);
            instructions.insert(0, Instruction::Label(default_label(switch_label)));
            instructions
        }
        Statement::Null => vec![],
        Statement::Labelled { label, statement } => {
            let mut instructions = emit_tacky_for_statement(*statement);
            instructions.insert(0, Instruction::Label(label));
            instructions
        }
        Statement::Goto(label) => {
            vec![Instruction::Jump(label)]
        }
    }
}

fn emit_tacky_for_block_item(item: BlockItem) -> Vec<Instruction> {
    match item {
        BlockItem::S(s) => emit_tacky_for_statement(s),
        BlockItem::D(d) => emit_local_declaration(d),
    }
}

fn emit_local_declaration(d: Declaration) -> Vec<Instruction> {
    match d {
        Declaration::VarDecl(vd) => emit_var_declaration(vd),
        Declaration::FunDecl(_) => vec![],
    }
}

fn emit_var_declaration(d: VariableDeclaration) -> Vec<Instruction> {
    match d {
        VariableDeclaration {
            name,
            init: Some(e),
        } => {
            // treat declaration with initializer like an assignment expression
            let (eval_assignment, _assignment_result) =
                emit_tacky_for_exp(Exp::Assignment(Exp::Var(name).into(), e.into()));
            eval_assignment
        }
        VariableDeclaration {
            name: _,
            init: None,
        } => {
            // don't generate instructions for declaration without initializer
            vec![]
        }
    }
}

fn emit_tacky_for_if_statement(
    condition: Exp,
    then_clause: Box<Statement>,
    else_clause: Option<Box<Statement>>,
) -> Vec<Instruction> {
    if let None = else_clause {
        // no else clause
        let end_label = make_label("if_end");
        let (mut eval_condition, c) = emit_tacky_for_exp(condition);
        eval_condition.push(Instruction::JumpIfZero(c, end_label.clone()));
        eval_condition.extend(emit_tacky_for_statement(*then_clause));
        eval_condition.push(Instruction::Label(end_label));
        eval_condition
    } else {
        let else_label = make_label("else");
        let end_label = make_label("if_end");
        let (mut eval_condition, c) = emit_tacky_for_exp(condition);
        eval_condition.push(Instruction::JumpIfZero(c, else_label.clone()));
        eval_condition.extend(emit_tacky_for_statement(*then_clause));
        eval_condition.extend(vec![
            Instruction::Jump(end_label.clone()),
            Instruction::Label(else_label),
        ]);
        eval_condition.extend(emit_tacky_for_statement(*else_clause.unwrap()));
        eval_condition.push(Instruction::Label(end_label));
        eval_condition
    }
}

fn emit_tacky_for_do_loop(body: Statement, condition: Exp, id: String) -> Vec<Instruction> {
    let start_label = make_label("do_loop_start");
    let cont_label = continue_label(id.clone());
    let br_label = break_label(id);
    let (eval_condition, c) = emit_tacky_for_exp(condition);
    let mut instructions = vec![Instruction::Label(start_label.clone())];
    instructions.extend(emit_tacky_for_statement(body));
    instructions.push(Instruction::Label(cont_label));
    instructions.extend(eval_condition);
    instructions.extend(vec![
        Instruction::JumpIfNotZero(c, start_label),
        Instruction::Label(br_label),
    ]);
    instructions
}

fn emit_tacky_for_while_loop(condition: Exp, body: Statement, id: String) -> Vec<Instruction> {
    let cont_label = continue_label(id.clone());
    let br_label = break_label(id);
    let (eval_condition, c) = emit_tacky_for_exp(condition);
    let mut instructions = vec![Instruction::Label(cont_label.clone())];
    instructions.extend(eval_condition);
    instructions.push(Instruction::JumpIfZero(c, br_label.clone()));
    instructions.extend(emit_tacky_for_statement(body));
    instructions.extend(vec![
        Instruction::Jump(cont_label),
        Instruction::Label(br_label),
    ]);
    instructions
}

fn emit_tacky_for_for_loop(
    init: ForInit,
    condition: Option<Exp>,
    post: Option<Exp>,
    body: Statement,
    id: String,
) -> Vec<Instruction> {
    // generate some labels
    let start_label = make_label("for_start");
    let cont_label = continue_label(id.clone());
    let br_label = break_label(id);
    let mut for_init_instructions = match init {
        ForInit::InitDecl(d) => emit_var_declaration(d),
        ForInit::InitExp(e) => match e.map(emit_tacky_for_exp) {
            Some((instrs, _)) => instrs,
            None => vec![],
        },
    };
    let test_condition = match condition.map(emit_tacky_for_exp) {
        Some((instrs, v)) => instrs
            .into_iter()
            .chain(once(Instruction::JumpIfZero(v, br_label.clone())))
            .collect(),
        None => vec![],
    };
    let post_instructions = match post.map(emit_tacky_for_exp) {
        Some((instrs, _post_result)) => instrs,
        None => vec![],
    };
    for_init_instructions.push(Instruction::Label(start_label.clone()));
    for_init_instructions.extend(test_condition);
    for_init_instructions.extend(emit_tacky_for_statement(body));
    for_init_instructions.push(Instruction::Label(cont_label));
    for_init_instructions.extend(post_instructions);
    for_init_instructions.extend(vec![
        Instruction::Jump(start_label),
        Instruction::Label(br_label),
    ]);
    for_init_instructions
}

fn emit_tacky_for_switch(
    condition: Exp,
    body: Statement,
    cases: SwitchCases,
    id: String,
) -> Vec<Instruction> {
    let mut instructions = vec![];
    let br_label = break_label(id.clone());
    let (eval_condition, c) = emit_tacky_for_exp(condition);
    instructions.extend(eval_condition);

    for case in &cases {
        if let Some(value) = case {
            let temp_var_name = make_temporary();
            let temp_var = TackyVal::Var(temp_var_name);
            instructions.push(Instruction::Binary {
                op: BinaryOperator::Equal,
                src1: c.clone(),
                src2: TackyVal::Constant(*value),
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

    instructions.extend(emit_tacky_for_statement(body));
    instructions.push(Instruction::Label(br_label));

    instructions
}

fn emit_function_declaration(
    AstFunction { name, params, body }: AstFunction,
) -> Option<FunctionDefinition> {
    match body {
        Some(Block(block_items)) => {
            let mut body_instructions: Vec<Instruction> = block_items
                .into_iter()
                .flat_map(emit_tacky_for_block_item)
                .collect();
            let extra_return = Instruction::Return(TackyVal::Constant(0));
            body_instructions.push(extra_return);
            Some(FunctionDefinition {
                name,
                params,
                body: body_instructions,
            })
        }
        None => None,
    }
}

pub fn generate(ast: AstProgram) -> Program {
    let AstProgram(fn_defs) = ast;
    let tacky_fn_defs = fn_defs
        .into_iter()
        .filter_map(emit_function_declaration)
        .collect();
    Program {
        functions: tacky_fn_defs,
    }
}
