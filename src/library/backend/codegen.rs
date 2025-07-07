use std::vec;

use crate::library::{
    assembly::{
        BinaryOperator, CondCode, Instruction, Operand, Program as AssemblyProgram, Reg,
        TopLevel as AssemblyTopLevel, UnaryOperator,
    },
    tacky::{
        BinaryOperator as TackyBinaryOp, Instruction as TackyInstruction, Program, TackyVal,
        TopLevel, UnaryOperator as TackyUnaryOp,
    },
};

const PARAM_PASSING_REGS: [Reg; 4] = [Reg::CX, Reg::DX, Reg::R8, Reg::R9];

fn convert_val(val: &TackyVal) -> Operand {
    match val {
        TackyVal::Constant(i) => Operand::Imm(*i),
        TackyVal::Var(v) => Operand::Pseudo(v.to_string()),
    }
}

fn convert_unop(op: &TackyUnaryOp) -> UnaryOperator {
    match op {
        TackyUnaryOp::Complement => UnaryOperator::Not,
        TackyUnaryOp::Negate => UnaryOperator::Neg,
        TackyUnaryOp::Not => {
            panic!("Internal error, can't convert TACKY 'not' directly to assembly")
        }
    }
}

fn convert_binop(op: &TackyBinaryOp) -> BinaryOperator {
    match op {
        TackyBinaryOp::Add => BinaryOperator::Add,
        TackyBinaryOp::Subtract => BinaryOperator::Sub,
        TackyBinaryOp::Multiply => BinaryOperator::Mult,
        TackyBinaryOp::BitwiseAnd => BinaryOperator::And,
        TackyBinaryOp::BitwiseOr => BinaryOperator::Or,
        TackyBinaryOp::Xor => BinaryOperator::Xor,
        TackyBinaryOp::LeftShift => BinaryOperator::Sal,
        TackyBinaryOp::RightShift => BinaryOperator::Sar,
        TackyBinaryOp::Divide
        | TackyBinaryOp::Mod
        | TackyBinaryOp::Equal
        | TackyBinaryOp::NotEqual
        | TackyBinaryOp::GreaterOrEqual
        | TackyBinaryOp::LessOrEqual
        | TackyBinaryOp::GreaterThan
        | TackyBinaryOp::LessThan => {
            panic!("Internal error: not a binary assembly instruction")
        }
    }
}

fn convert_cond_code(cond_code: &TackyBinaryOp) -> CondCode {
    match cond_code {
        TackyBinaryOp::Equal => CondCode::E,
        TackyBinaryOp::NotEqual => CondCode::NE,
        TackyBinaryOp::GreaterThan => CondCode::G,
        TackyBinaryOp::GreaterOrEqual => CondCode::GE,
        TackyBinaryOp::LessThan => CondCode::L,
        TackyBinaryOp::LessOrEqual => CondCode::LE,
        _ => panic!("Internal error: not a condition code"),
    }
}

fn convert_function_call(f: &str, args: &Vec<TackyVal>, dst: &TackyVal) -> Vec<Instruction> {
    let split_idx = args.len().min(4);
    let (reg_args, stack_args) = args.split_at(split_idx);
    // adjust stack alignment

    let stack_padding: usize = if stack_args.len() % 2 == 0 { 0 } else { 8 };

    let mut instructions = if stack_padding == 0 {
        vec![]
    } else {
        vec![Instruction::AllocateStack(
            stack_padding.try_into().unwrap(),
        )]
    };

    // pass args in registers
    instructions.extend(reg_args.iter().enumerate().map(|(idx, arg)| {
        let r = &PARAM_PASSING_REGS[idx];
        let assembly_arg = convert_val(arg);
        Instruction::Mov(assembly_arg, Operand::Reg(r.clone()))
    }));

    // pass args on the stack
    for arg in stack_args.iter().rev() {
        let assembly_arg = convert_val(arg);
        match assembly_arg {
            Operand::Imm(_) | Operand::Reg(_) => instructions.push(Instruction::Push(assembly_arg)),
            _ => {
                // copy into a register before pushing
                instructions.extend(vec![
                    Instruction::Mov(assembly_arg, Operand::Reg(Reg::AX)),
                    Instruction::Push(Operand::Reg(Reg::AX)),
                ]);
            }
        }
    }

    // allocate shadow space for the stack
    instructions.push(Instruction::AllocateStack(32));

    // call the function
    instructions.push(Instruction::Call(f.to_string()));

    // adjust stack pointer (32 bytes for shadow space + 8 bytes for each stack argument)
    let bytes_to_remove = 32 + (8 * stack_args.len()) + stack_padding;
    instructions.push(Instruction::DeallocateStack(bytes_to_remove));

    // retrieve return value
    let assembly_dst = convert_val(&dst);
    instructions.push(Instruction::Mov(Operand::Reg(Reg::AX), assembly_dst));

    instructions
}

fn convert_instruction(instruction: &TackyInstruction) -> Vec<Instruction> {
    match instruction {
        TackyInstruction::Copy { src, dst } => {
            let asm_src = convert_val(&src);
            let asm_dst = convert_val(&dst);
            vec![Instruction::Mov(asm_src, asm_dst)]
        }
        TackyInstruction::Return(tacky_val) => {
            let asm_val = convert_val(&tacky_val);
            vec![
                Instruction::Mov(asm_val, Operand::Reg(Reg::AX)),
                Instruction::Ret,
            ]
        }
        TackyInstruction::Unary {
            op: TackyUnaryOp::Not,
            src,
            dst,
        } => {
            let asm_src = convert_val(&src);
            let asm_dst = convert_val(&dst);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_src),
                Instruction::Mov(Operand::Imm(0), asm_dst.clone()),
                Instruction::SetCC(CondCode::E, asm_dst),
            ]
        }
        TackyInstruction::Unary { op, src, dst } => {
            let asm_op = convert_unop(&op);
            let asm_src = convert_val(&src);
            let asm_dst = convert_val(&dst);
            vec![
                Instruction::Mov(asm_src, asm_dst.clone()),
                Instruction::Unary(asm_op, asm_dst),
            ]
        }
        TackyInstruction::Binary {
            op,
            src1,
            src2,
            dst,
        } => {
            let asm_src1 = convert_val(&src1);
            let asm_src2 = convert_val(&src2);
            let asm_dst = convert_val(&dst);
            match op {
                // Relational operator
                TackyBinaryOp::Equal
                | TackyBinaryOp::NotEqual
                | TackyBinaryOp::GreaterThan
                | TackyBinaryOp::GreaterOrEqual
                | TackyBinaryOp::LessThan
                | TackyBinaryOp::LessOrEqual => {
                    let cond_code = convert_cond_code(&op);
                    vec![
                        Instruction::Cmp(asm_src2, asm_src1),
                        Instruction::Mov(Operand::Imm(0), asm_dst.clone()),
                        Instruction::SetCC(cond_code, asm_dst),
                    ]
                }
                // Division/modulo
                TackyBinaryOp::Divide | TackyBinaryOp::Mod => {
                    let result_reg = if op == &TackyBinaryOp::Divide {
                        Reg::AX
                    } else {
                        Reg::DX
                    };
                    vec![
                        Instruction::Mov(asm_src1, Operand::Reg(Reg::AX)),
                        Instruction::Cdq,
                        Instruction::Idiv(asm_src2),
                        Instruction::Mov(Operand::Reg(result_reg), asm_dst),
                    ]
                }
                // Bitwise shift
                TackyBinaryOp::LeftShift | TackyBinaryOp::RightShift => {
                    vec![
                        // Instruction::Mov(asm_src2, Operand::Reg(Reg::CX)),
                        Instruction::Mov(asm_src1, asm_dst.clone()),
                        Instruction::Binary {
                            op: convert_binop(&op),
                            src: asm_src2,
                            dst: asm_dst,
                        },
                    ]
                }
                // Addition/subtraction/multiplication
                _ => {
                    let asm_op = convert_binop(&op);
                    vec![
                        Instruction::Mov(asm_src1, asm_dst.clone()),
                        Instruction::Binary {
                            op: asm_op,
                            src: asm_src2,
                            dst: asm_dst,
                        },
                    ]
                }
            }
        }
        TackyInstruction::Jump(target) => vec![Instruction::Jmp(target.to_string())],
        TackyInstruction::JumpIfZero(cond, target) => {
            let asm_cond = convert_val(&cond);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_cond),
                Instruction::JmpCC(CondCode::E, target.to_string()),
            ]
        }
        TackyInstruction::JumpIfNotZero(cond, target) => {
            let asm_cond = convert_val(&cond);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_cond),
                Instruction::JmpCC(CondCode::NE, target.to_string()),
            ]
        }
        TackyInstruction::Label(l) => vec![Instruction::Label(l.to_string())],
        TackyInstruction::FunCall { f, args, dst } => convert_function_call(&f, &args, &dst),
    }
}

fn pass_params(param_list: &Vec<String>) -> Vec<Instruction> {
    let split_idx = param_list.len().min(4);
    let (register_params, stack_params) = param_list.split_at(split_idx);

    let mut instructions: Vec<Instruction> = vec![];

    // Windows x64 ABI: shadow space (32 bytes) is always reserved
    // Move register arguments to stack slots
    for (idx, param) in register_params.iter().enumerate() {
        let reg = &PARAM_PASSING_REGS[idx];
        // Windows: first arg at 16(%rbp), then 24, 32, 40
        let offset = 16 + (idx as isize) * 8;
        let stk = Operand::Stack(offset);
        instructions.push(Instruction::Mov(Operand::Reg(reg.clone()), stk.clone()));
        // Move from stack slot to pseudo for IR
        instructions.push(Instruction::Mov(stk, Operand::Pseudo(param.to_string())));
    }
    // Stack arguments: 5th arg at 48(%rbp), 6th at 56(%rbp), ...
    for (idx, param) in stack_params.iter().enumerate() {
        let offset = 16 + ((register_params.len() + idx) as isize) * 8;
        let stk = Operand::Stack(offset);
        instructions.push(Instruction::Mov(stk, Operand::Pseudo(param.to_string())));
    }

    instructions
}

fn convert_top_level(top_level: &TopLevel) -> AssemblyTopLevel {
    match top_level {
        TopLevel::FunctionDefinition {
            name,
            global,
            params,
            body,
        } => {
            let mut instructions = pass_params(&params);
            for instruction in body {
                instructions.extend(convert_instruction(instruction));
            }

            AssemblyTopLevel::Function {
                name: name.to_string(),
                global: *global,
                instructions,
            }
        }
        TopLevel::StaticVariable { name, global, init } => AssemblyTopLevel::StaticVariable {
            name: name.to_string(),
            global: *global,
            init: *init,
        },
    }
}

pub fn generate(program: &Program) -> AssemblyProgram {
    let Program { top_levels } = program;

    AssemblyProgram {
        top_levels: top_levels.into_iter().map(convert_top_level).collect(),
    }
}
