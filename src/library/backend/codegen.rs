use std::vec;

use crate::library::{
    assembly::{
        AsmType, BinaryOperator, CondCode, Instruction, Operand, Program as AssemblyProgram, Reg,
        TopLevel as AssemblyTopLevel, UnaryOperator,
    },
    backend::assembly_symbols::SymbolTable as BackendSymbolTable,
    r#const::{T, type_of_const},
    symbols::{Entry, IdentifierAttrs, SymbolTable},
    tacky::{
        BinaryOperator as TackyBinaryOp, Instruction as TackyInstruction, Program, TackyVal,
        TopLevel, UnaryOperator as TackyUnaryOp,
    },
    types::Type,
};

const PARAM_PASSING_REGS: [Reg; 4] = [Reg::CX, Reg::DX, Reg::R8, Reg::R9];
const ZERO: Operand = Operand::Imm(0);

pub struct CodeGen {
    pub symbol_table: SymbolTable,
    pub assembly_symbols: BackendSymbolTable,
}

fn convert_val(val: &TackyVal) -> Operand {
    match val {
        TackyVal::Constant(T::ConstInt(i)) => Operand::Imm(*i as i64),
        TackyVal::Constant(T::ConstLong(l)) => Operand::Imm(*l),
        TackyVal::Constant(T::ConstUInt(u)) => Operand::Imm(*u as i64),
        TackyVal::Constant(T::ConstULong(ul)) => Operand::Imm(*ul as i64),
        TackyVal::Var(v) => Operand::Pseudo(v.to_string()),
    }
}

fn convert_type(t: &Type) -> AsmType {
    match t {
        Type::Int | Type::UInt => AsmType::Longword,
        Type::Long | Type::ULong => AsmType::Quadword,
        Type::FunType { .. } => panic!("Internal error, converting function type to assembly"),
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

fn convert_cond_code(signed: bool, cond_code: &TackyBinaryOp) -> CondCode {
    match cond_code {
        TackyBinaryOp::Equal => CondCode::E,
        TackyBinaryOp::NotEqual => CondCode::NE,
        TackyBinaryOp::GreaterThan => {
            if signed {
                CondCode::G
            } else {
                CondCode::A
            }
        }
        TackyBinaryOp::GreaterOrEqual => {
            if signed {
                CondCode::GE
            } else {
                CondCode::AE
            }
        }
        TackyBinaryOp::LessThan => {
            if signed {
                CondCode::L
            } else {
                CondCode::B
            }
        }
        TackyBinaryOp::LessOrEqual => {
            if signed {
                CondCode::LE
            } else {
                CondCode::BE
            }
        }
        _ => panic!("Internal error: not a condition code"),
    }
}

impl CodeGen {
    pub fn new(symbols: SymbolTable) -> Self {
        CodeGen {
            symbol_table: symbols,
            assembly_symbols: BackendSymbolTable::new(),
        }
    }

    fn tacky_type(&self, tacky_val: &TackyVal) -> Type {
        match tacky_val {
            TackyVal::Constant(c) => type_of_const(c),
            TackyVal::Var(v) => self.symbol_table.get(v).t.clone(),
        }
    }

    fn asm_type(&self, v: &TackyVal) -> AsmType {
        convert_type(&self.tacky_type(v))
    }

    fn convert_function_call(
        &self,
        f: &str,
        args: &Vec<TackyVal>,
        dst: &TackyVal,
    ) -> Vec<Instruction> {
        let split_idx = args.len().min(4);
        let (reg_args, stack_args) = args.split_at(split_idx);
        // adjust stack alignment

        let stack_padding: usize = if stack_args.len() % 2 == 0 { 0 } else { 8 };

        let mut instructions = if stack_padding == 0 {
            vec![]
        } else {
            vec![Instruction::Binary {
                op: BinaryOperator::Sub,
                t: AsmType::Quadword,
                src: Operand::Imm(stack_padding as i64),
                dst: Operand::Reg(Reg::SP),
            }]
        };

        // pass args in registers
        instructions.extend(reg_args.iter().enumerate().map(|(idx, arg)| {
            let r = &PARAM_PASSING_REGS[idx];
            let assembly_arg = convert_val(arg);
            Instruction::Mov(self.asm_type(arg), assembly_arg, Operand::Reg(r.clone()))
        }));

        // pass args on the stack
        for arg in stack_args.iter().rev() {
            let assembly_arg = convert_val(arg);
            match assembly_arg {
                Operand::Imm(_) | Operand::Reg(_) => {
                    instructions.push(Instruction::Push(assembly_arg))
                }
                _ => {
                    let assembly_type = self.asm_type(arg);
                    if assembly_type == AsmType::Quadword {
                        instructions.push(Instruction::Push(assembly_arg));
                    } else {
                        // copy into a register before pushing
                        instructions.extend(vec![
                            Instruction::Mov(assembly_type, assembly_arg, Operand::Reg(Reg::AX)),
                            Instruction::Push(Operand::Reg(Reg::AX)),
                        ]);
                    }
                }
            }
        }

        // allocate shadow space for the stack
        instructions.push(Instruction::Binary {
            op: BinaryOperator::Sub,
            t: AsmType::Quadword,
            src: Operand::Imm(32),
            dst: Operand::Reg(Reg::SP),
        });

        // call the function
        instructions.push(Instruction::Call(f.to_string()));

        // adjust stack pointer (32 bytes for shadow space + 8 bytes for each stack argument)
        let bytes_to_remove = 32 + (8 * stack_args.len()) + stack_padding;
        instructions.push(Instruction::Binary {
            op: BinaryOperator::Add,
            t: AsmType::Quadword,
            src: Operand::Imm(bytes_to_remove as i64),
            dst: Operand::Reg(Reg::SP),
        });

        // retrieve return value
        let assembly_dst = convert_val(&dst);
        instructions.push(Instruction::Mov(
            self.asm_type(dst),
            Operand::Reg(Reg::AX),
            assembly_dst,
        ));

        instructions
    }

    fn convert_instruction(&self, instruction: &TackyInstruction) -> Vec<Instruction> {
        match instruction {
            TackyInstruction::Copy { src, dst } => {
                let t = self.asm_type(src);
                let asm_src = convert_val(&src);
                let asm_dst = convert_val(&dst);
                vec![Instruction::Mov(t, asm_src, asm_dst)]
            }
            TackyInstruction::Return(tacky_val) => {
                let t = self.asm_type(tacky_val);
                let asm_val = convert_val(&tacky_val);
                vec![
                    Instruction::Mov(t, asm_val, Operand::Reg(Reg::AX)),
                    Instruction::Ret,
                ]
            }
            TackyInstruction::Unary {
                op: TackyUnaryOp::Not,
                src,
                dst,
            } => {
                let src_t = self.asm_type(src);
                let dst_t = self.asm_type(dst);

                let asm_src = convert_val(&src);
                let asm_dst = convert_val(&dst);
                vec![
                    Instruction::Cmp(src_t, ZERO, asm_src),
                    Instruction::Mov(dst_t, ZERO, asm_dst.clone()),
                    Instruction::SetCC(CondCode::E, asm_dst),
                ]
            }
            TackyInstruction::Unary { op, src, dst } => {
                let t = self.asm_type(src);
                let asm_op = convert_unop(&op);
                let asm_src = convert_val(&src);
                let asm_dst = convert_val(&dst);
                vec![
                    Instruction::Mov(t.clone(), asm_src, asm_dst.clone()),
                    Instruction::Unary(asm_op, t, asm_dst),
                ]
            }
            TackyInstruction::Binary {
                op,
                src1,
                src2,
                dst,
            } => {
                let src_t = self.asm_type(src1);
                let dst_t = self.asm_type(dst);
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
                        let signed = self.tacky_type(src1).is_signed();
                        let cond_code = convert_cond_code(signed, &op);
                        vec![
                            Instruction::Cmp(src_t, asm_src2, asm_src1),
                            Instruction::Mov(dst_t, ZERO, asm_dst.clone()),
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
                        if self.tacky_type(src1).is_signed() || self.tacky_type(src2).is_signed() {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, Operand::Reg(Reg::AX)),
                                Instruction::Cdq(self.asm_type(src2).clone()),
                                Instruction::Idiv(self.asm_type(src2).clone(), asm_src2),
                                Instruction::Mov(dst_t, Operand::Reg(result_reg), asm_dst),
                            ]
                        } else {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, Operand::Reg(Reg::AX)),
                                Instruction::Mov(src_t.clone(), ZERO, Operand::Reg(Reg::DX)),
                                Instruction::Div(self.asm_type(src2).clone(), asm_src2),
                                Instruction::Mov(dst_t, Operand::Reg(result_reg), asm_dst),
                            ]
                        }
                    }
                    // Bitwise shift
                    TackyBinaryOp::LeftShift => {
                        if self.tacky_type(src1).is_signed() {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, asm_dst.clone()),
                                Instruction::Binary {
                                    op: BinaryOperator::Sal,
                                    t: src_t,
                                    src: asm_src2,
                                    dst: asm_dst,
                                },
                            ]
                        } else {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, asm_dst.clone()),
                                Instruction::Binary {
                                    op: BinaryOperator::Shl,
                                    t: src_t,
                                    src: asm_src2,
                                    dst: asm_dst,
                                },
                            ]
                        }
                    }
                    TackyBinaryOp::RightShift => {
                        if self.tacky_type(src1).is_signed() {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, asm_dst.clone()),
                                Instruction::Binary {
                                    op: BinaryOperator::Sar,
                                    t: src_t,
                                    src: asm_src2,
                                    dst: asm_dst,
                                },
                            ]
                        } else {
                            vec![
                                Instruction::Mov(src_t.clone(), asm_src1, asm_dst.clone()),
                                Instruction::Binary {
                                    op: BinaryOperator::Shr,
                                    t: src_t,
                                    src: asm_src2,
                                    dst: asm_dst,
                                },
                            ]
                        }
                    }
                    // Addition/subtraction/multiplication
                    _ => {
                        let asm_op = convert_binop(&op);
                        vec![
                            Instruction::Mov(src_t.clone(), asm_src1, asm_dst.clone()),
                            Instruction::Binary {
                                op: asm_op,
                                t: src_t,
                                src: asm_src2,
                                dst: asm_dst,
                            },
                        ]
                    }
                }
            }
            TackyInstruction::Jump(target) => vec![Instruction::Jmp(target.to_string())],
            TackyInstruction::JumpIfZero(cond, target) => {
                let t = self.asm_type(cond);
                let asm_cond = convert_val(&cond);
                vec![
                    Instruction::Cmp(t, ZERO, asm_cond),
                    Instruction::JmpCC(CondCode::E, target.to_string()),
                ]
            }
            TackyInstruction::JumpIfNotZero(cond, target) => {
                let t = self.asm_type(cond);
                let asm_cond = convert_val(&cond);
                vec![
                    Instruction::Cmp(t, ZERO, asm_cond),
                    Instruction::JmpCC(CondCode::NE, target.to_string()),
                ]
            }
            TackyInstruction::Label(l) => vec![Instruction::Label(l.to_string())],
            TackyInstruction::FunCall { f, args, dst } => {
                self.convert_function_call(&f, &args, &dst)
            }
            TackyInstruction::SignExtend { src, dst } => {
                let asm_src = convert_val(src);
                let asm_dst = convert_val(dst);
                vec![Instruction::Movsx(asm_src, asm_dst)]
            }
            TackyInstruction::Truncate { src, dst } => {
                let asm_src = convert_val(src);
                let asm_dst = convert_val(dst);
                vec![Instruction::Mov(AsmType::Longword, asm_src, asm_dst)]
            }
            TackyInstruction::ZeroExtend { src, dst } => {
                let asm_src = convert_val(src);
                let asm_dst = convert_val(dst);
                vec![Instruction::MovZeroExtend(asm_src, asm_dst)]
            }
        }
    }

    fn pass_params(&self, param_list: &Vec<String>) -> Vec<Instruction> {
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
            let param_t = self.asm_type(&TackyVal::Var(param.to_string()));
            instructions.push(Instruction::Mov(
                param_t.clone(),
                Operand::Reg(reg.clone()),
                stk.clone(),
            ));
            // Move from stack slot to pseudo for IR
            instructions.push(Instruction::Mov(
                param_t,
                stk,
                Operand::Pseudo(param.to_string()),
            ));
        }
        // Stack arguments: 5th arg at 48(%rbp), 6th at 56(%rbp), ...
        for (idx, param) in stack_params.iter().enumerate() {
            let offset = 16 + ((register_params.len() + idx) as isize) * 8;
            let stk = Operand::Stack(offset);
            let param_t = self.asm_type(&TackyVal::Var(param.to_string()));
            instructions.push(Instruction::Mov(
                param_t,
                stk,
                Operand::Pseudo(param.to_string()),
            ));
        }

        instructions
    }

    fn convert_top_level(&self, top_level: &TopLevel) -> AssemblyTopLevel {
        match top_level {
            TopLevel::FunctionDefinition {
                name,
                global,
                params,
                body,
            } => {
                let mut instructions = self.pass_params(&params);
                for instruction in body {
                    instructions.extend(self.convert_instruction(instruction));
                }

                AssemblyTopLevel::Function {
                    name: name.to_string(),
                    global: *global,
                    instructions,
                }
            }
            TopLevel::StaticVariable {
                name,
                global,
                t,
                init,
            } => AssemblyTopLevel::StaticVariable {
                name: name.to_string(),
                global: *global,
                alignment: t.get_alignment(),
                init: *init,
            },
        }
    }

    pub fn generate(&mut self, program: &Program) -> AssemblyProgram {
        let Program { top_levels } = program;

        let prog = AssemblyProgram {
            top_levels: top_levels
                .into_iter()
                .map(|top_level| self.convert_top_level(top_level))
                .collect(),
        };

        let assembly_symbols = &mut self.assembly_symbols;

        for (name, entry) in self.symbol_table.iter() {
            convert_symbol(assembly_symbols, &entry, &name);
        }
        prog
    }
}

fn convert_symbol(assembly_symbols: &mut BackendSymbolTable, entry: &Entry, name: &str) {
    match entry {
        Entry {
            t: Type::FunType { .. },
            attrs: IdentifierAttrs::FunAttr { defined, .. },
        } => assembly_symbols.add_fun(name, *defined),
        Entry {
            t,
            attrs: IdentifierAttrs::StaticAttr { .. },
        } => assembly_symbols.add_var(name, &convert_type(t), true),
        Entry { t, .. } => assembly_symbols.add_var(name, &convert_type(t), false),
    }
}
