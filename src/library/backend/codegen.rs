use std::{collections::HashMap, i64, vec};

use crate::library::{
    assembly::{
        AsmType, BinaryOperator, CondCode, Instruction, Operand, Program as AssemblyProgram, Reg,
        TopLevel as AssemblyTopLevel, UnaryOperator,
    },
    backend::assembly_symbols::SymbolTable as BackendSymbolTable,
    r#const::{T, type_of_const},
    initializers::StaticInit,
    symbols::{Entry, IdentifierAttrs, SymbolTable},
    tacky::{
        BinaryOperator as TackyBinaryOp, Instruction as TackyInstruction, Program, TackyVal,
        TopLevel, UnaryOperator as TackyUnaryOp,
    },
    types::Type,
    util::unique_ids::make_label,
};

const INT_PARAM_PASSING_REGS: [Reg; 4] = [Reg::CX, Reg::DX, Reg::R8, Reg::R9];

const DBL_PARAM_PASSING_REGS: [Reg; 4] = [Reg::XMM0, Reg::XMM1, Reg::XMM2, Reg::XMM3];

const ZERO: Operand = Operand::Imm(0);

#[derive(Debug)]
enum ArgLoc {
    Reg(Reg),
    Stack,
}

struct ArgAssignment {
    operand: Operand,
    asm_type: AsmType,
    loc: ArgLoc,
}

pub struct CodeGen {
    pub symbol_table: SymbolTable,
    pub assembly_symbols: BackendSymbolTable,
    pub constants: HashMap<u64, (String, i8)>,
}

fn convert_type(t: &Type) -> AsmType {
    match t {
        Type::Int | Type::UInt => AsmType::Longword,
        Type::Long | Type::ULong => AsmType::Quadword,
        Type::Double => AsmType::Double,
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
        TackyBinaryOp::Divide => BinaryOperator::DivDouble, // NB should only be called for operands on doubles
        TackyBinaryOp::BitwiseAnd => BinaryOperator::And,
        TackyBinaryOp::BitwiseOr => BinaryOperator::Or,
        TackyBinaryOp::Xor => BinaryOperator::Xor,
        TackyBinaryOp::LeftShift => BinaryOperator::Sal,
        TackyBinaryOp::RightShift => BinaryOperator::Sar,
        TackyBinaryOp::Mod
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
            constants: HashMap::with_capacity(10),
        }
    }

    fn add_constant(&mut self, alignment: i8, dbl: f64) -> String {
        let key = dbl.to_bits();
        // see if we've defined this double already
        if let Some((name, old_alignment)) = self.constants.get(&key) {
            let name = name.clone();
            let old_alignment = *old_alignment;
            // update alignment to max of current and new
            self.constants
                .insert(key, (name.clone(), alignment.max(old_alignment)));
            name
        } else {
            // we haven't defined it yet, add it to the table
            let name = make_label("dbl");
            self.constants.insert(key, (name.clone(), alignment));
            name
        }
    }

    fn convert_val(&mut self, val: &TackyVal) -> Operand {
        match val {
            TackyVal::Constant(T::ConstInt(i)) => Operand::Imm(*i as i64),
            TackyVal::Constant(T::ConstLong(l)) => Operand::Imm(*l),
            TackyVal::Constant(T::ConstUInt(u)) => Operand::Imm(*u as i64),
            TackyVal::Constant(T::ConstULong(ul)) => Operand::Imm(*ul as i64),
            TackyVal::Constant(T::ConstDouble(d)) => Operand::Data(self.add_constant(8, *d)),
            TackyVal::Var(v) => Operand::Pseudo(v.to_string()),
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

    // Helper functions for double comparisons w/ support for NaN
    fn convert_dbl_comparison(
        &mut self,
        op: &TackyBinaryOp,
        dst_t: &AsmType,
        asm_src1: Operand,
        asm_src2: Operand,
        asm_dst: Operand,
    ) -> Vec<Instruction> {
        let cond_code = convert_cond_code(false, op);
        /*
         * If op is A or AE, can perform usual comparisons;
         * these are true only if some flags are 0, so they'll be false for unordered results.
         * If op is B or BE, just flip operands and use A or AE instead.
         * If op is E or NE, need to check for parity afterwards.
         */
        let (cond_code, asm_src1, asm_src2) = match cond_code {
            CondCode::B => (CondCode::A, asm_src2, asm_src1),
            CondCode::BE => (CondCode::AE, asm_src2, asm_src1),
            _ => (cond_code, asm_src1, asm_src2),
        };
        let instrs = vec![
            Instruction::Cmp(AsmType::Double, asm_src2, asm_src1),
            Instruction::Mov(dst_t.clone(), ZERO, asm_dst.clone()),
            Instruction::SetCC(cond_code.clone(), asm_dst.clone()),
        ];
        let parity_instrs = match cond_code {
            // zero out destination if parity flag is set,
            // indicating unordered result
            CondCode::E => vec![
                Instruction::Mov(dst_t.clone(), ZERO, Operand::Reg(Reg::R9)),
                Instruction::SetCC(CondCode::NP, Operand::Reg(Reg::R9)),
                Instruction::Binary {
                    op: BinaryOperator::And,
                    t: dst_t.clone(),
                    src: Operand::Reg(Reg::R9),
                    dst: asm_dst.clone(),
                },
            ],
            // set destination to 1 if parity flag is set, indicating ordered result
            CondCode::NE => vec![
                Instruction::Mov(dst_t.clone(), ZERO, Operand::Reg(Reg::R9)),
                Instruction::SetCC(CondCode::P, Operand::Reg(Reg::R9)),
                Instruction::Binary {
                    op: BinaryOperator::Or,
                    t: dst_t.clone(),
                    src: Operand::Reg(Reg::R9),
                    dst: asm_dst.clone(),
                },
            ],
            _ => vec![],
        };
        [instrs, parity_instrs].concat()
    }

    fn classify_parameters(&mut self, tacky_vals: &Vec<TackyVal>) -> Vec<ArgAssignment> {
        let mut assignments = Vec::new();

        for (i, v) in tacky_vals.iter().enumerate() {
            let operand = self.convert_val(v);
            let t = self.asm_type(v);

            let loc = if i < 4 {
                match t {
                    AsmType::Double => ArgLoc::Reg(DBL_PARAM_PASSING_REGS[i].clone()),
                    _ => ArgLoc::Reg(INT_PARAM_PASSING_REGS[i].clone()),
                }
            } else {
                ArgLoc::Stack
            };

            assignments.push(ArgAssignment {
                operand,
                asm_type: t,
                loc,
            });
        }

        assignments
    }

    fn convert_function_call(
        &mut self,
        f: &str,
        args: &Vec<TackyVal>,
        dst: &TackyVal,
    ) -> Vec<Instruction> {
        let args = self.classify_parameters(args);

        // adjust stack alignment
        let stack_count = args.len().saturating_sub(4);
        let stack_padding: usize = if stack_count % 2 == 0 { 0 } else { 8 };

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

        let pass_reg_args = args.iter().filter(|a| matches!(a.loc, ArgLoc::Reg(_)));
        let pass_stack_args = args.iter().rev().filter(|a| matches!(a.loc, ArgLoc::Stack));

        // pass args in registers
        for arg in pass_reg_args {
            if let ArgLoc::Reg(r) = &arg.loc {
                match arg.asm_type {
                    AsmType::Double => instructions.push(Instruction::Mov(
                        AsmType::Double,
                        arg.operand.clone(),
                        Operand::Reg(r.clone()),
                    )),
                    _ => instructions.push(Instruction::Mov(
                        arg.asm_type.clone(),
                        arg.operand.clone(),
                        Operand::Reg(r.clone()),
                    )),
                }
            }
        }

        // pass args on stack
        for arg in pass_stack_args {
            match arg.operand {
                Operand::Imm(_) | Operand::Reg(_) => {
                    instructions.push(Instruction::Push(arg.operand.clone()))
                }
                _ => {
                    if arg.asm_type == AsmType::Quadword || arg.asm_type == AsmType::Double {
                        instructions.push(Instruction::Push(arg.operand.clone()))
                    } else {
                        // copy into a register before pushing
                        instructions.extend(vec![
                            Instruction::Mov(
                                arg.asm_type.clone(),
                                arg.operand.clone(),
                                Operand::Reg(Reg::AX),
                            ),
                            Instruction::Push(Operand::Reg(Reg::AX)),
                        ])
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
        let bytes_to_remove = 32 + 8 * stack_count + stack_padding;
        instructions.push(Instruction::Binary {
            op: BinaryOperator::Add,
            t: AsmType::Quadword,
            src: Operand::Imm(bytes_to_remove as i64),
            dst: Operand::Reg(Reg::SP),
        });

        // retrieve return value
        let assembly_dst = self.convert_val(&dst);
        let return_reg = if self.asm_type(dst) == AsmType::Double {
            Operand::Reg(Reg::XMM0)
        } else {
            Operand::Reg(Reg::AX)
        };
        instructions.push(Instruction::Mov(
            self.asm_type(dst),
            return_reg,
            assembly_dst,
        ));

        instructions
    }

    fn convert_instruction(&mut self, instruction: &TackyInstruction) -> Vec<Instruction> {
        match instruction {
            TackyInstruction::Copy { src, dst } => {
                let t = self.asm_type(src);
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                vec![Instruction::Mov(t, asm_src, asm_dst)]
            }
            TackyInstruction::Return(tacky_val) => {
                let t = self.asm_type(tacky_val);
                let asm_val = self.convert_val(&tacky_val);
                let ret_reg = if t == AsmType::Double {
                    Operand::Reg(Reg::XMM0)
                } else {
                    Operand::Reg(Reg::AX)
                };
                vec![Instruction::Mov(t, asm_val, ret_reg), Instruction::Ret]
            }
            TackyInstruction::Unary {
                op: TackyUnaryOp::Not,
                src,
                dst,
            } => {
                let src_t = self.asm_type(src);
                let dst_t = self.asm_type(dst);

                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                if src_t == AsmType::Double {
                    vec![
                        Instruction::Binary {
                            op: BinaryOperator::Xor,
                            t: AsmType::Double,
                            src: Operand::Reg(Reg::XMM0),
                            dst: Operand::Reg(Reg::XMM0),
                        },
                        Instruction::Cmp(src_t, asm_src, Operand::Reg(Reg::XMM0)),
                        Instruction::Mov(dst_t.clone(), ZERO, asm_dst.clone()),
                        Instruction::SetCC(CondCode::E, asm_dst.clone()),
                        // cmp with NaN sets both ZF and PF, but !NaN sould evaluate to 0,
                        // so we'll calculate:
                        // !x = ZF && !PF
                        Instruction::SetCC(CondCode::NP, Operand::Reg(Reg::R9)),
                        Instruction::Binary {
                            op: BinaryOperator::And,
                            t: dst_t,
                            src: Operand::Reg(Reg::R9),
                            dst: asm_dst,
                        },
                    ]
                } else {
                    vec![
                        Instruction::Cmp(src_t, ZERO, asm_src),
                        Instruction::Mov(dst_t, ZERO, asm_dst.clone()),
                        Instruction::SetCC(CondCode::E, asm_dst),
                    ]
                }
            }
            TackyInstruction::Unary {
                op: TackyUnaryOp::Negate,
                src,
                dst,
            } if self.tacky_type(src) == Type::Double => {
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                let negative_zero = self.add_constant(16, -0.0);
                vec![
                    Instruction::Mov(AsmType::Double, asm_src, asm_dst.clone()),
                    Instruction::Binary {
                        op: BinaryOperator::Xor,
                        t: AsmType::Double,
                        src: Operand::Data(negative_zero),
                        dst: asm_dst,
                    },
                ]
            }
            TackyInstruction::Unary { op, src, dst } => {
                let t = self.asm_type(src);
                let asm_op = convert_unop(&op);
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
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
                let asm_src1 = self.convert_val(&src1);
                let asm_src2 = self.convert_val(&src2);
                let asm_dst = self.convert_val(&dst);
                match op {
                    // Relational operator
                    TackyBinaryOp::Equal
                    | TackyBinaryOp::NotEqual
                    | TackyBinaryOp::GreaterThan
                    | TackyBinaryOp::GreaterOrEqual
                    | TackyBinaryOp::LessThan
                    | TackyBinaryOp::LessOrEqual => {
                        if src_t == AsmType::Double {
                            return self
                                .convert_dbl_comparison(op, &dst_t, asm_src1, asm_src2, asm_dst);
                        } else {
                            let signed = if src_t == AsmType::Double {
                                false
                            } else {
                                self.tacky_type(src1).is_signed()
                            };
                            let cond_code = convert_cond_code(signed, &op);
                            vec![
                                Instruction::Cmp(src_t, asm_src2, asm_src1),
                                Instruction::Mov(dst_t, ZERO, asm_dst.clone()),
                                Instruction::SetCC(cond_code, asm_dst),
                            ]
                        }
                    }
                    // Division/modulo
                    TackyBinaryOp::Divide | TackyBinaryOp::Mod if src_t != AsmType::Double => {
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
                let asm_cond = self.convert_val(&cond);
                let lbl = make_label("nan.jmp.end");
                if t == AsmType::Double {
                    vec![
                        Instruction::Binary {
                            op: BinaryOperator::Xor,
                            t: AsmType::Double,
                            src: Operand::Reg(Reg::XMM0),
                            dst: Operand::Reg(Reg::XMM0),
                        },
                        Instruction::Cmp(t, asm_cond, Operand::Reg(Reg::XMM0)),
                        // Comparison to NaN sets ZF and PF flag;
                        // to treat NaN as nonzero, skip over je instruction if PF flag is set
                        Instruction::JmpCC(CondCode::P, lbl.clone()),
                        Instruction::JmpCC(CondCode::E, target.to_string()),
                        Instruction::Label(lbl),
                    ]
                } else {
                    vec![
                        Instruction::Cmp(t, ZERO, asm_cond),
                        Instruction::JmpCC(CondCode::E, target.to_string()),
                    ]
                }
            }
            TackyInstruction::JumpIfNotZero(cond, target) => {
                let t = self.asm_type(cond);
                let asm_cond = self.convert_val(&cond);
                if t == AsmType::Double {
                    vec![
                        Instruction::Binary {
                            op: BinaryOperator::Xor,
                            t: AsmType::Double,
                            src: Operand::Reg(Reg::XMM0),
                            dst: Operand::Reg(Reg::XMM0),
                        },
                        Instruction::Cmp(t, asm_cond, Operand::Reg(Reg::XMM0)),
                        Instruction::JmpCC(CondCode::NE, target.to_string()),
                        Instruction::JmpCC(CondCode::P, target.to_string()),
                    ]
                } else {
                    vec![
                        Instruction::Cmp(t, ZERO, asm_cond),
                        Instruction::JmpCC(CondCode::NE, target.to_string()),
                    ]
                }
            }
            TackyInstruction::Label(l) => vec![Instruction::Label(l.to_string())],
            TackyInstruction::FunCall { f, args, dst } => {
                self.convert_function_call(&f, &args, &dst)
            }
            TackyInstruction::SignExtend { src, dst } => {
                let asm_src = self.convert_val(src);
                let asm_dst = self.convert_val(dst);
                vec![Instruction::Movsx(asm_src, asm_dst)]
            }
            TackyInstruction::Truncate { src, dst } => {
                let asm_src = self.convert_val(src);
                let asm_dst = self.convert_val(dst);
                vec![Instruction::Mov(AsmType::Longword, asm_src, asm_dst)]
            }
            TackyInstruction::ZeroExtend { src, dst } => {
                let asm_src = self.convert_val(src);
                let asm_dst = self.convert_val(dst);
                vec![Instruction::MovZeroExtend(asm_src, asm_dst)]
            }
            TackyInstruction::IntToDouble { src, dst } => {
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                let t = self.asm_type(&src);
                vec![Instruction::Cvtsi2sd(t, asm_src, asm_dst)]
            }
            TackyInstruction::DoubleToInt { src, dst } => {
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                let t = self.asm_type(&dst);
                vec![Instruction::Cvttsd2si(t, asm_src, asm_dst)]
            }
            TackyInstruction::UIntToDouble { src, dst } => {
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                if self.tacky_type(src) == Type::UInt {
                    vec![
                        Instruction::MovZeroExtend(asm_src, Operand::Reg(Reg::R9)),
                        Instruction::Cvtsi2sd(AsmType::Quadword, Operand::Reg(Reg::R9), asm_dst),
                    ]
                } else {
                    let out_of_bounds = make_label("ulong2dbl.oob");
                    let end_lbl = make_label("ulong2dbl.end");
                    let (r1, r2) = (Operand::Reg(Reg::R8), Operand::Reg(Reg::R9));
                    vec![
                        // check whether asm_src is w/in range of long
                        Instruction::Cmp(AsmType::Quadword, ZERO, asm_src.clone()),
                        Instruction::JmpCC(CondCode::L, out_of_bounds.clone()),
                        // it's in range, just use normal cvtsi2sd then jump to end
                        Instruction::Cvtsi2sd(AsmType::Quadword, asm_src.clone(), asm_dst.clone()),
                        Instruction::Jmp(end_lbl.clone()),
                        // it's out of bounds
                        Instruction::Label(out_of_bounds),
                        // halve source and round to dd
                        Instruction::Mov(AsmType::Quadword, asm_src, r1.clone()),
                        Instruction::Mov(AsmType::Quadword, r1.clone(), r2.clone()),
                        Instruction::Unary(UnaryOperator::Shr, AsmType::Quadword, r2.clone()),
                        Instruction::Binary {
                            op: BinaryOperator::And,
                            t: AsmType::Quadword,
                            src: Operand::Imm(1),
                            dst: r1.clone(),
                        },
                        Instruction::Binary {
                            op: BinaryOperator::Or,
                            t: AsmType::Quadword,
                            src: r1,
                            dst: r2.clone(),
                        },
                        // convert to double, then double it
                        Instruction::Cvtsi2sd(AsmType::Quadword, r2, asm_dst.clone()),
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            t: AsmType::Double,
                            src: asm_dst.clone(),
                            dst: asm_dst,
                        },
                        Instruction::Label(end_lbl),
                    ]
                }
            }
            TackyInstruction::DoubleToUInt { src, dst } => {
                let asm_src = self.convert_val(&src);
                let asm_dst = self.convert_val(&dst);
                if self.tacky_type(dst) == Type::UInt {
                    vec![
                        Instruction::Cvttsd2si(AsmType::Quadword, asm_src, Operand::Reg(Reg::R9)),
                        Instruction::Mov(AsmType::Longword, Operand::Reg(Reg::R9), asm_dst),
                    ]
                } else {
                    let out_of_bounds = make_label("dbl2ulong.oob");
                    let end_lbl = make_label("dbl2ulong.end");
                    let upper_bound = self.add_constant(8, 9223372036854775808.0);
                    let upper_bound_as_int =
                        // interpreted as signed integer, upper bound wraps around to become
                        // minimum int
                        Operand::Imm(i64::MIN);
                    let (r, x) = (Operand::Reg(Reg::R9), Operand::Reg(Reg::XMM7));
                    vec![
                        Instruction::Cmp(
                            AsmType::Double,
                            Operand::Data(upper_bound.clone()),
                            asm_src.clone(),
                        ),
                        Instruction::JmpCC(CondCode::AE, out_of_bounds.clone()),
                        Instruction::Cvttsd2si(AsmType::Quadword, asm_src.clone(), asm_dst.clone()),
                        Instruction::Jmp(end_lbl.clone()),
                        Instruction::Label(out_of_bounds),
                        Instruction::Mov(AsmType::Double, asm_src, x.clone()),
                        Instruction::Binary {
                            op: BinaryOperator::Sub,
                            t: AsmType::Double,
                            src: Operand::Data(upper_bound),
                            dst: x.clone(),
                        },
                        Instruction::Cvttsd2si(AsmType::Quadword, x, asm_dst.clone()),
                        Instruction::Mov(AsmType::Quadword, upper_bound_as_int, r.clone()),
                        Instruction::Binary {
                            op: BinaryOperator::Add,
                            t: AsmType::Quadword,
                            src: r,
                            dst: asm_dst,
                        },
                        Instruction::Label(end_lbl),
                    ]
                }
            }
        }
    }

    fn pass_params(&mut self, param_list: &Vec<TackyVal>) -> Vec<Instruction> {
        let params = self.classify_parameters(param_list);

        let mut instructions: Vec<Instruction> = vec![];

        for (idx, param) in params.iter().enumerate() {
            match &param.loc {
                ArgLoc::Reg(reg) => {
                    instructions.extend(vec![Instruction::Mov(
                        param.asm_type.clone(),
                        Operand::Reg(reg.clone()),
                        param.operand.clone(),
                    )]);
                    // let offset = 16 + (idx as isize) * 8;
                    // let stk = Operand::Stack(offset);
                    // instructions.push(Instruction::Mov(
                    //     param.asm_type.clone(),
                    //     Operand::Reg(reg.clone()),
                    //     stk.clone(),
                    // ));
                    // // Move from stack slot to pseudo for IR
                    // instructions.push(Instruction::Mov(
                    //     param.asm_type.clone(),
                    //     stk,
                    //     param.operand.clone(),
                    // ));
                }
                ArgLoc::Stack => {
                    // Stack arguments: 5th arg at 48(%rbp), 6th at 56(%rbp), ...
                    let offset = 16 + (idx as isize) * 8;
                    let stk = Operand::Stack(offset);
                    instructions.push(Instruction::Mov(
                        param.asm_type.clone(),
                        stk,
                        param.operand.clone(),
                    ));
                }
            }
        }

        // Windows x64 ABI: shadow space (32 bytes) is always reserved
        // Move register arguments to stack slots
        // for (idx, param) in params
        //     .iter()
        //     .filter(|param| matches!(param.loc, ArgLoc::Reg(_)))
        //     .enumerate()
        // {
        //     // let reg = &PARAM_PASSING_REGS[idx];
        //     // Windows: first arg at 16(%rbp), then 24, 32, 40
        //     let offset = 16 + (idx as isize) * 8;
        //     let stk = Operand::Stack(offset);
        //     // let param_t = self.asm_type(&TackyVal::Var(param.to_string()));
        //     let reg = match &param.loc {
        //         ArgLoc::Reg(r) => r,
        //         ArgLoc::Stack => panic!("XEP"),
        //     };
        //     instructions.push(Instruction::Mov(
        //         param.asm_type.clone(),
        //         Operand::Reg(reg.clone()),
        //         stk.clone(),
        //     ));
        //     // Move from stack slot to pseudo for IR
        //     instructions.push(Instruction::Mov(
        //         param.asm_type.clone(),
        //         stk,
        //         param.operand.clone(),
        //     ));
        // }
        // // Stack arguments: 5th arg at 48(%rbp), 6th at 56(%rbp), ...
        // for (idx, param) in params
        //     .iter()
        //     .filter(|param| matches!(param.loc, ArgLoc::Stack))
        //     .enumerate()
        // {
        //     let offset = 16 + ((4 + idx) as isize) * 8;
        //     let stk = Operand::Stack(offset);
        //     instructions.push(Instruction::Mov(
        //         param.asm_type.clone(),
        //         stk,
        //         param.operand.clone(),
        //     ));
        // }

        instructions
    }

    fn convert_top_level(&mut self, top_level: &TopLevel) -> AssemblyTopLevel {
        match top_level {
            TopLevel::FunctionDefinition {
                name,
                global,
                params,
                body,
            } => {
                let params_as_tacky = params
                    .iter()
                    .map(|name| TackyVal::Var(name.to_string()))
                    .collect();
                let mut instructions = self.pass_params(&params_as_tacky);
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

        // clear the hashmap (necessary if we're compiling multiple sources)
        self.constants.clear();

        let tls: Vec<_> = top_levels
            .into_iter()
            .map(|top_level| self.convert_top_level(top_level))
            .collect();

        let assembly_symbols = &mut self.assembly_symbols;

        let mut constants: Vec<_> = self
            .constants
            .iter()
            .map(|(k, v)| convert_constant(assembly_symbols, *k, v))
            .collect();

        constants.extend(tls);

        let prog = AssemblyProgram {
            top_levels: constants,
        };

        for (name, entry) in self.symbol_table.iter() {
            convert_symbol(assembly_symbols, &entry, &name);
        }
        prog
    }
}

fn convert_constant(
    assembly_symbols: &mut BackendSymbolTable,
    key: u64,
    value: &(String, i8),
) -> AssemblyTopLevel {
    let (name, alignment) = value;

    let dbl = f64::from_bits(key);
    assembly_symbols.add_constant(name, &AsmType::Double);
    AssemblyTopLevel::StaticConstant {
        name: name.clone(),
        alignment: *alignment,
        init: StaticInit::DoubleInit(dbl),
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
