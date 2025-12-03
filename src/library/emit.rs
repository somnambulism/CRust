use std::fs::File;
use std::io::{Result, Write};

use crate::library::assembly::AsmType;
use crate::library::initializers::StaticInit;

use super::assembly::{BinaryOperator, CondCode, Program, Reg, TopLevel, UnaryOperator};
use super::backend::assembly_symbols::SymbolTable;
use super::{
    assembly::{Instruction, Operand},
    settings::Target,
};

pub struct CodeEmitter {
    platform: Target,
    file: File,
    symbols: SymbolTable,
}

fn suffix(t: &AsmType) -> String {
    match t {
        AsmType::Longword => "l".to_string(),
        AsmType::Quadword => "q".to_string(),
        AsmType::Double => "sd".to_string(),
    }
}

fn show_long_reg(reg: &Reg) -> String {
    match reg {
        Reg::AX => "%eax".to_string(),
        Reg::CX => "%ecx".to_string(),
        Reg::DX => "%edx".to_string(),
        Reg::R8 => "%r8d".to_string(),
        Reg::R9 => "%r9d".to_string(),
        Reg::R10 => "%r10d".to_string(),
        Reg::R11 => "%r11d".to_string(),
        Reg::SP => panic!("Internal error: no 32-bit RSP"),
        Reg::BP => panic!("Internal error: no 32-bit RBP"),
        _ => panic!(
            "Internal error: can't store longword type in XMM register {:?}",
            reg
        ),
    }
}

fn show_quadword_reg(reg: &Reg) -> String {
    match reg {
        Reg::AX => "%rax".to_string(),
        Reg::CX => "%rcx".to_string(),
        Reg::DX => "%rdx".to_string(),
        Reg::R8 => "%r8".to_string(),
        Reg::R9 => "%r9".to_string(),
        Reg::R10 => "%r10".to_string(),
        Reg::R11 => "%r11".to_string(),
        Reg::SP => "%rsp".to_string(),
        Reg::BP => "%rbp".to_string(),
        _ => panic!(
            "Internal error: can't store quadword type in XMM register {:?}",
            reg
        ),
    }
}

fn show_double_reg(reg: &Reg) -> String {
    match reg {
        Reg::XMM0 => "%xmm0".to_string(),
        Reg::XMM1 => "%xmm1".to_string(),
        Reg::XMM2 => "%xmm2".to_string(),
        Reg::XMM3 => "%xmm3".to_string(),
        Reg::XMM7 => "%xmm7".to_string(),
        Reg::XMM14 => "%xmm14".to_string(),
        Reg::XMM15 => "%xmm15".to_string(),
        _ => panic!(
            "Internal error: can't store double type in general-purpose register {:?}",
            reg
        ),
    }
}

fn show_byte_reg(reg: &Reg) -> String {
    match reg {
        Reg::AX => "%al".to_string(),
        Reg::CX => "%cl".to_string(),
        Reg::DX => "%dl".to_string(),
        Reg::R8 => "%r8b".to_string(),
        Reg::R9 => "%r9b".to_string(),
        Reg::R10 => "%r10b".to_string(),
        Reg::R11 => "%r11b".to_string(),
        Reg::SP => panic!("Internal error: no one-byte RSP"),
        Reg::BP => panic!("Internal error: no one-byte RBP"),
        _ => panic!(
            "Internal error: can't store byte type in XMM register {:?}",
            reg
        ),
    }
}

fn show_unary_instruction(operator: &UnaryOperator) -> String {
    match operator {
        UnaryOperator::Neg => "neg".to_string(),
        UnaryOperator::Not => "not".to_string(),
        UnaryOperator::Shr => "shr".to_string(),
    }
}

fn show_binary_instruction(operator: &BinaryOperator) -> String {
    match operator {
        BinaryOperator::Add => "add".to_string(),
        BinaryOperator::Sub => "sub".to_string(),
        BinaryOperator::Mult => "imul".to_string(),
        BinaryOperator::DivDouble => "div".to_string(),
        BinaryOperator::And => "and".to_string(),
        BinaryOperator::Or => "or".to_string(),
        BinaryOperator::Xor => "xor".to_string(),
        BinaryOperator::Sal => "sal".to_string(),
        BinaryOperator::Sar => "sar".to_string(),
        BinaryOperator::Shl => "shl".to_string(),
        BinaryOperator::Shr => "shr".to_string(),
    }
}

fn show_cond_code(cond_code: &CondCode) -> String {
    match cond_code {
        CondCode::E => "e".to_string(),
        CondCode::NE => "ne".to_string(),
        CondCode::G => "g".to_string(),
        CondCode::GE => "ge".to_string(),
        CondCode::L => "l".to_string(),
        CondCode::LE => "le".to_string(),
        CondCode::A => "a".to_string(),
        CondCode::AE => "ae".to_string(),
        CondCode::B => "b".to_string(),
        CondCode::BE => "be".to_string(),
        CondCode::P => "p".to_string(),
        CondCode::NP => "np".to_string(),
    }
}

impl CodeEmitter {
    pub fn new(platform: Target, filename: &str, symbols: SymbolTable) -> Result<Self> {
        let file = File::create(filename)?;
        Ok(Self {
            platform,
            file,
            symbols,
        })
    }

    fn align_directive(&self) -> String {
        match self.platform {
            Target::OsX => ".balign".to_string(),
            Target::Linux => ".align".to_string(),
            Target::Windows => ".align".to_string(),
        }
    }

    fn show_label(&self, name: &str) -> String {
        match self.platform {
            Target::OsX => format!("_{}", name),
            _ => name.to_string(),
        }
    }

    fn show_local_label(&self, name: &str) -> String {
        match self.platform {
            Target::OsX => format!("L{}", name),
            _ => format!(".L{}", name),
        }
    }

    fn show_fun_name(&self, f: &str) -> String {
        match self.platform {
            Target::OsX => format!("_{}", f),
            Target::Linux => {
                if self.symbols.is_defined(f) {
                    f.to_string()
                } else {
                    format!("{}@PLT", f)
                }
            }
            Target::Windows => f.to_string(),
        }
    }

    fn show_operand(&self, t: &AsmType, operand: &Operand) -> String {
        match operand {
            Operand::Reg(r) => match t {
                AsmType::Longword => show_long_reg(r),
                AsmType::Quadword => show_quadword_reg(r),
                AsmType::Double => show_double_reg(r),
            },
            Operand::Imm(i) => format!("${}", i),
            Operand::Memory(r, 0) => format!("({})", show_quadword_reg(r)),
            Operand::Memory(r, i) => format!("{}({})", i, show_quadword_reg(r)),
            Operand::Data(name) => {
                let lbl = if self.symbols.is_constant(name) {
                    self.show_local_label(name)
                } else {
                    self.show_label(name)
                };
                format!("{}(%rip)", lbl)
            }
            // printing out pseudoregister is only for debugging
            Operand::Pseudo(name) => format!("%{}", name),
        }
    }

    fn show_byte_operand(&self, operand: &Operand) -> String {
        match operand {
            Operand::Reg(r) => show_byte_reg(r),
            other => self.show_operand(&AsmType::Longword, other),
        }
    }

    fn emit_instruction(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Mov(t, src, dst) => writeln!(
                self.file,
                "\tmov{} {}, {}",
                suffix(t),
                self.show_operand(t, &src),
                self.show_operand(t, &dst)
            ),
            Instruction::Unary(operator, t, dst) => writeln!(
                self.file,
                "\t{}{} {}",
                show_unary_instruction(operator),
                suffix(t),
                self.show_operand(t, dst)
            ),
            Instruction::Binary {
                op: BinaryOperator::Xor,
                t: AsmType::Double,
                src,
                dst,
            } => {
                writeln!(
                    self.file,
                    "\txorpd {}, {}",
                    self.show_operand(&AsmType::Double, src),
                    self.show_operand(&AsmType::Double, dst)
                )
            }
            Instruction::Binary {
                op: BinaryOperator::Mult,
                t: AsmType::Double,
                src,
                dst,
            } => {
                writeln!(
                    self.file,
                    "\tmulsd {}, {}",
                    self.show_operand(&AsmType::Double, src),
                    self.show_operand(&AsmType::Double, dst)
                )
            }
            Instruction::Binary {
                op: op @ BinaryOperator::Sal | op @ BinaryOperator::Sar,
                t,
                src,
                dst,
            } => writeln!(
                self.file,
                "\t{}{} {}, {}",
                show_binary_instruction(op),
                suffix(t),
                self.show_byte_operand(src),
                self.show_operand(t, dst)
            ),
            Instruction::Binary { op, t, src, dst } => writeln!(
                self.file,
                "\t{}{} {}, {}",
                show_binary_instruction(op),
                suffix(t),
                self.show_operand(t, src),
                self.show_operand(t, dst)
            ),
            Instruction::Cmp(AsmType::Double, src, dst) => writeln!(
                self.file,
                "\tcomisd {}, {}",
                self.show_operand(&AsmType::Double, src),
                self.show_operand(&AsmType::Double, dst)
            ),
            Instruction::Cmp(t, src, dst) => writeln!(
                self.file,
                "\tcmp{} {}, {}",
                suffix(t),
                self.show_operand(t, src),
                self.show_operand(t, dst),
            ),
            Instruction::Idiv(t, operand) => {
                writeln!(
                    self.file,
                    "\tidiv{} {}",
                    suffix(t),
                    self.show_operand(t, operand)
                )
            }
            Instruction::Div(t, operand) => {
                writeln!(
                    self.file,
                    "\tdiv{} {}",
                    suffix(t),
                    self.show_operand(t, operand)
                )
            }
            Instruction::Lea(src, dst) => writeln!(
                self.file,
                "\tleaq {}, {}",
                self.show_operand(&AsmType::Quadword, src),
                self.show_operand(&AsmType::Quadword, dst)
            ),
            Instruction::Cdq(AsmType::Longword) => writeln!(self.file, "\tcdq"),
            Instruction::Cdq(AsmType::Quadword) => writeln!(self.file, "\tcqo"),
            Instruction::Jmp(lbl) => writeln!(self.file, "\tjmp {}", self.show_local_label(lbl)),
            Instruction::JmpCC(code, lbl) => writeln!(
                self.file,
                "\tj{} {}",
                show_cond_code(code),
                self.show_local_label(lbl),
            ),
            Instruction::SetCC(code, operand) => writeln!(
                self.file,
                "\tset{} {}",
                show_cond_code(code),
                self.show_byte_operand(operand),
            ),
            Instruction::Label(lbl) => writeln!(self.file, "{}:", self.show_local_label(lbl)),
            Instruction::Push(op) => {
                writeln!(
                    self.file,
                    "\tpushq {}",
                    self.show_operand(&AsmType::Quadword, op)
                )
            }
            Instruction::Call(f) => {
                writeln!(self.file, "\tcall {}", self.show_fun_name(f))
            }
            Instruction::Movsx(src, dst) => {
                writeln!(
                    self.file,
                    "\tmovslq {}, {}",
                    self.show_operand(&AsmType::Longword, src),
                    self.show_operand(&AsmType::Quadword, dst)
                )
            }
            Instruction::Cvtsi2sd(t, src, dst) => writeln!(
                self.file,
                "\tcvtsi2sd{} {}, {}",
                suffix(t),
                self.show_operand(t, src),
                self.show_operand(&AsmType::Double, dst)
            ),
            Instruction::Cvttsd2si(t, src, dst) => writeln!(
                self.file,
                "\tcvttsd2si{} {}, {}",
                suffix(t),
                self.show_operand(&AsmType::Double, src),
                self.show_operand(t, dst)
            ),
            Instruction::Ret => writeln!(self.file, "\tmovq %rbp, %rsp\n\tpopq %rbp\n\tret"),
            Instruction::Cdq(AsmType::Double) => {
                panic!("Intenral error: can't apply cdq to double type")
            }
            Instruction::MovZeroExtend(..) => {
                panic!(
                    "Internal error: MovZeroExtend should have been removed in instruction rewrite pass"
                )
            }
        }
    }

    fn emit_global_directive(&mut self, global: bool, label: &str) -> Result<()> {
        if global {
            writeln!(self.file, "\t.globl {}", label)
        } else {
            Ok(())
        }
    }

    fn emit_zero_init(&mut self, init: &StaticInit) -> Result<()> {
        match init {
            StaticInit::IntInit(_) | StaticInit::UIntInit(_) => writeln!(self.file, "\t.zero 4"),
            StaticInit::LongInit(_) | StaticInit::ULongInit(_) => writeln!(self.file, "\t.zero 8"),
            StaticInit::DoubleInit(_) => {
                panic!("Internal error: should never use zeroinit for doubles")
            }
            _ => todo!(),
        }
    }

    fn emit_init(&mut self, init: &StaticInit) -> Result<()> {
        match init {
            StaticInit::IntInit(i) => writeln!(self.file, "\t.long {}", i),
            StaticInit::LongInit(l) => writeln!(self.file, "\t.quad {}", l),
            StaticInit::UIntInit(u) => writeln!(self.file, "\t.long {}", u),
            StaticInit::ULongInit(l) => writeln!(self.file, "\t.quad {}", l),
            StaticInit::DoubleInit(d) => writeln!(self.file, "\t.quad {}", d.to_bits()),
            _ => todo!(),
        }
    }

    fn emit_constant(&mut self, name: &str, alignment: i8, init: &StaticInit) -> Result<()> {
        let constant_section_name = match self.platform {
            Target::Linux => ".section .rodata",
            Target::OsX => {
                if alignment == 8 {
                    ".literal 8"
                } else if alignment == 16 {
                    ".literal 16"
                } else {
                    panic!(
                        "Internal error: found constant with bad alignment {} {}",
                        name, alignment
                    )
                }
            }
            Target::Windows => ".section .rdata",
        };
        writeln!(self.file, "\t{}", constant_section_name)?;
        writeln!(self.file, "\t{} {}", self.align_directive(), alignment)?;
        writeln!(self.file, "{}:", self.show_local_label(name))?;
        self.emit_init(init)?;
        // macOS linker gets cranky if you write only 8 bytes to .literal16 section
        // CRust supports only Windows, but who knows
        if constant_section_name == ".literal16" {
            self.emit_init(&StaticInit::LongInit(0))?;
        }
        Ok(())
    }

    fn emit_tl(&mut self, tl: &TopLevel) -> Result<()> {
        match tl {
            TopLevel::Function {
                name,
                global,
                instructions,
            } => {
                let label = self.show_label(&name);

                self.emit_global_directive(*global, &label)?;
                writeln!(self.file, "\t.text")?;
                writeln!(self.file, "{}:", label)?;
                writeln!(self.file, "\tpushq %rbp")?;
                writeln!(self.file, "\tmovq %rsp, %rbp")?;
                for instr in instructions {
                    self.emit_instruction(&instr)?;
                }
                writeln!(self.file)?;
                Ok(())
            }
            TopLevel::StaticVariable {
                name,
                global,
                init,
                alignment,
            }
            // is_zero is false for all doubles
            if init.is_zero() => {
                let label = self.show_label(&name);
                self.emit_global_directive(*global, &label)?;
                writeln!(self.file, "\t.bss")?;
                writeln!(self.file, "\t{} {}", self.align_directive(), alignment)?;
                writeln!(self.file, "{}:", label)?;
                self.emit_zero_init(&init)?;
                writeln!(self.file)?;
                Ok(())
            }
            TopLevel::StaticVariable {
                name,
                global,
                init,
                alignment,
            } => {
                let label = self.show_label(&name);
                self.emit_global_directive(*global, &label)?;
                writeln!(self.file, "\t.data")?;
                writeln!(self.file, "\t{} {}", self.align_directive(), alignment)?;
                writeln!(self.file, "{}:", label)?;
                self.emit_init(&init)?;
                writeln!(self.file)?;
                Ok(())
            }
            TopLevel::StaticConstant { name, alignment, init } => self.emit_constant(name, *alignment, init)
        }
    }

    fn emit_stack_note(&mut self) -> Result<()> {
        match self.platform {
            Target::Linux => writeln!(self.file, "\t.section .note.GNU-stack,\"\",@progbits"),
            _ => Ok(()),
        }
    }

    pub fn emit(&mut self, program: &Program) -> Result<()> {
        for tl in &program.top_levels {
            self.emit_tl(&tl)?;
        }
        self.emit_stack_note()?;
        Ok(())
    }
}
