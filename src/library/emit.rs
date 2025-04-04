use std::fs::File;
use std::io::{Result, Write};

use super::assembly::{BinaryOperator, CondCode, FunctionDefinition, Program, Reg, UnaryOperator};
use super::{
    assembly::{Instruction, Operand},
    settings::Target,
};

pub struct CodeEmitter {
    platform: Target,
    file: File,
}

impl CodeEmitter {
    pub fn new(platform: Target, filename: &str) -> Result<Self> {
        let file = File::create(filename)?;
        Ok(Self { platform, file })
    }

    fn show_operand(&self, operand: &Operand) -> String {
        match operand {
            Operand::Reg(Reg::AX) => "%eax".to_string(),
            Operand::Reg(Reg::CX) => "%ecx".to_string(),
            Operand::Reg(Reg::DX) => "%edx".to_string(),
            Operand::Reg(Reg::R10) => "%r10d".to_string(),
            Operand::Reg(Reg::R11) => "%r11d".to_string(),
            Operand::Imm(i) => format!("${}", i),
            Operand::Stack(i) => format!("{}(%rbp)", i),
            // printing out pseudoregister is only for debugging
            Operand::Pseudo(name) => format!("%{}", name),
        }
    }

    fn show_byte_operand(&self, operand: &Operand) -> String {
        match operand {
            Operand::Reg(Reg::AX) => "%al".to_string(),
            Operand::Reg(Reg::CX) => "%cl".to_string(),
            Operand::Reg(Reg::DX) => "%dl".to_string(),
            Operand::Reg(Reg::R10) => "%r10b".to_string(),
            Operand::Reg(Reg::R11) => "%r11b".to_string(),
            other => self.show_operand(other),
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

    fn show_unary_instruction(&self, operator: &UnaryOperator) -> String {
        match operator {
            UnaryOperator::Neg => "negl".to_string(),
            UnaryOperator::Not => "notl".to_string(),
        }
    }

    fn show_binary_instruction(&self, operator: &BinaryOperator) -> String {
        match operator {
            BinaryOperator::Add => "addl".to_string(),
            BinaryOperator::Sub => "subl".to_string(),
            BinaryOperator::Mult => "imull".to_string(),
            BinaryOperator::And => "andl".to_string(),
            BinaryOperator::Or => "orl".to_string(),
            BinaryOperator::Xor => "xorl".to_string(),
            BinaryOperator::Sal => "sall".to_string(),
            BinaryOperator::Sar => "sarl".to_string(),
        }
    }

    fn show_cond_code(&self, cond_code: &CondCode) -> String {
        match cond_code {
            CondCode::E => "e".to_string(),
            CondCode::NE => "ne".to_string(),
            CondCode::G => "g".to_string(),
            CondCode::GE => "ge".to_string(),
            CondCode::L => "l".to_string(),
            CondCode::LE => "le".to_string(),
        }
    }

    fn emit_instruction(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Mov(src, dst) => writeln!(
                self.file,
                "\tmovl {}, {}",
                self.show_operand(&src),
                self.show_operand(&dst)
            ),
            Instruction::Unary(operator, dst) => writeln!(
                self.file,
                "\t{} {}",
                self.show_unary_instruction(operator),
                self.show_operand(dst)
            ),
            Instruction::Binary {
                op: op @ BinaryOperator::Sal | op @ BinaryOperator::Sar,
                src,
                dst,
            } => writeln!(
                self.file,
                "\t{} {}, {}",
                self.show_binary_instruction(op),
                self.show_byte_operand(src),
                self.show_operand(dst)
            ),
            Instruction::Binary { op, src, dst } => writeln!(
                self.file,
                "\t{} {}, {}",
                self.show_binary_instruction(op),
                self.show_operand(src),
                self.show_operand(dst)
            ),
            Instruction::Cmp(src, dst) => writeln!(
                self.file,
                "\tcmpl {}, {}",
                self.show_operand(src),
                self.show_operand(dst),
            ),
            Instruction::Idiv(operand) => {
                writeln!(self.file, "\tidivl {}", self.show_operand(operand))
            }
            Instruction::Cdq => writeln!(self.file, "\tcdq"),
            Instruction::Jmp(lbl) => writeln!(self.file, "\tjmp {}", self.show_local_label(lbl)),
            Instruction::JmpCC(code, lbl) => writeln!(
                self.file,
                "\tj{} {}",
                self.show_cond_code(code),
                self.show_local_label(lbl),
            ),
            Instruction::SetCC(code, operand) => writeln!(
                self.file,
                "\tset{} {}",
                self.show_cond_code(code),
                self.show_byte_operand(operand),
            ),
            Instruction::Label(lbl) => writeln!(self.file, "{}:", self.show_local_label(lbl)),
            Instruction::AllocateStack(i) => writeln!(self.file, "\tsubq ${}, %rsp", i),
            Instruction::Ret => writeln!(self.file, "\tmovq %rbp, %rsp\n\tpopq %rbp\n\tret"),
        }
    }

    fn emit_function(&mut self, function: &FunctionDefinition) -> Result<()> {
        let label = self.show_label(&function.name);
        writeln!(self.file, "\t.globl {}", label)?;
        writeln!(self.file, "{}:", label)?;
        writeln!(self.file, "\tpushq %rbp")?;
        writeln!(self.file, "\tmovq %rsp, %rbp")?;
        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        Ok(())
    }

    fn emit_stack_note(&mut self) -> Result<()> {
        match self.platform {
            Target::Linux => writeln!(self.file, "\t.section .note.GNU-stack,\"\",@progbits"),
            Target::Windows => writeln!(self.file, "\t.section .drectve"),
            _ => Ok(()),
        }
    }

    pub fn emit(&mut self, program: &Program) -> Result<()> {
        self.emit_function(&program.function)?;
        self.emit_stack_note()?;
        Ok(())
    }
}
