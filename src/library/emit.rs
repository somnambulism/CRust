use std::fs::File;
use std::io::{Result, Write};

use super::assembly::{BinaryOperator, FunctionDefinition, Program, Reg, UnaryOperator};
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
            Operand::Reg(Reg::DX) => "%edx".to_string(),
            Operand::Reg(Reg::R10) => "%r10d".to_string(),
            Operand::Reg(Reg::R11) => "%r11d".to_string(),
            Operand::Imm(i) => format!("${}", i),
            Operand::Stack(i) => format!("{}(%rbp)", i),
            // printing out pseudoregister is only for debugging
            Operand::Pseudo(name) => format!("%{}", name),
        }
    }

    fn show_label(&self, name: &str) -> String {
        match self.platform {
            Target::OsX => format!("_{}", name),
            _ => name.to_string(),
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
            Instruction::Binary { op, src, dst } => writeln!(
                self.file,
                "\t{} {}, {}",
                self.show_binary_instruction(op),
                self.show_operand(src),
                self.show_operand(dst)
            ),
            Instruction::Idiv(operand) => {
                writeln!(self.file, "\tidivl {}", self.show_operand(operand))
            }
            Instruction::Cdq => writeln!(self.file, "\tcdq"),
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
        if self.platform == Target::Windows {
            writeln!(self.file, "\tcall __main")?;
        }
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
