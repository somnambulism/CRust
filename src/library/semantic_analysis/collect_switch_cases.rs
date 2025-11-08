use std::fmt;

use crate::library::{
    ast::{
        block_items::{Block, BlockItem, Declaration, FunctionDeclaration, Program, Statement},
        typed_exp::{InnerExp, TypedExp},
    },
    r#const::T,
    const_convert::const_convert,
    types::Type,
    util::unique_ids::make_label,
};

#[derive(Debug)]
pub enum SwitchAnalysisError {
    CaseOutsideSwitch,
    DuplicateCase(String),
    NonConstantCaseLabel,
}

impl fmt::Display for SwitchAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwitchAnalysisError::CaseOutsideSwitch => {
                write!(f, "Case/default found outside of a switch statement")
            }
            SwitchAnalysisError::DuplicateCase(msg) => {
                write!(f, "Duplicate case found: {}", msg)
            }
            SwitchAnalysisError::NonConstantCaseLabel => {
                write!(f, "Case label is not a constant expression")
            }
        }
    }
}

impl std::error::Error for SwitchAnalysisError {}

#[derive(Default)]
pub struct SwitchAnalyzer {
    pub ctx: Option<(Type, Vec<(Option<T>, String)>)>,
}

impl SwitchAnalyzer {
    pub fn new() -> Self {
        Self { ctx: None }
    }

    fn analyze_case_or_default(
        &mut self,
        key: Option<T>,
        lbl: &str,
        inner: Statement<TypedExp>,
    ) -> Result<(Statement<TypedExp>, String), SwitchAnalysisError> {
        let (switch_t, cases) = self
            .ctx
            .as_mut()
            .ok_or(SwitchAnalysisError::CaseOutsideSwitch)?;
        let key = key.map(|k| const_convert(&switch_t, &k));

        if cases.iter().any(|(k, _)| *k == key) {
            let msg = key
                .as_ref()
                .map(|k| format!("Duplicate cases: {:?}", k))
                .unwrap_or_else(|| "Duplicate default".into());
            return Err(SwitchAnalysisError::DuplicateCase(msg));
        }

        let case_id = make_label(lbl);
        cases.push((key.clone(), case_id.clone()));

        let new_inner = self.analyze_statement(inner)?;

        Ok((new_inner, case_id))
    }

    fn analyze_statement(
        &mut self,
        stmt: Statement<TypedExp>,
    ) -> Result<Statement<TypedExp>, SwitchAnalysisError> {
        match stmt {
            Statement::Default(inner, _) => {
                let (new_stmt, default_id) =
                    self.analyze_case_or_default(None, "default", *inner)?;
                Ok(Statement::Default(Box::new(new_stmt), default_id))
            }
            Statement::Case(expr, inner, _) => {
                let key = match &expr.e {
                    InnerExp::Constant(c) => Some(c.clone()),
                    _ => return Err(SwitchAnalysisError::NonConstantCaseLabel),
                };
                let (new_stmt, case_id) = self.analyze_case_or_default(key, "case", *inner)?;
                Ok(Statement::Case(expr, Box::new(new_stmt), case_id))
            }
            Statement::Switch {
                control, body, id, ..
            } => {
                let switch_t = &control.get_type();
                let old_ctx = self.ctx.take();
                self.ctx = Some((switch_t.clone(), Vec::new()));

                let new_body = self.analyze_statement(*body)?;
                let cases = self.ctx.take().map(|(_, v)| v).unwrap_or_default();

                self.ctx = old_ctx;
                Ok(Statement::Switch {
                    control,
                    body: Box::new(new_body),
                    id,
                    cases,
                })
            }
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => {
                let new_then = self.analyze_statement(*then_clause)?;
                let new_else = match else_clause {
                    Some(e) => Some(Box::new(self.analyze_statement(*e)?)),
                    None => None,
                };
                Ok(Statement::If {
                    condition,
                    then_clause: Box::new(new_then),
                    else_clause: new_else,
                })
            }
            Statement::Compound(block) => {
                let new_block = self.analyze_block(block)?;
                Ok(Statement::Compound(new_block))
            }
            Statement::While {
                condition,
                body,
                id,
            } => Ok(Statement::While {
                condition,
                body: Box::new(self.analyze_statement(*body)?),
                id,
            }),
            Statement::DoWhile {
                condition,
                body,
                id,
            } => Ok(Statement::DoWhile {
                condition,
                body: Box::new(self.analyze_statement(*body)?),
                id,
            }),
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => Ok(Statement::For {
                init,
                condition,
                post,
                body: Box::new(self.analyze_statement(*body)?),
                id,
            }),
            Statement::LabelledStatement(lbl, stmt_box) => Ok(Statement::LabelledStatement(
                lbl,
                Box::new(self.analyze_statement(*stmt_box)?),
            )),
            Statement::Return(_)
            | Statement::Null
            | Statement::Expression(_)
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Goto(_) => Ok(stmt),
        }
    }

    fn analyze_block_item(
        &mut self,
        item: BlockItem<TypedExp>,
    ) -> Result<BlockItem<TypedExp>, SwitchAnalysisError> {
        match item {
            BlockItem::S(stmt) => Ok(BlockItem::S(self.analyze_statement(stmt)?)),
            decl => Ok(decl),
        }
    }

    fn analyze_block(
        &mut self,
        Block(items): Block<TypedExp>,
    ) -> Result<Block<TypedExp>, SwitchAnalysisError> {
        let mut out = Vec::with_capacity(items.len());

        for item in items.into_iter() {
            out.push(self.analyze_block_item(item)?);
        }

        Ok(Block(out))
    }

    fn analyze_function_def(
        &mut self,
        fun_decl: FunctionDeclaration<TypedExp>,
    ) -> Result<FunctionDeclaration<TypedExp>, SwitchAnalysisError> {
        match fun_decl.body {
            Some(b) => {
                let blk = self.analyze_block(b)?;
                Ok(FunctionDeclaration {
                    body: Some(blk),
                    ..fun_decl
                })
            }
            None => Ok(fun_decl),
        }
    }

    fn analyze_decl(
        &mut self,
        d: Declaration<TypedExp>,
    ) -> Result<Declaration<TypedExp>, SwitchAnalysisError> {
        match d {
            Declaration::FunDecl(fd) => Ok(Declaration::FunDecl(self.analyze_function_def(fd)?)),
            other => Ok(other),
        }
    }

    pub fn analyze_program(
        &mut self,
        program: Program<TypedExp>,
    ) -> Result<Program<TypedExp>, SwitchAnalysisError> {
        let Program(decls) = program;
        let analyzed = decls
            .into_iter()
            .map(|d| self.analyze_decl(d))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Program(analyzed))
    }
}

pub fn analyze_switches(
    program: Program<TypedExp>,
) -> Result<Program<TypedExp>, SwitchAnalysisError> {
    let mut analyzer = SwitchAnalyzer::new();
    analyzer.analyze_program(program)
}
