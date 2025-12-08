use std::collections::HashSet;

use crate::library::{
    ast::block_items::{Block, BlockItem, Declaration, FunctionDeclaration, Program, Statement},
    ast::untyped_exp::{Exp, Initializer},
};

pub struct LabelsResolver {
    defined: HashSet<String>,
    used: HashSet<String>,
    fn_name: String,
}

impl LabelsResolver {
    pub fn new(fn_name: &str) -> Self {
        LabelsResolver {
            defined: HashSet::new(),
            used: HashSet::new(),
            fn_name: fn_name.to_string(),
        }
    }

    fn transform_lbl(&self, lbl: &str) -> String {
        format!("{}.{}", self.fn_name, lbl)
    }

    fn collect_labels_from_statement(
        &mut self,
        stmt: Statement<Initializer, Exp>,
    ) -> Result<Statement<Initializer, Exp>, String> {
        match stmt {
            Statement::Goto(lbl) => {
                self.used.insert(lbl.clone());
                Ok(Statement::Goto(self.transform_lbl(&lbl)))
            }
            Statement::LabelledStatement(lbl, inner) => {
                if self.defined.contains(&lbl) {
                    return Err(format!("Duplicate label: {}", lbl));
                }
                self.defined.insert(lbl.clone());
                let renamed_inner = self.collect_labels_from_statement(*inner)?;
                Ok(Statement::LabelledStatement(
                    self.transform_lbl(&lbl),
                    Box::new(renamed_inner),
                ))
            }
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => {
                let then_clause = Box::new(self.collect_labels_from_statement(*then_clause)?);
                let else_clause = match else_clause {
                    Some(stmt) => {
                        let s = self.collect_labels_from_statement(*stmt)?;
                        Some(Box::new(s))
                    }
                    None => None,
                };
                Ok(Statement::If {
                    condition,
                    then_clause,
                    else_clause,
                })
            }
            Statement::Compound(block) => {
                let block = self.collect_labels_from_block_items(block.0)?;
                Ok(Statement::Compound(block))
            }
            Statement::While {
                condition,
                body,
                id,
            } => {
                let body = Box::new(self.collect_labels_from_statement(*body)?);
                Ok(Statement::While {
                    condition,
                    body,
                    id,
                })
            }
            Statement::DoWhile {
                body,
                condition,
                id,
            } => {
                let body = Box::new(self.collect_labels_from_statement(*body)?);
                Ok(Statement::DoWhile {
                    condition,
                    body,
                    id,
                })
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => {
                let body = Box::new(self.collect_labels_from_statement(*body)?);
                Ok(Statement::For {
                    init,
                    condition,
                    post,
                    body,
                    id,
                })
            }
            Statement::Switch {
                control,
                body,
                cases,
                id,
            } => {
                let body = Box::new(self.collect_labels_from_statement(*body)?);
                Ok(Statement::Switch {
                    control,
                    body,
                    cases,
                    id,
                })
            }
            Statement::Case(v, stmt, id) => {
                let stmt = Box::new(self.collect_labels_from_statement(*stmt)?);
                Ok(Statement::Case(v, stmt, id))
            }
            Statement::Default(stmt, id) => {
                let stmt = Box::new(self.collect_labels_from_statement(*stmt)?);
                Ok(Statement::Default(stmt, id))
            }
            Statement::Return(_)
            | Statement::Null
            | Statement::Expression(_)
            | Statement::Break(_)
            | Statement::Continue(_) => Ok(stmt),
        }
    }

    fn collect_labels_from_block_items(
        &mut self,
        items: Vec<BlockItem<Initializer, Exp>>,
    ) -> Result<Block<Initializer, Exp>, String> {
        let mut out = Vec::with_capacity(items.len());

        for item in items.into_iter() {
            match item {
                BlockItem::S(stmt) => {
                    let stmt = self.collect_labels_from_statement(stmt)?;
                    out.push(BlockItem::S(stmt));
                }
                BlockItem::D(decl) => {
                    out.push(BlockItem::D(decl));
                }
            }
        }

        Ok(Block(out))
    }
}

fn validate_labels_in_fun(
    fn_decl: FunctionDeclaration<Initializer, Exp>,
) -> Result<FunctionDeclaration<Initializer, Exp>, String> {
    match fn_decl.body {
        Some(block) => {
            let mut labels_resolver = LabelsResolver::new(&fn_decl.name);
            let renamed_block = labels_resolver.collect_labels_from_block_items(block.0)?;

            let undefined: HashSet<String> = labels_resolver
                .used
                .difference(&labels_resolver.defined)
                .cloned()
                .collect();
            if !undefined.is_empty() {
                let mut vec: Vec<String> = undefined.into_iter().collect();
                vec.sort();
                return Err(format!("Undefined labels: {}", vec.join(", ")));
            }

            Ok(FunctionDeclaration {
                body: Some(renamed_block),
                ..fn_decl
            })
        }
        None => Ok(fn_decl),
    }
}

fn validate_labels_in_decl(
    decl: Declaration<Initializer, Exp>,
) -> Result<Declaration<Initializer, Exp>, String> {
    match decl {
        Declaration::FunDecl(fd) => Ok(Declaration::FunDecl(validate_labels_in_fun(fd)?)),
        other => Ok(other),
    }
}

pub fn validate_labels(
    program: Program<Initializer, Exp>,
) -> Result<Program<Initializer, Exp>, String> {
    let mut out: Vec<Declaration<Initializer, Exp>> = Vec::with_capacity(program.0.len());
    for decl in program.0.into_iter() {
        out.push(validate_labels_in_decl(decl)?);
    }
    Ok(Program(out))
}
