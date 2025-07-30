use std::collections::HashMap;

use crate::library::{
    ast::block_items::{Block, BlockItem, Declaration, FunctionDeclaration, Program, Statement},
    ast::untyped_exp::Exp,
    util::unique_ids::make_label,
};

pub struct LabelsResolver {
    label_map: HashMap<String, String>,
}

impl LabelsResolver {
    pub fn new() -> Self {
        LabelsResolver {
            label_map: HashMap::new(),
        }
    }

    fn resolve_labelled_statement(&mut self, statement: Statement<Exp>) -> Statement<Exp> {
        match statement {
            Statement::Labelled { label, statement } => {
                if self.label_map.contains_key(&label) {
                    panic!("Duplicate label declaration");
                }

                // Generate a new label name and store it in the map
                let unique_name = make_label(&label);
                self.label_map.insert(label, unique_name.clone());

                Statement::Labelled {
                    label: unique_name,
                    statement: self.resolve_labelled_statement(*statement).into(),
                }
            }

            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition,
                then_clause: Box::new(self.resolve_labelled_statement(*then_clause)),
                else_clause: else_clause.map(|e| Box::new(self.resolve_labelled_statement(*e))),
            },

            Statement::Compound(block) => {
                let Block(items) = block;
                let resolved_items = items
                    .into_iter()
                    .map(|item| self.resolve_labelled_block_item(item))
                    .collect();
                let resolved_block = Block(resolved_items);
                Statement::Compound(resolved_block)
            }

            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => Statement::Switch {
                condition,
                body: Box::new(self.resolve_labelled_statement(*body)),
                cases,
                id,
            },
            Statement::Default { body, switch_label } => Statement::Default {
                body: Box::new(self.resolve_labelled_statement(*body)),
                switch_label,
            },

            Statement::DoWhile {
                body,
                condition,
                id,
            } => Statement::DoWhile {
                body: Box::new(self.resolve_labelled_statement(*body)),
                condition,
                id,
            },
            Statement::While {
                condition,
                body,
                id,
            } => Statement::While {
                condition,
                body: Box::new(self.resolve_labelled_statement(*body)),
                id,
            },
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => Statement::For {
                init,
                condition,
                post,
                body: Box::new(self.resolve_labelled_statement(*body)),
                id,
            },

            _ => statement,
        }
    }

    fn resolve_goto_statement(&mut self, statement: Statement<Exp>) -> Statement<Exp> {
        match statement {
            Statement::Goto(label) => {
                let resolved_label = self.label_map.get(&label).unwrap_or_else(|| {
                    panic!("Goto label not found: {}", label);
                });
                Statement::Goto(resolved_label.clone())
            }
            Statement::Labelled { label, statement } => Statement::Labelled {
                label,
                statement: Box::new(self.resolve_goto_statement(*statement)),
            },
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition,
                then_clause: Box::new(self.resolve_goto_statement(*then_clause)),
                else_clause: else_clause.map(|e| Box::new(self.resolve_goto_statement(*e))),
            },
            Statement::Compound(block) => {
                let Block(items) = block;
                let resolved_items = items
                    .into_iter()
                    .map(|item| self.resolve_goto_block_item(item))
                    .collect();
                let resolved_block = Block(resolved_items);
                Statement::Compound(resolved_block)
            }
            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => Statement::Switch {
                condition,
                body: Box::new(self.resolve_goto_statement(*body)),
                cases,
                id,
            },
            Statement::Case {
                condition,
                body,
                switch_label,
            } => Statement::Case {
                condition,
                body: Box::new(self.resolve_goto_statement(*body)),
                switch_label,
            },
            Statement::Default { body, switch_label } => Statement::Default {
                body: Box::new(self.resolve_goto_statement(*body)),
                switch_label,
            },
            Statement::DoWhile {
                body,
                condition,
                id,
            } => Statement::DoWhile {
                body: Box::new(self.resolve_goto_statement(*body)),
                condition,
                id,
            },
            Statement::While {
                condition,
                body,
                id,
            } => Statement::While {
                condition,
                body: Box::new(self.resolve_goto_statement(*body)),
                id,
            },
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => Statement::For {
                init,
                condition,
                post,
                body: Box::new(self.resolve_goto_statement(*body)),
                id,
            },
            _ => statement,
        }
    }

    fn resolve_labelled_block_item(&mut self, item: BlockItem<Exp>) -> BlockItem<Exp> {
        if let BlockItem::S(s) = item {
            // resolving a statement does not change the variable map
            let resolved_s = self.resolve_labelled_statement(s);
            BlockItem::S(resolved_s)
        } else {
            item
        }
    }

    fn resolve_goto_block_item(&mut self, item: BlockItem<Exp>) -> BlockItem<Exp> {
        if let BlockItem::S(s) = item {
            // resolving a statement does not change the variable map
            let resolved_s = self.resolve_goto_statement(s);
            BlockItem::S(resolved_s)
        } else {
            item
        }
    }

    fn resolve_block(&mut self, Block(items): Block<Exp>) -> Block<Exp> {
        let resolved_labels_items: Vec<BlockItem<Exp>> = items
            .into_iter()
            .map(|item| self.resolve_labelled_block_item(item))
            .collect();

        let resolved_items: Vec<BlockItem<Exp>> = resolved_labels_items
            .into_iter()
            .map(|item| self.resolve_goto_block_item(item))
            .collect();

        Block(resolved_items)
    }

    fn resolve_decl(&mut self, decl: Declaration<Exp>) -> Declaration<Exp> {
        match decl {
            Declaration::FunDecl(func) => {
                let resolved_body = func.body.map(|body| self.resolve_block(body));
                Declaration::FunDecl(FunctionDeclaration {
                    body: resolved_body,
                    ..func
                })
            }
            var_decl => var_decl,
        }
    }
}

pub fn resolve_labels(Program(fn_defs): Program<Exp>) -> Program<Exp> {
    let fn_defs = fn_defs
        .into_iter()
        .map(|fn_def| {
            let mut resolver = LabelsResolver::new();
            resolver.resolve_decl(fn_def)
        })
        .collect::<Vec<_>>();
    Program(fn_defs)
}
