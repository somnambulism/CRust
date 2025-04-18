use std::collections::HashMap;

use crate::library::{
    ast::{Block, BlockItem, FunctionDefinition, Program, Statement},
    unique_ids::make_label,
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

    fn resolve_labelled_statement(&mut self, statement: Statement) -> Statement {
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

            _ => statement,
        }
    }

    fn resolve_goto_statement(&mut self, statement: Statement) -> Statement {
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
            _ => statement,
        }
    }

    fn resolve_labelled_block_item(&mut self, item: BlockItem) -> BlockItem {
        if let BlockItem::S(s) = item {
            // resolving a statement does not change the variable map
            let resolved_s = self.resolve_labelled_statement(s);
            BlockItem::S(resolved_s)
        } else {
            item
        }
    }

    fn resolve_goto_block_item(&mut self, item: BlockItem) -> BlockItem {
        if let BlockItem::S(s) = item {
            // resolving a statement does not change the variable map
            let resolved_s = self.resolve_goto_statement(s);
            BlockItem::S(resolved_s)
        } else {
            item
        }
    }

    fn resolve_block(&mut self, Block(items): Block) -> Block {
        let resolved_labels_items: Vec<BlockItem> = items
            .into_iter()
            .map(|item| self.resolve_labelled_block_item(item))
            .collect();

        let resolved_items: Vec<BlockItem> = resolved_labels_items
            .into_iter()
            .map(|item| self.resolve_goto_block_item(item))
            .collect();

        Block(resolved_items)
    }

    fn resolve_function_def(
        &mut self,
        FunctionDefinition { name, body }: FunctionDefinition,
    ) -> FunctionDefinition {
        let resolved_body = self.resolve_block(body);

        FunctionDefinition {
            name,
            body: resolved_body,
        }
    }

    pub fn resolve(&mut self, program: Program) -> Program {
        Program {
            function: self.resolve_function_def(program.function),
        }
    }
}
