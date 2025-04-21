use crate::library::{
    ast::{Block, BlockItem, FunctionDefinition, Program, Statement},
    unique_ids::make_label,
};

pub struct LoopsLabeller {
    current_label: Option<String>,
}

impl LoopsLabeller {
    pub fn new() -> Self {
        LoopsLabeller {
            current_label: None,
        }
    }

    fn label_statement(&mut self, statement: Statement) -> Statement {
        match statement {
            Statement::Break(_) => {
                if let Some(label) = &self.current_label {
                    Statement::Break(label.clone())
                } else {
                    panic!("Break outside of loop")
                }
            }
            Statement::Continue(_) => {
                if let Some(label) = &self.current_label {
                    Statement::Continue(label.clone())
                } else {
                    panic!("Continue outside of loop")
                }
            }
            Statement::While {
                condition,
                body,
                id: _,
            } => {
                let saved_label = self.current_label.clone();
                let new_id = make_label("while");
                self.current_label = Some(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.current_label = saved_label;
                Statement::While {
                    condition,
                    body: labelled_body.into(),
                    id: new_id,
                }
            }
            Statement::DoWhile {
                body,
                condition,
                id: _,
            } => {
                let saved_label = self.current_label.clone();
                let new_id = make_label("do_while");
                self.current_label = Some(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.current_label = saved_label;
                Statement::DoWhile {
                    body: labelled_body.into(),
                    condition,
                    id: new_id,
                }
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                id: _,
            } => {
                let saved_label = self.current_label.clone();
                let new_id = make_label("for");
                self.current_label = Some(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.current_label = saved_label;
                Statement::For {
                    init,
                    condition,
                    post,
                    body: labelled_body.into(),
                    id: new_id,
                }
            }
            Statement::Compound(blk) => Statement::Compound(self.label_block(blk)),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition,
                then_clause: self.label_statement(*then_clause).into(),
                else_clause: else_clause.map(|stmt| self.label_statement(*stmt).into()),
            },
            Statement::Labelled { label, statement } => {
                let labelled_statement = self.label_statement(*statement);
                Statement::Labelled {
                    label,
                    statement: labelled_statement.into(),
                }
            }
            Statement::Null
            | Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Goto(_) => statement,
        }
    }

    fn label_block_item(&mut self, item: BlockItem) -> BlockItem {
        match item {
            BlockItem::S(s) => BlockItem::S(self.label_statement(s)),
            BlockItem::D(_) => item,
        }
    }

    fn label_block(&mut self, Block(b): Block) -> Block {
        Block(
            b.into_iter()
                .map(|item| self.label_block_item(item))
                .collect(),
        )
    }

    fn label_function_def(
        &mut self,
        FunctionDefinition { name, body }: FunctionDefinition,
    ) -> FunctionDefinition {
        FunctionDefinition {
            name,
            body: self.label_block(body),
        }
    }

    pub fn label_loops(&mut self, Program { function }: Program) -> Program {
        Program {
            function: self.label_function_def(function),
        }
    }
}
