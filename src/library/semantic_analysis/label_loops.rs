use std::collections::HashSet;

use crate::library::{
    ast::{Block, BlockItem, FunctionDeclaration, Program, Statement, SwitchCases},
    util::unique_ids::make_label,
};

pub struct LoopsLabeller {
    loop_label: Option<String>,
    switch_label: Option<String>,
    switch_cases: SwitchCases,
    break_stack: Vec<String>,
}

impl LoopsLabeller {
    pub fn new() -> Self {
        LoopsLabeller {
            loop_label: None,
            switch_label: None,
            switch_cases: HashSet::new(),
            break_stack: vec![],
        }
    }

    fn label_statement(&mut self, statement: Statement) -> Statement {
        match statement {
            Statement::Break(_) => {
                if let Some(label) = self.break_stack.last() {
                    Statement::Break(label.clone())
                } else {
                    panic!("Break outside of loop or switch")
                }
            }
            Statement::Continue(_) => {
                if let Some(label) = &self.loop_label {
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
                let saved_label = self.loop_label.clone();
                let new_id = make_label("while");
                self.loop_label = Some(new_id.clone());
                self.break_stack.push(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.break_stack.pop();
                self.loop_label = saved_label;
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
                let saved_label = self.loop_label.clone();
                let new_id = make_label("do_while");
                self.loop_label = Some(new_id.clone());
                self.break_stack.push(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.break_stack.pop();
                self.loop_label = saved_label;
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
                let saved_label = self.loop_label.clone();
                let new_id = make_label("for");
                self.loop_label = Some(new_id.clone());
                self.break_stack.push(new_id.clone());
                let labelled_body = self.label_statement(*body);
                self.break_stack.pop();
                self.loop_label = saved_label;
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
            Statement::Switch {
                condition, body, ..
            } => {
                let saved_label = self.switch_label.clone();
                let new_id = make_label("switch");
                self.switch_label = Some(new_id.clone());
                self.break_stack.push(new_id.clone());
                let saved_switch_cases = self.switch_cases.clone();
                self.switch_cases = HashSet::new();
                let labelled_body = self.label_statement(*body);
                self.break_stack.pop();
                self.switch_label = saved_label;
                let statement = Statement::Switch {
                    condition,
                    body: labelled_body.into(),
                    cases: self.switch_cases.clone(),
                    id: new_id,
                };
                self.switch_cases = saved_switch_cases;
                statement
            }
            Statement::Case {
                condition,
                body,
                switch_label: _,
            } => {
                let switch_label = self.switch_label.clone();
                if let Some(label) = &switch_label {
                    if self.switch_cases.contains(&Some(condition)) {
                        panic!("Duplicate case {} in switch", condition);
                    }
                    self.switch_cases.insert(Some(condition));
                    return Statement::Case {
                        condition,
                        body: self.label_statement(*body).into(),
                        switch_label: label.clone(),
                    };
                } else {
                    panic!("Case outside of switch");
                }
            }
            Statement::Default {
                body,
                switch_label: _,
            } => {
                let switch_label = self.switch_label.clone();
                if let Some(label) = &switch_label {
                    if self.switch_cases.contains(&None) {
                        panic!("Duplicate default case in switch");
                    }
                    self.switch_cases.insert(None);
                    return Statement::Default {
                        body: self.label_statement(*body).into(),
                        switch_label: label.clone(),
                    };
                } else {
                    panic!("Default outside of switch");
                }
            }
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

    fn label_function_def(&mut self, func: FunctionDeclaration) -> FunctionDeclaration {
        FunctionDeclaration {
            body: func.body.map(|body| self.label_block(body)),
            ..func
        }
    }

    pub fn label_loops(&mut self, Program(fn_defs): Program) -> Program {
        Program(
            fn_defs
                .into_iter()
                .map(|fn_def| self.label_function_def(fn_def))
                .collect(),
        )
    }
}
