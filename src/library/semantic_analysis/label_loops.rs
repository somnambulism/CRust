use crate::library::{
    ast::{
        block_items::{Block, BlockItem, Declaration, FunctionDeclaration, Program, Statement},
        untyped_exp::{Exp, Initializer},
    },
    util::unique_ids::make_label,
};

pub struct LoopsLabeller {
    current_break_id: Option<String>,
    current_continue_id: Option<String>,
}

impl LoopsLabeller {
    pub fn new() -> Self {
        LoopsLabeller {
            current_break_id: None,
            current_continue_id: None,
        }
    }

    fn label_statement(
        &mut self,
        statement: Statement<Initializer, Exp>,
    ) -> Statement<Initializer, Exp> {
        match statement {
            Statement::Break(_) => {
                if let Some(l) = &self.current_break_id {
                    Statement::Break(l.clone())
                } else {
                    panic!("Break outside of loop or switch")
                }
            }
            Statement::Continue(_) => {
                if let Some(l) = &self.current_continue_id {
                    Statement::Continue(l.clone())
                } else {
                    panic!("Continue outside of loop")
                }
            }
            Statement::While {
                condition,
                body,
                id: _,
            } => {
                let new_id = make_label("while");
                let old_continue_id = self.current_continue_id.clone();
                let old_break_id = self.current_break_id.clone();
                self.current_continue_id = Some(new_id.clone());
                self.current_break_id = Some(new_id.clone());
                let stmt = Statement::While {
                    condition,
                    body: self.label_statement(*body).into(),
                    id: new_id,
                };
                self.current_continue_id = old_continue_id;
                self.current_break_id = old_break_id;
                stmt
            }
            Statement::DoWhile {
                body,
                condition,
                id: _,
            } => {
                let new_id = make_label("do_while");
                let old_continue_id = self.current_continue_id.clone();
                let old_break_id = self.current_break_id.clone();
                self.current_continue_id = Some(new_id.clone());
                self.current_break_id = Some(new_id.clone());
                let stmt = Statement::DoWhile {
                    body: self.label_statement(*body).into(),
                    condition,
                    id: new_id,
                };
                self.current_continue_id = old_continue_id;
                self.current_break_id = old_break_id;
                stmt
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                id: _,
            } => {
                let new_id = make_label("for");
                let old_continue_id = self.current_continue_id.clone();
                let old_break_id = self.current_break_id.clone();
                self.current_continue_id = Some(new_id.clone());
                self.current_break_id = Some(new_id.clone());
                let stmt = Statement::For {
                    init,
                    condition,
                    post,
                    body: self.label_statement(*body).into(),
                    id: new_id,
                };
                self.current_continue_id = old_continue_id;
                self.current_break_id = old_break_id;
                stmt
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
            Statement::LabelledStatement(label, statement) => {
                let labelled_statement = self.label_statement(*statement);
                Statement::LabelledStatement(label, labelled_statement.into())
            }
            Statement::Default(stmt, id) => {
                Statement::Default(self.label_statement(*stmt).into(), id)
            }
            Statement::Case(v, stmt, id) => {
                Statement::Case(v, self.label_statement(*stmt).into(), id)
            }
            Statement::Switch {
                control,
                body,
                cases,
                ..
            } => {
                let new_break_id = make_label("switch");
                let old_break_id = self.current_break_id.clone();
                self.current_break_id = Some(new_break_id.clone());
                let laballed_body = self.label_statement(*body);
                self.current_break_id = old_break_id;
                Statement::Switch {
                    control,
                    body: laballed_body.into(),
                    id: new_break_id,
                    cases,
                }
            }
            Statement::Null
            | Statement::Return(_)
            | Statement::Expression(_)
            | Statement::Goto(_) => statement,
        }
    }

    fn label_block_item(
        &mut self,
        item: BlockItem<Initializer, Exp>,
    ) -> BlockItem<Initializer, Exp> {
        match item {
            BlockItem::S(s) => BlockItem::S(self.label_statement(s)),
            BlockItem::D(_) => item,
        }
    }

    fn label_block(&mut self, Block(b): Block<Initializer, Exp>) -> Block<Initializer, Exp> {
        Block(
            b.into_iter()
                .map(|item| self.label_block_item(item))
                .collect(),
        )
    }

    fn label_decl(&mut self, decl: Declaration<Initializer, Exp>) -> Declaration<Initializer, Exp> {
        match decl {
            Declaration::FunDecl(func) => {
                self.current_break_id = None;
                self.current_continue_id = None;
                Declaration::FunDecl(FunctionDeclaration {
                    body: func.body.map(|body| self.label_block(body)),
                    ..func
                })
            }
            var_decl => var_decl,
        }
    }

    pub fn label_loops(
        &mut self,
        Program(decls): Program<Initializer, Exp>,
    ) -> Program<Initializer, Exp> {
        Program(
            decls
                .into_iter()
                .map(|decl| self.label_decl(decl))
                .collect(),
        )
    }
}
