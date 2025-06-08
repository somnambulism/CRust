use crate::library::{
    ast::{
        Block, BlockItem, Declaration, Exp, ForInit, FunctionDeclaration, Program, Statement,
        VariableDeclaration,
    },
    symbols::{Entry, SymbolTable},
    types::Type,
};

pub struct TypeChecker {
    pub symbol_table: SymbolTable,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            symbol_table: SymbolTable::new(),
        }
    }

    fn typecheck_exp(&mut self, exp: &Exp) {
        match exp {
            Exp::FunCall { f, args } => {
                let t = &self.symbol_table.get(f).t;
                match t {
                    Type::Int => {
                        panic!("Tried to use variable {} as function name", f);
                    }
                    Type::FunType { param_count } => {
                        if args.len() != *param_count {
                            panic!("Function {} called with wrong number of arguments", f);
                        } else {
                            args.iter().for_each(|arg| self.typecheck_exp(arg));
                        }
                    }
                }
            }
            Exp::Var(v) => {
                let t = &self.symbol_table.get(v).t;
                if let Type::FunType { .. } = t {
                    panic!("Tried to use function name {} as variable", v);
                }
            }
            Exp::Unary(_, inner) => self.typecheck_exp(&inner),
            Exp::Binary(_, e1, e2) => {
                self.typecheck_exp(&e1);
                self.typecheck_exp(&e2);
            }
            Exp::PostfixDecrement(inner)
            | Exp::PostfixIncrement(inner)
            | Exp::PrefixDecrement(inner)
            | Exp::PrefixIncrement(inner) => self.typecheck_exp(&inner),
            Exp::Assignment(lhs, rhs) | Exp::CompoundAssign(_, lhs, rhs) => {
                self.typecheck_exp(&lhs);
                self.typecheck_exp(&rhs);
            }
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => {
                self.typecheck_exp(&condition);
                self.typecheck_exp(&then_result);
                self.typecheck_exp(&else_result);
            }
            Exp::Constant(_) => (),
        }
    }

    fn typecheck_block(&mut self, b: &Block) {
        b.0.iter().for_each(|item| self.typecheck_block_item(item));
    }

    fn typecheck_block_item(&mut self, block_item: &BlockItem) {
        match block_item {
            BlockItem::S(s) => self.typecheck_statement(s),
            BlockItem::D(d) => self.typecheck_decl(d),
        }
    }

    fn typecheck_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Return(e) => self.typecheck_exp(e),
            Statement::Expression(e) => self.typecheck_exp(e),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&then_clause);
                if let Some(else_clause) = else_clause {
                    self.typecheck_statement(&else_clause);
                }
            }
            Statement::Switch {
                condition, body, ..
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&body);
            }
            Statement::Case { body, .. } => self.typecheck_statement(&body),
            Statement::Default { body, .. } => self.typecheck_statement(&body),
            Statement::Compound(block) => self.typecheck_block(block),
            Statement::While {
                condition, body, ..
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&body);
            }
            Statement::DoWhile {
                body, condition, ..
            } => {
                self.typecheck_statement(&body);
                self.typecheck_exp(condition);
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                ..
            } => {
                match init {
                    ForInit::InitDecl(d) => self.typecheck_var_decl(&d),
                    ForInit::InitExp(e) => {
                        if let Some(e) = e {
                            self.typecheck_exp(e);
                        }
                    }
                }
                if let Some(condition) = condition {
                    self.typecheck_exp(condition);
                }
                if let Some(post) = post {
                    self.typecheck_exp(post);
                }
                self.typecheck_statement(&body);
            }
            Statement::Labelled { statement, .. } => self.typecheck_statement(statement),
            Statement::Null | Statement::Break(_) | Statement::Continue(_) | Statement::Goto(_) => {
                ()
            }
        }
    }

    fn typecheck_decl(&mut self, decl: &Declaration) {
        match decl {
            Declaration::VarDecl(vd) => self.typecheck_var_decl(vd),
            Declaration::FunDecl(fd) => self.typecheck_fn_decl(fd),
        }
    }

    fn typecheck_var_decl(&mut self, var_decl: &VariableDeclaration) {
        let VariableDeclaration { name, init } = var_decl;

        self.symbol_table.add_var(name.as_str(), Type::Int);
        if let Some(init) = init {
            self.typecheck_exp(init);
        }
    }

    fn typecheck_fn_decl(&mut self, func_decl: &FunctionDeclaration) {
        let FunctionDeclaration { name, params, body } = func_decl;

        let fun_type = Type::FunType {
            param_count: params.len(),
        };
        let has_body = body.is_some();

        // helper closure to validate that current declaration is compatible with
        // prior one
        let check_against_previous = |entry: &Entry| {
            let Entry {
                t: prev_t,
                is_defined,
                ..
            } = entry;
            if prev_t != &fun_type {
                panic!("Redeclared function {} with a different type", name);
            } else if *is_defined && has_body {
                panic!("Defined body of function {} twice", name);
            } else {
                ()
            }
        };

        let old_decl = self.symbol_table.get_opt(name.as_str());
        if let Some(old_decl) = old_decl {
            check_against_previous(old_decl);
        }

        let already_defined = old_decl.map_or(false, |entry| entry.is_defined);

        self.symbol_table
            .add_fun(name.as_str(), fun_type, already_defined || has_body);

        if has_body {
            for param in params {
                self.symbol_table.add_var(&param, Type::Int)
            }
        }

        if let Some(body) = body {
            self.typecheck_block(body);
        }
    }

    pub fn typecheck(&mut self, program: &Program) {
        program
            .0
            .iter()
            .for_each(|fn_decl| self.typecheck_fn_decl(fn_decl));
    }
}
