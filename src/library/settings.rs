#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Lex,
    Parse,
    Validate,
    Codegen,
    Tacky,
    Assembly,
    Executable,
}

#[derive(PartialEq)]
pub enum Target {
    OsX,
    Linux,
    Windows,
}

pub fn current_platform() -> Target {
    if cfg!(target_os = "macos") {
        Target::OsX
    } else if cfg!(target_os = "linux") {
        Target::Linux
    } else if cfg!(target_os = "windows") {
        Target::Windows
    } else {
        panic!("Unsupported platform");
    }
}
