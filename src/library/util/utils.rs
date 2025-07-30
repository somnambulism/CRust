pub mod string_util {
    pub fn chop_suffix(s: &str, n: usize) -> &str {
        &s[..s.len().saturating_sub(n)]
    }
}
