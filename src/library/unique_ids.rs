use std::sync::atomic::{AtomicUsize, Ordering};

pub fn make_temporary() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tmp.{}", n)
}

pub fn make_label(prefix: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}.{}", prefix, n)
}
