pub fn round_away_from_zero(n: isize, x: isize) -> isize {
    // We're assuming that n > 0
    assert!(n > 0, "n must be positive");

    let r = x % n;
    if r == 0 {
        x
    } else if x < 0 {
        x - n - r
    } else {
        x + n - r
    }
}
