use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::library::{
    r#const::{T, type_of_const},
    types::Type,
};

/*
 * Convert constant to an int64. If constant is smaller than int64 it will be
 * zero- or sign-extended to preserve value; if it's the same size we preserve
 * its representation.
 */
fn const_to_int64(c: &T) -> i64 {
    match c {
        T::ConstChar(c) => *c as i64,
        T::ConstUChar(uc) => *uc as i64,
        T::ConstInt(i) => *i as i64,
        T::ConstUInt(ui) => *ui as i64,
        T::ConstLong(l) => *l,
        T::ConstULong(ul) => *ul as i64,
        T::ConstDouble(d) => *d as i64,
    }
}

/*
 * Convert int64 to a constant. Preserve the value if possible and wrap modulo
 * the size of the target type otherwise.
 */
fn const_of_int64(v: i64, target_type: &Type) -> T {
    match target_type {
        Type::Char | Type::SChar => T::ConstChar(v as i8),
        Type::UChar => T::ConstUChar(v as u8),
        Type::Int => T::ConstInt(v as i32),
        Type::Long => T::ConstLong(v),
        Type::UInt => T::ConstUInt(v as u32),
        Type::ULong | Type::Pointer(_) => T::ConstULong(v as u64),
        Type::Double => T::ConstDouble(v as f64),
        Type::FunType { .. } | Type::Array { .. } => panic!(
            "Internal error: can't convert constant to non-scalae type {:?}",
            target_type
        ),
    }
}

pub fn const_convert(target_type: &Type, c: &T) -> T {
    if type_of_const(c) == *target_type {
        return c.clone();
    } else {
        match (target_type, c) {
            /*
             * Because some values in the range of both double and ulong are outside the
             * range of int64, we need to handle conversions between double and ulong as
             * special cases instead of converting through int64
             */
            (Type::Double, T::ConstULong(ul)) => {
                let z = BigInt::from(*ul);
                T::ConstDouble(z.to_f64().unwrap())
            }
            (Type::ULong, T::ConstDouble(d)) => {
                // Convert double to u64 by truncation toward zero.
                let as_u64 = *d as u64;
                T::ConstULong(as_u64)
            }
            _ => {
                /*
                 * Convert c to int64, then to target_type, to avoid exponential explosion
                 * of different cases. Conversion to int64 preserves value (except when
                 * converting from out-of-range ulong, where it preserves representation).
                 * Conversion from int64 to const wraps modulo const size.
                 */
                let as_int64 = const_to_int64(c);
                return const_of_int64(as_int64, target_type);
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn preserve_int_value() {
        let i = T::ConstInt(1000);
        assert_eq!(const_convert(&Type::Long, &i), T::ConstLong(1000));
    }

    #[test]
    fn preserve_negative_int_value() {
        let i = T::ConstInt(-1000);
        assert_eq!(const_convert(&Type::Long, &i), T::ConstLong(-1000));
    }

    #[test]
    fn preserve_long_value() {
        let l = T::ConstLong(200000);
        assert_eq!(const_convert(&Type::Int, &l), T::ConstInt(200000));
    }

    #[test]
    fn truncate_positive_long() {
        // l is 2^52 + 5
        let l = T::ConstLong(4503599627370501);
        assert_eq!(const_convert(&Type::Int, &l), T::ConstInt(5));
    }

    #[test]
    fn truncate_positive_long_to_negative() {
        // l is 2^52 - 5
        let l = T::ConstLong(4503599627370491);
        assert_eq!(const_convert(&Type::Int, &l), T::ConstInt(-5));
    }

    #[test]
    fn truncate_negative_long_to_zero() {
        // l is -2^33
        let l = T::ConstLong(-8589934592);
        assert_eq!(const_convert(&Type::Int, &l), T::ConstInt(0));
    }

    #[test]
    fn truncate_negative_long_to_negative() {
        // l is -2^33 - 100
        let l = T::ConstLong(-8589934692);
        assert_eq!(const_convert(&Type::Int, &l), T::ConstInt(-100));
    }

    #[test]
    fn trivial_uint_to_int() {
        let ui = T::ConstUInt(100);
        assert_eq!(const_convert(&Type::Int, &ui), T::ConstInt(100));
    }

    #[test]
    fn wrapping_uint_to_int() {
        let ui = T::ConstUInt(4294967200);
        assert_eq!(const_convert(&Type::Int, &ui), T::ConstInt(-96));
    }

    #[test]
    fn trivial_int_to_uint() {
        let i = T::ConstInt(1000);
        assert_eq!(const_convert(&Type::UInt, &i), T::ConstUInt(1000));
    }

    #[test]
    fn wrapping_int_to_uint() {
        let i = T::ConstInt(-1000);
        assert_eq!(const_convert(&Type::UInt, &i), T::ConstUInt(4294966296));
    }

    #[test]
    fn int_to_ulong() {
        let i = T::ConstInt(-10);
        assert_eq!(
            const_convert(&Type::ULong, &i),
            T::ConstULong(18446744073709551606)
        );
    }

    #[test]
    fn uint_to_long() {
        let ui = T::ConstUInt(4294967200);
        assert_eq!(const_convert(&Type::Long, &ui), T::ConstLong(4294967200));
    }

    #[test]
    fn long_to_uint() {
        let l = T::ConstLong(-9223372036854774574);
        assert_eq!(const_convert(&Type::UInt, &l), T::ConstUInt(1234));
    }

    #[test]
    fn ulong_to_int() {
        let ul = T::ConstULong(4294967200);
        assert_eq!(const_convert(&Type::Int, &ul), T::ConstInt(-96));
    }

    #[test]
    fn ulong_to_uint() {
        let ul = T::ConstULong(1152921506754330624);
        assert_eq!(const_convert(&Type::UInt, &ul), T::ConstUInt(2147483648));
    }
}
