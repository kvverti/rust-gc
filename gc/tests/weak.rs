use gc::{Gc, Weak, Trace, Finalize};


#[test]
fn round_trip() {
    let x = Gc::new(47);
    let y = Gc::downgrade(x.clone());

    assert!(y.upgrade().is_some(), "Weak is populated");
}

#[test]
fn clearing() {
    let x = Gc::new(47);
    let y = Gc::downgrade(x);

    gc::force_collect();

    assert!(y.upgrade().is_none(), "Weak has been cleared");
}

#[test]
fn ephemeron() {
    let key = Gc::new(1);
    let value = Weak::with_key(&key, 2);

    assert!(value.upgrade().is_some(), "Weak is accessible");

    drop(key);
    gc::force_collect();

    assert!(value.upgrade().is_none(), "Weak has been cleared");
}

#[test]
fn key_reference() {
    #[derive(Finalize, Trace)]
    struct Value {
        key: Gc<i32>,
    }

    let key = Gc::new(1);
    let value = Value { key: key.clone() };
    let value = Weak::with_key(&key, value);

    assert!(value.upgrade().is_some(), "Weak is accessible");

    drop(key);
    gc::force_collect();

    assert!(value.upgrade().is_none(), "Weak has been cleared");
}