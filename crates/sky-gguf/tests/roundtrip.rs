use sky_gguf::*;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn write_read_round_trip_all_kv_types_and_dtypes() {
    let mut w = GgufWriter::new();
    w.add_kv("general.architecture", "modern-bert")
        .add_kv("a.u8", 7u8)
        .add_kv("a.i8", -7i8)
        .add_kv("a.u16", 700u16)
        .add_kv("a.i16", -700i16)
        .add_kv("a.u32", 70_000u32)
        .add_kv("a.i32", -70_000i32)
        .add_kv("a.f32", 1.5f32)
        .add_kv("a.bool", true)
        .add_kv("a.u64", u64::MAX)
        .add_kv("a.i64", i64::MIN)
        .add_kv("a.f64", -2.25f64)
        .add_kv("a.str", "héllo")
        .add_kv("a.strs", OwnedValue::strings(&["Ġthe", "a b", ""]))
        .add_kv(
            "a.ints",
            OwnedValue::Array(vec![OwnedValue::I32(1), OwnedValue::I32(-2)]),
        )
        .add_kv(
            "a.nested",
            OwnedValue::Array(vec![
                OwnedValue::strings(&["x"]),
                OwnedValue::strings(&["y", "z"]),
            ]),
        )
        .add_kv("a.empty", OwnedValue::Array(vec![]));

    let mut seed = 1u64;
    let x: Vec<f32> = (0..64 * 3).map(|_| lcg(&mut seed)).collect();
    // PyTorch shape [3, 64] -> ggml dims [64, 3]
    for (name, ty) in [
        ("t.f32", GgmlType::F32),
        ("t.f16", GgmlType::F16),
        ("t.bf16", GgmlType::BF16),
        ("t.q8", GgmlType::Q8_0),
    ] {
        w.add_tensor_f32(name, &[64, 3], ty, &x).unwrap();
    }
    w.add_tensor_f32("t.vec", &[5], GgmlType::F32, &[1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    let bytes = w.to_bytes().unwrap();

    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.alignment, 32);
    assert_eq!(g.get_u64("general.alignment"), Some(32));
    assert_eq!(g.architecture(), Some("modern-bert"));
    assert_eq!(g.get("a.u8"), Some(&Value::U8(7)));
    assert_eq!(g.get("a.i8"), Some(&Value::I8(-7)));
    assert_eq!(g.get("a.u16"), Some(&Value::U16(700)));
    assert_eq!(g.get("a.i16"), Some(&Value::I16(-700)));
    assert_eq!(g.get("a.u32"), Some(&Value::U32(70_000)));
    assert_eq!(g.get("a.i32"), Some(&Value::I32(-70_000)));
    assert_eq!(g.get("a.f32"), Some(&Value::F32(1.5)));
    assert_eq!(g.get_bool("a.bool"), Some(true));
    assert_eq!(g.get("a.u64"), Some(&Value::U64(u64::MAX)));
    assert_eq!(g.get("a.i64"), Some(&Value::I64(i64::MIN)));
    assert_eq!(g.get_f64("a.f64"), Some(-2.25));
    assert_eq!(g.get_str("a.str"), Some("héllo"));
    assert_eq!(
        g.get_array("a.strs").unwrap().strings().unwrap(),
        vec!["Ġthe", "a b", ""]
    );
    assert_eq!(
        g.get_array("a.ints").unwrap().to_i64_vec().unwrap(),
        vec![1, -2]
    );
    let nested = g.get_array("a.nested").unwrap();
    assert_eq!(nested.elem_type(), ValueType::Array);
    let inner: Vec<Vec<&str>> = nested
        .iter()
        .map(|v| v.as_array().unwrap().strings().unwrap())
        .collect();
    assert_eq!(inner, vec![vec!["x"], vec!["y", "z"]]);
    assert!(g.get_array("a.empty").unwrap().is_empty());

    for t in &g.tensors {
        assert_eq!(
            (g.data_offset + t.offset) % 32,
            0,
            "{} is not aligned",
            t.name
        );
    }
    let info = g.tensor("t.f32").unwrap();
    assert_eq!(info.dims, vec![64, 3]);
    assert_eq!(info.shape(), vec![3, 64]);
    assert_eq!(g.tensor_f32("t.f32").unwrap(), x);
    let f16 = g.tensor_f32("t.f16").unwrap();
    let bf16 = g.tensor_f32("t.bf16").unwrap();
    let q8 = g.tensor_f32("t.q8").unwrap();
    for i in 0..x.len() {
        assert!((f16[i] - x[i]).abs() < 1e-3);
        assert!((bf16[i] - x[i]).abs() < 1e-2);
        assert!((q8[i] - x[i]).abs() < 1.0 / 127.0);
    }
    assert_eq!(g.tensor("t.q8").unwrap().size, 3 * 2 * 34);
    assert_eq!(
        g.tensor_f32("t.vec").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );

    // re-serializing the parsed file gives the same bytes
    let mut w2 = GgufWriter::new();
    for (k, v) in &g.kvs {
        w2.add_kv(*k, OwnedValue::from_value(v));
    }
    for t in &g.tensors {
        w2.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())
            .unwrap();
    }
    assert_eq!(w2.to_bytes().unwrap(), bytes);
}

#[test]
fn q8_0_error_bound() {
    let mut seed = 42u64;
    let x: Vec<f32> = (0..32 * 256)
        .map(|i| lcg(&mut seed) * (1.0 + (i % 7) as f32))
        .collect();
    let q = quantize_q8_0(&x);
    assert_eq!(q.len(), x.len() / 32 * 34);
    let y = dequantize(GgmlType::Q8_0, &q).unwrap();
    for (bx, by) in x.chunks(32).zip(y.chunks(32)) {
        let amax = bx.iter().fold(0f32, |m, v| m.max(v.abs()));
        // half a quantization step, plus the f16 rounding of the scale (relative 2^-11)
        let bound = amax / 127.0 * 0.5 + amax * (1.0 / 2048.0) + 1e-7;
        for (a, b) in bx.iter().zip(by) {
            assert!((a - b).abs() <= bound, "{a} vs {b} (bound {bound})");
        }
    }
    // zero blocks stay zero, extremes map to +-127
    let z = dequantize(GgmlType::Q8_0, &quantize_q8_0(&[0.0; 32])).unwrap();
    assert!(z.iter().all(|&v| v == 0.0));
    let mut e = [0.0f32; 32];
    e[3] = -2.0;
    e[4] = 1.0;
    let qe = quantize_q8_0(&e);
    assert_eq!(qe[2 + 3] as i8, -127);
    assert_eq!(qe[2 + 4] as i8, 64); // 63.5 rounds away from zero
}

#[test]
fn rejects_bad_input() {
    assert!(Gguf::parse(b"not a gguf").is_err());
    let mut w = GgufWriter::new();
    assert!(w.add_tensor("x", &[3], GgmlType::F32, vec![0; 8]).is_err());
    assert!(w
        .add_tensor_f32("q", &[16], GgmlType::Q8_0, &[0.0; 16])
        .is_err());
    w.add_tensor_f32("y", &[4], GgmlType::F32, &[0.0; 4])
        .unwrap();
    let bytes = w.to_bytes().unwrap();
    // the last 16 bytes are padding after the 16 data bytes
    for cut in [4, 20, bytes.len() - 17] {
        assert!(
            Gguf::parse(&bytes[..cut]).is_err(),
            "truncated at {cut} should fail"
        );
    }
}
