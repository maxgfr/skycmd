//! Generator output validates against the decision-record contract.

use sky_games::gen::{generate, split_for_seed};
use sky_games::GAMES;
use sky_schema::{DecisionRecord, QType, Split};

#[test]
fn records_validate_and_targets_sum_to_one() {
    for game in GAMES {
        let (recs, stats) = generate(game, 600, 3).unwrap();
        assert_eq!(recs.len(), 600);
        assert_eq!(stats.records, 600);
        let mut secondary = 0;
        let mut french = 0;
        for r in &recs {
            r.validate()
                .unwrap_or_else(|e| panic!("{game}: {e}: {r:?}"));
            assert_eq!(r.src, format!("game:{game}"));
            let sum: f32 = r.target.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "{game}: {sum}");
            let budget = if game == "doom" { 30 } else { 24 };
            assert!(r.state.split_whitespace().count() <= budget);
            // JSONL round trip.
            let line = serde_json::to_string(r).unwrap();
            assert!(!line.contains('\n'));
            let back: DecisionRecord = serde_json::from_str(&line).unwrap();
            assert_eq!(&back, r);
            secondary += usize::from(r.t != QType::Choice);
            french += usize::from(
                r.ins.contains('é')
                    || r.ins.contains("le drone")
                    || r.ins.contains("Choisis")
                    || r.ins.contains("Quel"),
            );
        }
        let share = secondary as f64 / recs.len() as f64;
        assert!(
            (0.25..=0.35).contains(&share),
            "{game}: secondary share {share}"
        );
        assert!(french > 60, "{game}: only {french} French samples");
        assert!(stats.splits.iter().sum::<usize>() == 600);
        assert!(stats.report().contains(game));
    }
}

#[test]
fn generation_is_deterministic() {
    let (a, _) = generate("rescue", 200, 9).unwrap();
    let (b, _) = generate("rescue", 200, 9).unwrap();
    assert_eq!(a, b);
}

#[test]
fn seed_ranges_define_splits() {
    let (recs, stats) = generate("dodge", 3000, 0).unwrap();
    assert!(
        stats.splits[1] > 0 && stats.splits[2] > 0,
        "{:?}",
        stats.splits
    );
    assert!(recs.iter().any(|r| r.split == Split::Val));
    assert_eq!(split_for_seed(1_000_018), Split::Val);
}
