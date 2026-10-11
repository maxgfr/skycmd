# skycmd-s1 training throughput

Measured by `sky-train bench` (mlx-rs 0.32 / MLX 0.32.2, Metal, compiled loss + gradient) on random tokens. Warmup steps are excluded; timing covers forward + backward + optimizer update (Muon on the encoder hidden matrices, AdamW elsewhere). `bf16` runs matmuls and activations in bfloat16 with f32 master weights and f32 losses. `tok/s` counts input tokens (batch x seq); `params` counts the trained model (encoder + MLM head for `mlm`, encoder + decision head for `decision`).

| model | params | phase | precision | batch | seq | step (ms) | tok/s | peak mem (MB) | steps |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| s1-pico | 152960 | mlm | bf16 | 128 | 128 | 24.3 | 673236 | 367 | 10+50 |
| s1-pico | 202305 | decision | bf16 | 64 | 192 | 20.5 | 598104 | 308 | 10+50 |
| s1-nano | 937216 | mlm | bf16 | 128 | 128 | 55.8 | 293518 | 1035 | 10+50 |
| s1-nano | 1134209 | decision | bf16 | 64 | 192 | 50.6 | 243002 | 926 | 10+50 |
| s1-micro | 5053952 | mlm | bf16 | 128 | 128 | 172.7 | 94883 | 1939 | 10+50 |
| s1-micro | 6630913 | decision | bf16 | 64 | 192 | 173.4 | 70881 | 1702 | 10+50 |
| s1-pico | 152960 | mlm | f32 | 128 | 128 | 28.4 | 576123 | 626 | 10+50 |
| s1-pico | 202305 | decision | f32 | 64 | 192 | 28.6 | 429542 | 603 | 10+50 |
| s1-nano | 937216 | mlm | f32 | 128 | 128 | 113.5 | 144362 | 1880 | 10+50 |
| s1-nano | 1134209 | decision | f32 | 64 | 192 | 97.9 | 125460 | 1805 | 10+50 |
| s1-micro | 5053952 | mlm | f32 | 128 | 128 | 287.5 | 56984 | 3507 | 10+50 |
| s1-micro | 6630913 | decision | f32 | 64 | 192 | 286.7 | 42857 | 3320 | 10+50 |
