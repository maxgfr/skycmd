# s6-route-nano: the run

Pre-registration: `docs/s6-nano2-prereg.md`, section 5. One run, seed 0; checkpoint `last`.

| phase | steps | working minutes | budget |
|---|---:|---:|---:|
| pilot nano | 3271 | 20.01 | 20 |
| pilot micro | 3191 | 20.01 | 20 |
| main | 20504 | 120.00 | 120 |

Brake at start: `SKY_DUTY SKY_COOL` = `0.5 45:15`, file `runs/brake`.

- `runs/s6-nano2-pilot/nano`: brake: duty 0.85, 50 min of work, 10 of rest (runs/brake) (2 times)
- `runs/s6-nano2-pilot/nano`: brake: duty 0.95, 55 min of work, 5 of rest (runs/brake) (2 times)
- `runs/s6-nano2-pilot/nano`: cooling down for N s (SKY_COOL) (1 times)
- `runs/s6-nano2-pilot/micro`: brake: duty 0.95, 55 min of work, 5 of rest (runs/brake) (2 times)
- `runs/s6-nano2-pilot/micro`: brake: duty 0.85, 50 min of work, 10 of rest (runs/brake) (1 times)
- `runs/s6-nano2`: brake: duty 0.95, 55 min of work, 5 of rest (runs/brake) (1 times)
- `runs/s6-nano2`: cooling down for N s (SKY_COOL) (3 times)
- `runs/s6-nano2`: brake: duty 0.85, 50 min of work, 10 of rest (runs/brake) (1 times)

## Pilots (validation slates, ternary, full depth)

| nano | micro | potion-cos | abort bar | size | abort |
|---:|---:|---:|---:|---|---|
| 0.5777972027972028 | 0.638986013986014 | 0.7255244755244755 | 0.5755244755244755 | micro | False |

tau_ask = 0.7293800711631775 (the smallest top probability at which validation accuracy among answered requests is >= 0.95 (section 3); 1.0 when none reaches it); at it on validation: {'tau': 0.7293800711631775, 'accuracy': 0.9501187648456056, 'coverage': 0.736013986013986}.
