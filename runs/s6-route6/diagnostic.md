# s6-route v6: the published files read blind (docs/s6-route6-prereg.md 9.1)

| file | system | tool-right | blind accuracy | c | 95% (slates) | none share | blind none gap | fewest-words share |
|---|---|---:|---:|---:|---|---:|---:|---:|
| v_tools | model part, B0 | 915 | 0.1257 | 0.1429 | [0.1043, 0.1472] | 0.2421 | +0.0085 | 0.1814 |
| v_tools | model part, B1 | 915 | 0.0951 | 0.1429 | [0.0761, 0.1143] | 0.0970 | -0.0176 | 0.1989 |
| v_tools | pair, B0 | 915 | 0.1104 | 0.1429 | [0.0906, 0.1310] | 0.8191 | +0.0461 | 0.1738 |
| v_tools | embedding part (potion-cos-4bit), B0 | 915 | 0.0995 | 0.1429 | [0.0823, 0.1203] | 0.8365 | +0.0133 | 0.1574 |
| v_tools_masked | model part, B0 | 915 | 0.0787 | 0.1429 | [0.0615, 0.0959] | 0.0000 | +0.0000 | 0.0612 |
| v_tools_masked | model part, B1 | 915 | 0.1093 | 0.1429 | [0.0899, 0.1298] | 0.8881 | -0.0021 | 0.1705 |
| v_tools_masked | pair, B0 | 915 | 0.0699 | 0.1429 | [0.0546, 0.0864] | 0.0000 | +0.0000 | 0.0579 |
| v_tools_masked | embedding part (potion-cos-4bit), B0 | 915 | 0.0918 | 0.1429 | [0.0739, 0.1121] | 0.8488 | +0.0471 | 0.1891 |
| f_tools | model part, B0 | 756 | 0.1310 | 0.1429 | [0.1078, 0.1563] | 0.2190 | -0.0225 | 0.1812 |
| f_tools | model part, B1 | 756 | 0.2632 | 0.1429 | [0.2316, 0.2945] | 0.0772 | +0.0423 | 0.2288 |
| f_tools | pair, B0 | 756 | 0.1402 | 0.1429 | [0.1158, 0.1658] | 0.9481 | +0.0119 | 0.1548 |
| f_tools | embedding part (potion-cos-4bit), B0 | 756 | 0.1442 | 0.1429 | [0.1207, 0.1708] | 0.9503 | +0.0093 | 0.1521 |
| f_tools_masked | model part, B0 | 756 | 0.0489 | 0.1429 | [0.0341, 0.0645] | 0.0000 | +0.0000 | 0.0370 |
| f_tools_masked | model part, B1 | 756 | 0.1944 | 0.1429 | [0.1676, 0.2219] | 0.9079 | -0.0304 | 0.1931 |
| f_tools_masked | pair, B0 | 756 | 0.0503 | 0.1429 | [0.0346, 0.0655] | 0.0000 | +0.0000 | 0.0516 |
| f_tools_masked | embedding part (potion-cos-4bit), B0 | 756 | 0.1627 | 0.1429 | [0.1368, 0.1884] | 0.9598 | +0.0304 | 0.1839 |
