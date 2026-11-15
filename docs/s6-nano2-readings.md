# s6-nano v2 (s6-route-nano): the readings the builder took

The thirty-four places where `docs/s6-nano2-prereg.md` had to be read to be built, each with the
reading taken. They are accepted as written here by amendment 1 of that file, with its two
changes applied: every tool of a renamed slate is renamed (reading 13), and each tool option is
handed its BM25 score and rank (reading 27). Counts are those of the build of 15/11/2026; no
test file was read beyond counts.

1. **"When a source's request has no call, its slates are natural and hard, gold *none*"
   (4.4).** Applied to When2Call only, whose labels say that no tool fits. A route/tools request
   whose first reply calls no tool is dropped (75,194 glaive rows): many such replies ask the
   user for information that a tool then needs.
2. **When2Call "request_for_info → the called tool" (4.1).** Training: the only listed tool;
   else the tool the rejected response of the preference set calls, when it is listed; else the
   request is dropped (5,812 dropped). Test (`test/mcq`): its `target_tool`, else the tool its
   `tool_call` answer names. tool_call items take the name of their own call.
3. **Normalised name (4.2, item 1).** Lower-cased; the words are the runs of letters and
   digits, every other character a separator; one leading `get`, `fetch`, `retrieve`, `find`,
   `search`, `lookup` or `query` dropped when another word follows; the words joined by `_`. No
   split of camelCase.
4. **What is clustered (4.2).** The embedded text is the option text (`name: description`, the
   description cut as section 3 says). The union-find takes the tools of route/tools and of
   When2Call, so that the guard of item 6 covers every tool a training slate can show. A
   When2Call request is dropped when its gold is in a test cluster, or, for a training request,
   in a validation cluster; the tools of those clusters are removed from its slates. Intents are
   not clustered (none falls within the cluster rule of a test-cluster tool).
5. **The 85 / 5 / 10 split (4.2, item 3).** The first byte b of the hash, read as b / 256:
   below 0.85 train, below 0.90 validation, else test.
6. **"goes to the test side" (4.2, item 4).** Across every group and every test-only source:
   a normalised text present on several sides is kept on the most held-out one (test, then
   validation, then training) and dropped from the others. No request is moved from its split.
7. **The caps (4.2, item 5).** 30 per cluster applies to the training requests of route/tools
   and of When2Call; 10 per cluster to the test and also to the validation of route/tools,
   which is built like the test. "The first by sha256 of the text": of the request's text as
   stored, ties by request id.
8. **Source subsets.** Seal-Tools: its training file only. API-Bank: `lv1-api-train` and
   `lv2-api-train`; calls of `ToolSearcher` dropped (3,170) and calls of a tool the item does
   not list dropped (2,992); the request is the last user turn, which is often understood only
   with the dialogue before it.
9. **"First sentence" (section 3).** Up to the first `.`, `!` or `?` followed by a space or by
   the end of the text, then at most 30 words.
10. **Training distractors (4.4).** Never a tool of the gold's cluster, nor another tool the
    request calls. The tool options of a slate are shuffled, *none* always last. A slate of two
    options (the gold alone) has no removed copy.
11. **"The nearest tools" of a hard slate (4.4).** Nearest to the gold tool (potion-base-2M
    cosine between option texts), not to the request; for a request without a gold, its own
    tools, then their nearest tools in turn.
12. **The share of *none* (4.4).** When2Call's no-call slates alone are about 35% of its
    slates; the removed copies are dropped first, then no-call slates at random, down to 20%
    (14,453 training and 705 validation no-call slates dropped). Confirmed by amendment 1.
13. **Renamed slates (4.4; changed by amendment 1).** Every tool of the slate is renamed
    `tool_<k>`, the k distinct and drawn at random from 1 to 99; the gold's description is the
    teacher's rewrite (4.4), every other description untouched. (As first built only the gold
    was renamed, which gave the answer away.)
14. **Removed test slates (4.4, 7.1).** They keep 8 options: the gold is replaced by one more
    distractor of the same rule; T-metatool and the MASSIVE probe the same. The tool options of
    a test slate are shuffled, *none* last.
15. **Validation slates (4.4).** One test-like slate per validation request of route/tools
    (`v_tools`, 1,144); the calibration (`s2-calibrate`) and tau_ask are read on them, not on
    the trainer's validation records.
16. **The false-negative filter (4.5).** The teacher sees the request and the tools of its hard
    slate, numbered, without *none* and without being told which is right. When it does not name
    the right tool as able, the request is dropped with all its slates; each distractor it names
    is replaced by the next of the gold's 16 nearest tools not in the slate, without a second
    check.
17. **Rewrites (4.5).** One per gold tool of the training and validation requests of every
    group, intents included (14,263), where 4.5 says "each training tool".
18. **Banking77 names (4.3).** The dataset's label names (`visa_or_mastercard`), from the
    options of the decision file.
19. **The MASSIVE probe (7.1).** The held-out scenarios are datetime, email, play and
    recommendation. A test utterance with its intent, 6 random intents of all 60 and *none*;
    20% removed; every held-out test utterance is kept (848 English, 809 French).
20. **Real-looking keys (4.6, item 5).** A token is masked when it starts with a provider prefix
    (`sk-`, `AKIA`, `ghp_`, `AIza`, …) followed by at least 12 characters, or has at least 24
    characters of letters, digits and `-_+/` with digits, both cases and at least 4 bits of
    entropy per character. 103 tool schemas or descriptions hold one; 280 strings were masked in
    requests.
21. **The head window (section 3).** "An option shrunk below its own length" is read as the
    `head` share of `sky-train s2-truncation`, which counts any cut of the question and options.
    Measured per size with its own configuration and tokenizer; when no candidate meets 5%, the
    rule gives no window and the training refuses to start.
22. **Micro (section 2).** "The same with its own config": its tokenizer (`tok-s6-4096`) and the
    state window of its configuration (384).
23. **static-ft (section 6).** model2vec 0.10.0's `StaticModelForPairSimilarity` takes pairs and
    uses the other pairs of a batch as negatives; it has no way to pass hard negatives, so it is
    trained on the (request, gold option) pairs alone, on the CPU. Its byte budget counts the
    packed vectors and their scales, not the tokenizer.
24. **potion-cos-4bit (section 6).** Per token, a float16 scale and 4-bit codes in −7..7
    (1,003,952 bytes).
25. **pair-lr (section 6).** Its training set does not fit in memory: `partial_fit` over chunks
    with the stopping rule of `fit` (tol 1e-3 on the epoch's training loss, 5 epochs without
    improvement, at most 1,000) and a shuffle per epoch. Features are binary, three namespaces
    (characters of the request, of the option, word pairs) kept apart by column permutations,
    each row L2-normalised; weights stored as int8.
26. **name-prior (section 6).** Names counted over every training slate, the `tool_<k>` names
    of renamed slates included; all its probability on the chosen option.
27. **BM25 (section 6; extended by amendment 1).** bm25s 0.3.16, Lucene variant, its defaults
    (k1 1.5, b 0.75, English stopwords, scikit-learn's token pattern), the documents being the
    slate's own tool options and the request the query; *none* by a fitted threshold. By
    amendment 1 the same scores are handed to the model: for each tool option, its BM25 score
    (two decimals) and its rank in the slate, computed by one function the builder and the
    engine share (`sky_infer::route`, checked against a fixture written with bm25s). They go
    into the state, after the request, as one entry per tool option keyed by its name
    (`BM25: <name> score <s> rank <r>; …`), so that the options' texts, hence the cached
    encoding of a tool list, do not depend on the request. The baselines do not see them.
28. **The byte budget B (section 6).** B is the evaluated file's size, known after training; the
    rehearsal uses 715,136 bytes and the size-matched baselines are refitted to the trained
    file's size if it differs.
29. **theta for *none* (section 6).** The threshold of the highest validation accuracy; of equal
    accuracies, the lowest. The temperature minimises the validation NLL (a grid on log T, then a
    golden-section refinement).
30. **"The answer changes" (G2, 7.3).** Compared by option position: the twin slate puts a new
    distractor of the same rule in the gold's place and keeps every other option where it was.
    Compared by tool, the answer would always change.
31. **BFCL (7.1).** "v2 `multiple` and `irrelevance`" are the files `BFCL_v3_multiple` and
    `BFCL_v3_irrelevance` at the pinned commit (200 and 240 items).
32. **tool-routing-data (7.1).** Each tool with the first of its two descriptions; every query
    against all 10 tools and *none*.
33. **tau_ask (section 3).** When no threshold reaches 0.95 on validation, tau_ask is 1.0 (the
    router always asks).
34. **The data folder (section 5).** `data/route` is the data folder outside the repository
    (`skycmd-data/s6-route/build`, next to the checkout), as every data folder of this project.
    The When2Call test has 1,863 slates: 137 of its 2,000 items list no tool and have *none* as
    their answer, so no slate.
