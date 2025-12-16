# SAE Feature Audit Report — Transformer Layer 2 (Concise)

**Purpose:** Present the key findings from the provided SAE top-activation snippets by grouping features into **linguistic families** and labeling features as **monosemantic vs mixed**.

---

## Executive summary

- **Layer 2 is structure-heavy.** Most strong, clean features are **formatting/orthography** detectors (paragraph breaks, periods, quotes, story openers).
- **Some early “semantic-ish” signals exist** (pronoun/coreference, POV/perception framing), but they’re weaker and sometimes mixed.
- **Redundancy is common:** multiple features learn near-duplicates (e.g., multiple period variants, multiple opener variants).

---

## Feature families (cluster view)

### A) Document structure & punctuation (dominant)
- **Paragraph breaks / scene transitions:** 1593, 1621  
- **Sentence-final period variants (`.`):** 455, 520, 1463, 1673  
- **Dialogue punctuation:** 1818 (opening-quote-ish), 300 (close-quote `."`), 26 (minor overlap)

### B) Story-template detectors
- **Indefinite article in opener (“Once upon a time… there was a …”):** 1084, 734 (near-duplicate)
- **“there” in opener (“Once upon a time there was…”):** 26

### C) Reference tracking / POV glue (semi-semantic)
- **Pronoun coreference “it”:** 1200  
- **Possessive + perception/affect frame (“his/her”, “looked”, “felt”):** 622  

### D) Weak semantics / noisy bundles
- **Props + story meta bundle:** 152 (bucket/boat/drink/group/family + moral/End)
- **Affect/quality adjectives:** 1308 (scared/friendly/delicious/sweet/tall)
- **Intent/intensifiers:** 1275 (wanted/so/too/get/higher/got/sang)
- **Subword stems / tokenizer artifacts:** 1560 (pengu/whist/…)
- **Beat/transition cues:** 137 (When/!/newlines)
- **Low-specificity glue:** 343
- **Unclear mixed:** 748

---

## Feature catalogue (presentable labels)

### High-confidence monosemantic (clean triggers)
- **1593 — Paragraph break (`\n\n`) detector**
- **1621 — Paragraph break variant (`\n\n`)**
- **300 — Close-quote punctuation pattern (`."`)**
- **1084 — Story-opener indefinite article `a`**
- **734 — Story-opener `a` (duplicate/variant of 1084)**
- **26 — “there” in story opener** (with minor quote bleed)
- **1200 — Pronoun “it” detector**

### Medium-confidence (meaningful but context-dependent / overlapping)
- **1818 — Dialogue boundary marker** (opening quotes + some periods → “speech/boundary”)
- **455 / 520 / 1463 / 1673 — Period family** (`.`) split by context (wrap-up vs action narration vs template narration)
- **622 — POV glue** (possessives + “looked/felt” narration framing)
- **137 — Beat transition cues** (temporal connector + emphasis + structural breaks)

### Low-confidence / polysemantic / likely artifacts
- **152 — Kids-story scaffold bundle** (props + moral/End → theme feature, not single concept)
- **1308 — Attribute/affect field** (plausible coherence, needs more evidence)
- **1275 — Narrative drive field** (broad; needs more evidence)
- **1560 — Tokenizer/BPE-shaped stem feature**
- **343 — Background function-word glue**
- **748 — Unclear; needs more activations/negatives**

---

## Notable phenomena worth presenting

### 1) Redundancy / subtyping
Near-duplicates suggest the SAE split the same token family by **context**:
- **1084 ≈ 734** (opener `a`)
- **1593 ≈ 1621** (paragraph break)
- **455 / 520 / 1463 / 1673** (period variants)

### 2) Monosemantic ≠ semantic
The cleanest features here are **formatting primitives** (newline, periods, quotes). This is an important, valid finding: early layers often encode **text mechanics**.

### 3) Early narrative mechanics
Two useful semi-semantic hooks show up despite formatting dominance:
- **Coreference:** 1200 (“it”)
- **POV/perception framing:** 622 (“his/her”, “looked”, “felt”)

---

## Conclusions

1) **Layer 2 SAE features are mostly structural** (orthography + narrative scaffolding).  
2) **Redundancy is expected** and visible across multiple families.  
3) **Some narrative semantics appears** (reference + POV), but it’s not the majority at this layer.  
4) **Several features are polysemantic or tokenizer-shaped**, consistent with early-layer representations.
