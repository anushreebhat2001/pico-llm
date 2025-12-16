# SAE Monosemantic Feature Report (transformer_layer2) — Key Interpretations

This is a **readable, compact** interpretation pass over the features you pasted.  
Theme: many “features” at this layer behave like **token / formatting detectors** (quotes, periods, newlines, story-openers, pronouns), with a few **semi-semantic** clusters (objects, affect/quality adjectives).

---

## Quick clustering (what’s going on overall)

### 1) Formatting & punctuation (dominant)
- **Paragraph breaks / section transitions:** 1593, 1621 (very clean)
- **Sentence-final period:** 455, 520, 1463, 1673 (very overlapping)
- **Quoted dialogue punctuation:** 1818 (often opening quote), 300 (often `."` close-quote pattern), 26 (sometimes)

**Interesting note:** you’re seeing **feature redundancy**: multiple SAEs carve up the same simple token family (e.g., `.` and `\n\n`) with slightly different context preferences.

### 2) “Once upon a time…” story-starters (template detection)
- 1084 and 734 are near-duplicates: strong on **indefinite article `a`** in the classic opener: “Once upon a time, there was a …”
- 26 also overlaps via **“there”** in “Once upon a time there was …”

### 3) Reference / POV glue words
- 1200: pronoun **“it”** (reference tracking)
- 622: **his/her** + **looked/felt** (possessive + perception/affect framing)

### 4) Weakly semantic “storybook content”
- 152: props & narrative meta (“bucket”, “boat”, “drink”, “group”, “family”, plus “moral”, “End”)
- 1308: descriptive adjectives / states (“scared”, “friendly”, “tall”, “delicious”, “sweet”)
- 1275: intention / intensifiers (“wanted”, “so”, “too”, “get”, “higher”, “got”, “sang”)
- 1560: “kid-story vocabulary stems / subwords” (“pengu”, “whist”, “kittens”, “flowers”, plus common glue like “the/so/tried”)

---

## Feature-by-feature (key interpretations)

### Feature 1593 — **Paragraph break token (`\\n\\n`)**
- **Best interpretation:** detects **new paragraph / scene break**.
- **Evidence:** activations align with `<<<\n>>>` separators before new beats (“Mom says…”, “The End.”, etc.).
- **Why it’s interesting:** this is one of the cleanest “monosemantic” ones in your list.

### Feature 1621 — **Paragraph break token (`\\n\\n`), variant**
- **Best interpretation:** same as 1593, but likely **slightly different context weighting** (e.g., break after dialogue vs after narration).
- **Interesting:** redundancy with 1593 suggests the SAE learned multiple ways to represent the same formatting primitive.

---

### Feature 1084 — **Indefinite article `a` in story-openers**
- **Best interpretation:** `a` (especially in “Once upon a time, there was a …”).
- **Evidence:** nearly all top hits are `Once upon a time, there was<<< a>>> …`
- **Interesting:** extremely “token-level”—useful for studying how the model encodes **template phrases**.

### Feature 734 — **Indefinite article `a` (duplicate of 1084)**
- **Best interpretation:** same as 1084; likely a **redundant detector** for the same token/context.
- **Interesting:** redundancy can come from SAE capacity + slightly different training optima.

---

### Feature 26 — **“there” in story opener (+ some quote overlap)**
- **Primary:** detects **`there`** in “Once upon a time there was…”.
- **Secondary (polysemy):** sometimes fires on **dialogue quote punctuation** (e.g., around `"My name is Spirit,"`).
- **Interesting:** shows how a “mostly monosemantic” token feature can pick up a second hook if both appear in similar early-story contexts.

---

### Feature 1818 — **Dialogue opening quote / sentence boundary punctuation**
- **Primary:** **opening quote** in dialogue: `<<< ">>>Good job…`, `<<< ">>>We will call you…`
- **Secondary:** also hits **period** in narration: `red<<<.>>>`, `cheered<<<.>>>`
- **Interpretation:** “**speech / boundary marker**” rather than a semantic concept.
- **Interesting:** likely reflects how the tokenizer + training make quotes/periods co-occur with similar structural roles.

### Feature 300 — **Close-quote pattern (`."`)**
- **Best interpretation:** detects **`."`-style punctuation at end of quoted speech**.
- **Evidence:** examples consistently show `dead<<<.">>>`, `Lily<<<.">>>`, etc.
- **Interesting:** a neat example of the SAE isolating a *very specific orthographic pattern*.

---

### Feature 455 — **Sentence-final period (`.`), narration-biased**
- **Best interpretation:** detects **`.`** at sentence ends in ordinary narration.
- **Interesting:** overlaps heavily with 520/1463/1673; differences are likely subtle (e.g., “middle of story” vs “wrap-up” contexts).

### Feature 520 — **Sentence-final period (`.`), closure-biased**
- **Best interpretation:** also a **`.`** detector, but many hits look like **end-of-scene / end-of-story emotional closure** (“together.”, “trustworthy.”, etc.).
- **Interesting:** “same token, different *genre position*.”

### Feature 1463 — **Period after activity/description clauses**
- **Best interpretation:** another `.` detector; the examples cluster around “did X. Then…” action narration.
- **Interesting:** suggests SAEs can split punctuation into **contextual subtypes**.

### Feature 1673 — **Period in classic “Once there was…” narration**
- **Best interpretation:** period after “Once …” style narrative clauses and other sentence ends.
- **Interesting:** bridges “story template” + punctuation in one feature.

---

### Feature 1200 — **Pronoun reference: “it”**
- **Best interpretation:** detects **`it`** (coreference / referring back to an object).
- **Evidence:** multiple strong hits are explicit `<<< it>>>`.
- **Interesting:** useful if you’re probing how early layers track **object persistence** in kids’ stories.

### Feature 622 — **Possessive + perception frame (“his/her”, “looked”, “felt”)**
- **Best interpretation:** **character POV glue**: possession + seeing/feeling verbs.
- **Evidence:** `<<< his>>>`, `<<< her>>>`, `<<< looked>>>`, `<<< felt>>>`.
- **Interesting:** closer to “semantic-ish” than punctuation; it’s about **narrative viewpoint mechanics**.

---

### Feature 152 — **Story props + moral/ending markers (polysemantic)**
- **What it seems to group:** concrete objects + social units + story meta:
  - props: **bucket, boat, drink**
  - social: **group, family**
  - meta: **moral, End**
- **Interpretation:** a “**children’s-story scaffold**” bundle more than one concept.
- **Interesting:** *not* monosemantic; this is a good example of a “theme feature” that spans multiple token families.

### Feature 1308 — **Affect/quality adjectives (semi-semantic)**
- **Likely core:** **descriptive state/quality words**: scared, friendly, tall, delicious, sweet.
- **Interpretation:** “**attribute / evaluation language**” in narration.
- **Interesting:** these words tend to appear in emotionally-guided kids’ narratives, so the model may compress them as a style vector.

### Feature 1275 — **Intent + intensifiers (semi-semantic, noisy)**
- **Likely core:** “**want/try/so/too/get**” style causal/intensity language.
- **Interpretation:** language of **goal pursuit** and **escalation** (“wanted… so… got… higher…”).
- **Interesting:** this is the kind of feature that can look scattered but still be coherent at the level of “narrative drive”.

### Feature 1560 — **Subword stems in kid-story vocabulary (noisy)**
- **Evidence includes:** `pengu` (penguin), `whist` (whistling), plus animals/objects (kittens, flowers) and high-frequency glue (“the”, “so”, “tried”).
- **Interpretation:** likely a **tokenization artifact**: grouping frequent **stems/subwords** common in this dataset.
- **Interesting:** helps diagnose when you’re seeing **BPE-shape features** rather than meaning.

---

### Feature 137 — **Beat transition markers (“When”, `\\n\\n`, `!`)**
- **Interpretation:** detects “**something just changed**” cues:
  - temporal connector: **When**
  - emphasis: **!**
  - structure: **paragraph break**
- **Interesting:** a nice example of a feature that encodes **narrative rhythm** rather than a noun/verb concept.

### Feature 748 — **Weak / unclear (likely glue around play/strength/dialogue)**
- **Observed:** `play`, `strong`, `I`, `what`, plus some encoding artifacts.
- **Best guess:** a **dialogue-ish + action-ish** mixture typical of kids’ stories.
- **Confidence:** low; needs more top-activations or negative examples to sharpen.

### Feature 343 — **Function-word / generic glue (low specificity)**
- **Observed:** all/it/on/at/to + generic story scaffolding.
- **Interpretation:** likely a **high-frequency background feature** (not meaningfully monosemantic).
- **Interesting:** these often appear when the SAE is forced to allocate capacity to ubiquitous tokens.

---

## Practical “so what?” for your analysis workflow
- If you want **meaningful monosemantic features**, prioritize ones like **1593/1621 (paragraph breaks)**, **1084/734 (story template article)**, **1200 (it)**, **300 (close-quote)**.
- Treat the cluster **455/520/1463/1673** as mostly “**period variants**” unless you’re explicitly studying formatting.
- For “cool semantic-ish hooks,” start from **622 (POV/perception)** and **1308 (affect/qualities)**.

