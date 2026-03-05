# Feature Dictionary — exp_001_sae_v2/v3

| Unit | Label | Top tokens | Mean z | Note |
|---|---|---|---|---|
| 48 | temporal/occurrence | late, late, appears, sentence, return | 0.933 | late/appears in sentence |
| 253 | repetition/translation cue | twice, French, Spanish, hot, reversed | 0.930 | mix of “twice/translation/temp” |
| 198 | analogy scaffold | king, In, 1, What, The | 0.920 | analogy/question framing |
| 69 | list/translation | French, ',, 1, 1, list | 0.913 | list tokens + translation context |
| 131 | string operation | appears, ,, compute, reversed, word | 0.904 | compute/reverse wording |
| 111 | return/translate-capital | return, Trans, Trans, capital, capital | 0.902 | function-return + capital translation |
| 83 | country slot/blank | Germany, France, __, this, of | 0.871 | country names with blank |
| 187 | colon/analogy separator | :, the, :, analogy, : | 0.868 | colon separators in analogies |
| 84 | sorting/factorial-ish | as, sorted, that, orial, ab | 0.857 | sort/factorial suffix cues |
| 63 | math prompt preamble | France, the, Given, Given, numbers | 0.857 | “Given … numbers” scaffold |

## v3 — nya features (lager 5, dict 512, LayerNorm, 800 steg)

| Unit / kluster | Label | Top tokens (exempel) | Notering | Kommentar |
|---|---|---|---|---|
| 472 / 468 / 57 / 156 / 346 (subspace) | antonym/opposite | opposite, dark, cold, light, tall | decoder‑subspace snarare än en enda neuron | Använd som kluster vid patch; PCA visar gemensam riktning (antonym). |
| 132 / 133 | analogy pivot | as, like, colon contexts | Del av analogi‑pivot (“as … as”) | Aktiverar på relationsord; användbar för analogy patch. |
| 421 / 144 | polarity / temperature | hot, cold | Förändras med motsatspar | 421 ~ “hot”, 144 ~ “cold”; bra för kontrolltest av antonymklustret. |
| 212 / 360 / 279 | parentheses / structural marker | parentheses, open, close | Strukturell syntax | Markerar paren‑öppning/stängning; dyker upp i kodliknande text. |
| 396 / 478 / 217 | translation greeting | morning, good | Översättnings‑cue | Aktiveras på hälsningsfraser (“good morning”). |
| 137 / 87 | role pairs (president/queen) | president, queen | roll/royal slot | Del av analogi “X to Y” med titlar. |

*Tip*: När du jobbar med “opposite” ska du tänka feature‑space, inte enskild unit. Kombinera klustervektorerna (medel eller PCA‑komp) för patch/analys.
