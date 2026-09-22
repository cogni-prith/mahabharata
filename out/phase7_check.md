# Phase 7 explorer checks

**Below the Phase 6 entry gate.** The asserted graph contains 1444 triples; 1369 remain after symmetric deduplication.
These checks verify execution, not historical truth or predictive accuracy.

The tests import `out/kg-core.js`, the same code used by `out/index.html`.
RotatE: 200 complex dimensions; seed 20260917. UMAP seed 20260917.

## Natural-language queries

| Query | Result |
|---|---|
| Who fought Bhishma? | relation / Bhishma |
| Who was Karna killed by? | relation / Karna |
| Who is Arjuna’s father? | relation / Arjuna |
| Who are Dhritarashtra's children? | relation / Dhritarashtra |
| Who does Krishna speak to? | relation / Krishna |
| Who studied under Drona? | relation / Drona |
| Who rules Madra? | relation / Madra |
| Who is like Karna? | nearest / Karna |
| Who is Draupadi? | nearest / Droupadi |
| Who killed Abhimanyu? | contested / Abhimanyu |
| Who are Bhishma's enemies? | unavailable / Bhishma |
| What is the meaning of dharma? | error |

## Analogy ranks

| Expression | Duryodhana rank |
|---|---|
| Arjuna - Kunti + Gandhari | 78 |
| Arjuna - Pandu + Dhritarashtra | 11 |
| Yudhishthira - Kunti + Gandhari | 15 |

The analogy target is not forced into the results. Its scientific acceptance gate remains unmet.

## Graph integrity recheck

1190 displayed links; 1145 withheld source records.
Displayed links differ from the bundled model graph.
Generic parentage remains a source statement with unspecified kind, not a biological assertion.

## Browser verification

Run `.venv/bin/python src/test_viewer_browser.py` for offline Chrome rendering,
canvas pixels, point selection, orbit, dimension switching, queries, evidence,
export, and mobile/tablet/desktop layout. Screenshots and its report go to
`build/viewer_checks/`. This Node check does not substitute for a browser test.
