# Phase 7 gate. The three query modes end to end

**Below the Phase 6 entry gate.** The model was trained on 1694 of 1771 asserted triples,
77 removed as symmetric duplicates, against a stated floor of 10,000,
on instruction. This report shows the modes execute on
real data and returns their output verbatim. The rankings are model hypotheses,
labelled `predicted` and `uncited` on the page.

Entities 683, relations 9, stored dimension 400 float32 (200 complex). Training seed 20260917, UMAP seed 20260917.
Same data file and same scoring code as `out/index.html`.

## Mode 1. Nearest neighbours in vector space

- **Arjuna**: 1. Dharmaraja 0.3345, 2. Jishnu 0.2954, 3. Chekitana 0.2735, 4. Dhananjaya 0.2710, 5. Bibhatsu 0.2662, 6. Partha 0.2582, 7. Yudhishthira 0.2551, 8. Gudakesha 0.2418
- **Karna**: 1. Duryodhana 0.3485, 2. Shalya 0.2831, 3. Jayadratha 0.2808, 4. Bhishma 0.2691, 5. Shrutayudha 0.2657, 6. Shikhandi 0.2584, 7. Ashvatthama 0.2385, 8. Satyaki 0.2335
- **Droupadi**: 1. Nandini 0.3830, 2. Duhshala 0.3689, 3. Ulupi 0.3138, 4. Brihadbala 0.2665, 5. Vyasa 0.2650, 6. Samyamani 0.2622, 7. Pritha 0.2446, 8. Vishnu 0.1877
- **Duryodhana**: 1. Karna 0.3485, 2. Ashvatthama 0.2795, 3. Kripa 0.2643, 4. Shakuni 0.2363, 5. Duhshasana 0.2319, 6. Shrutayu 0.2290, 7. Shalya 0.2163, 8. Drona 0.1993

## Mode 2. Link prediction, incomplete triple

- **(Duryodhana, foughtWith, ?)**: 1. Shini -2.5174, 2. Yuyudhana -2.7023, 3. Sahadeva -2.7932, 4. Madri -2.8086, 5. Somadatta -2.8243, 6. Krishna -2.8422, 7. Madhava -2.8484, 8. Ghatotkacha -2.8901. Of the top 8, 5 are already asserted, the rest are predicted and uncited.
- **(Arjuna, hasBiologicalParent, ?)**: 1. Vasava -2.7835, 2. Bhima -3.0717, 3. Pandu -3.1636, 4. Drona -3.1639, 5. Partha -3.1727, 6. Yudhishthira -3.1781, 7. Bhimasena -3.2437, 8. Krishna -3.2504. Of the top 8, 1 are already asserted, the rest are predicted and uncited.
- **(Yudhishthira, speaksTo, ?)**: 1. Madhusudana -2.6088, 2. Pundarikaksha -2.6568, 3. Yajnaseni -2.6620, 4. Bhimasena -2.7511, 5. Vasudeva -2.7560, 6. Arjuna -2.7597, 7. Vidura -2.7656, 8. Suyodhana -2.7703. Of the top 8, 8 are already asserted, the rest are predicted and uncited.

## Mode 3. Analogy by vector arithmetic

- **Arjuna - Kunti + Gandhari**: 1. Vaishampayana 0.2608, 2. Kiriti 0.2393, 3. Brahma 0.2152, 4. Sanjaya 0.1952, 5. Gudakesha 0.1712, 6. Ghatotkacha 0.1692, 7. Kounteya 0.1619, 8. Yudhishthira 0.1477. Duryodhana ranks 48 of 680.
- **Arjuna - Pandu + Dhritarashtra**: 1. Maitreya 0.3332, 2. Dharmaraja 0.3137, 3. Pritha 0.2855, 4. Partha 0.2605, 5. Bhishma 0.2483, 6. Gangeya 0.2356, 7. Dhananjaya 0.2259, 8. Kounteya 0.2238. Duryodhana ranks 19 of 680.
- **Yudhishthira - Kunti + Gandhari**: 1. Vaishampayana 0.2483, 2. Kiriti 0.2208, 3. Sanjaya 0.2004, 4. Radheya 0.1991, 5. Vishoka 0.1747, 6. Kounteya 0.1622, 7. Brahma 0.1588, 8. Arjuna 0.1549. Duryodhana ranks 11 of 680.

All three modes return ranked output.
The analogy demo is not working. Its best form here is `Yudhishthira - Kunti + Gandhari`, which puts Duryodhana at rank 11;
the other forms tried put it at 48 and 19.
A demo that has to land at rank 1 does not land at rank 11. That is the
expected consequence of training below the gate and it is not presented as working.


## Mode 4. The page executes, and the question parser

The page script runs to completion outside a browser and answers on boot.

- `who killed Karna` → Karna slain by : 6 asserted, then the model’s ranking
- `who are Bhishma's enemies` → Bhishma fought : 21 asserted, then the model’s ranking
- `who is Arjuna’s father` → Arjuna parent : 2 asserted, then the model’s ranking
- `who are Dhritarashtra's children` → Dhritarashtra child : 11 asserted, then the model’s ranking
- `who does Krishna speak to` → Krishna speaks to : 17 asserted, then the model’s ranking
- `who studied under Drona` → Drona instructed : 1 asserted, then the model’s ranking
- `who rules Madra` → Madra ruled by : 4 asserted, then the model’s ranking
- `who is like Karna` → Nearest neighbours of Karna in vector space, cosine
- `who killed Abhimanyu` → Abhimanyu slain by : 0 asserted, then the model’s ranking
- `who is Droupadi` → Nearest neighbours of Droupadi in vector space, cosine
- `who is Draupadi` → not understood, reported as such
- `what is the meaning of dharma` → not understood, reported as such

10 of 12 questions parse to a predicate and an entity.
Questions outside the pattern list are refused with the reason, never answered by guess.
