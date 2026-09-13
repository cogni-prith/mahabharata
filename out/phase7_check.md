# Phase 7 gate. The three query modes end to end

**Below the Phase 6 entry gate.** The model was trained on 1,845 triples against
a stated floor of 10,000, on instruction. This report shows the modes execute on
real data and returns their output verbatim. The rankings are model hypotheses,
labelled `predicted` and `uncited` on the page.

Entities 688, relations 9, stored dimension 400 float32 (200 complex). Training seed 20260917, UMAP seed 20260917.
Same data file and same scoring code as `out/index.html`.

## Mode 1. Nearest neighbours in vector space

- **Arjuna**: 1. Dharmaraja 0.3659, 2. Dhananjaya 0.3348, 3. Partha 0.3108, 4. Chekitana 0.3045, 5. Jishnu 0.2908, 6. Gudakesha 0.2763, 7. Abhimanyu 0.2731, 8. Phalguna 0.2444
- **Karna**: 1. Shalya 0.2762, 2. Gangeya 0.2734, 3. Jayadratha 0.2729, 4. Duryodhana 0.2575, 5. Radheya 0.2266, 6. Duhshasana 0.2209, 7. Bhishma 0.2166, 8. Satyaki 0.2001
- **Droupadi**: 1. Dhoumya 0.2695, 2. Jayadratha 0.2507, 3. Brihadbala 0.2307, 4. Dhrishtadyumna 0.2275, 5. Ghatotkacha 0.2208, 6. Pritha 0.2190, 7. Sairandhri 0.2056, 8. Vrisha 0.1903
- **Duryodhana**: 1. Shalya 0.3290, 2. Duhshasana 0.2647, 3. Suyodhana 0.2580, 4. Karna 0.2575, 5. Ashvatthama 0.2571, 6. Bhishma 0.2446, 7. Kouravya 0.2375, 8. Radheya 0.2162

## Mode 2. Link prediction, incomplete triple

- **(Duryodhana, foughtWith, ?)**: 1. Shini -2.7021, 2. Yuyudhana -2.7059, 3. Krishna -2.7559, 4. Yudhishthira -2.7724, 5. Satyaki -2.8269, 6. Madhava -2.8345, 7. Vrikodara -2.8491, 8. Ghatotkacha -2.8916. Of the top 8, 6 are already asserted, the rest are predicted and uncited.
- **(Arjuna, hasBiologicalParent, ?)**: 1. Vasava -2.8147, 2. Kunti -2.8259, 3. Yudhishthira -3.1492, 4. Krishna -3.1640, 5. Partha -3.2151, 6. Indra -3.2797, 7. Karna -3.2875, 8. Bhima -3.2923. Of the top 8, 2 are already asserted, the rest are predicted and uncited.
- **(Yudhishthira, speaksTo, ?)**: 1. Sanjaya -2.6875, 2. Yajnaseni -2.6920, 3. Pundarikaksha -2.7123, 4. Kshatta -2.7288, 5. Vidura -2.7567, 6. Lomasha -2.7657, 7. Arjuna -2.7658, 8. Bhimasena -2.7857. Of the top 8, 8 are already asserted, the rest are predicted and uncited.

## Mode 3. Analogy by vector arithmetic

- **Arjuna - Kunti + Gandhari**: 1. Jaya 0.2659, 2. Nriga 0.2480, 3. Sanatsujata 0.2396, 4. Vijaya 0.2298, 5. Partha 0.1944, 6. Sanjaya 0.1907, 7. Dhananjaya 0.1845, 8. Dharmaraja 0.1827. Duryodhana ranks 29 of 685.
- **Arjuna - Pandu + Dhritarashtra**: 1. Dharmaraja 0.3300, 2. Dhananjaya 0.2765, 3. Bhishma 0.2172, 4. Bhimasena 0.2130, 5. Partha 0.1825, 6. Nriga 0.1721, 7. Duhshasana 0.1632, 8. Dritarashtra 0.1632. Duryodhana ranks 49 of 685.
- **Yudhishthira - Kunti + Gandhari**: 1. Sanatsujata 0.2275, 2. Nriga 0.2021, 3. Maitreya 0.1879, 4. Jaya 0.1846, 5. Vijaya 0.1713, 6. Anshuman 0.1698, 7. Palita 0.1524, 8. Duryodhana 0.1405. Duryodhana ranks 8 of 685.

All three modes return ranked output.
The analogy demo is not working. Its best form here is `Yudhishthira - Kunti + Gandhari`, which puts Duryodhana at rank 8;
the other forms tried put it at 29 and 49.
A demo that has to land at rank 1 does not land at rank 8. That is the
expected consequence of training below the gate and it is not presented as working.
