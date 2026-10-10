/* Phonetic key for romanised Sanskrit, Bengali and Hindi names.

   The same rule list as src/indic_phonetic.py, for the Lineage mode, which
   runs from file:// with no server and so cannot ask Python. Orjun, Orjuno,
   Arjun and Arjuna all key to "arjun"; Korno and Karna to "karn".

   A shared key is a lead, never an identity. lineage.js offers a single key
   match as a spelling reading and several as a set to choose from.

   src/test_phonetic.py runs this file under node and checks that every label
   gets the same key here as in Python, and that the guard lists agree. Edit
   both files together. */
(function (root) {
  'use strict';
  // Order matters. Each rule sees the output of the one above it.
  const RULES = [
    [/[^a-z ]/g, ""],
    [/ksh|kss|x/g, "ks"],
    [/chh|ch/g, "c"],
    [/sh/g, "s"],
    [/ph|f/g, "p"],
    [/([bcdgjkpt])h/g, "$1"],
    [/[vw]/g, "b"],
    [/z/g, "j"],
    [/q/g, "k"],
    [/(?<=[^aeiou ])y\b/g, "i"],
    [/^y|(?<= )y/g, "j"],
    [/(?<=[^aeiou ])y/g, "j"],
    [/(?<=[^aeiou ])ri(?=[^aeiou ])/g, "r"],
    [/ee|ii|ie|ey/g, "i"],
    [/oo|uu/g, "u"],
    [/au|ou|ow/g, "o"],
    [/ai|oi|ei/g, "e"],
    [/aa/g, "a"],
    [/o/g, "a"],
    [/(.)\1+/g, "$1"],
    [/a\b/g, ""],
  ];
  // See src/indic_phonetic.py for how CORPUS_WORDS is derived.
  const CORPUS_WORDS = `
    afab agneya away baab baad baba babb bada badd bath bbaaad bbab bbad
    bbed beda bhrit bind boar boat boats born both call camp darbha dies
    fail feet fell forth four fourth goad harsh hour jest kara maid main
    many marries monkey month part path pour rakshas rich road roam sabha
    saddhyas sand sang sank seed shall shook shot show song sorrow sorry
    soul sound south sudha summon swan tail tamas vadha vain wash wind
    without woman wood worn worth your
  `.split(/\s+/).filter(Boolean);
  const QUESTION_WORDS = `
    about also brother could data describe details does explain father
    from give have host husband info into killed know like list mean meant
    more most mother much name only related relation relationship should
    sister some story such summary tell than that their them then there
    these they this those upon very were what when where which whom whose
    wife will with would
  `.split(/\s+/).filter(Boolean);
  const GUARD = new Set(CORPUS_WORDS.concat(QUESTION_WORDS));
  const MIN_LENGTH = 4;

  const stripDiacritics = s => String(s).normalize('NFKD').replace(/[\u0300-\u036f]/g, '');

  function key(name) {
    let s = stripDiacritics(name).toLowerCase().replace(/-/g, ' ').replace(/\s+/g, ' ').trim();
    for (const [pat, rep] of RULES) s = s.replace(pat, rep);
    return s.replace(/\s+/g, ' ').trim();
  }

  /* The key to look a typed token up by, or null if it must not be tried:
     too short to tell names apart, or an English word whose key collides
     with a name ("part" and Partha). */
  function phoneticKey(token) {
    const t = stripDiacritics(token).toLowerCase().trim();
    if (t.length < MIN_LENGTH || GUARD.has(t)) return null;
    const k = key(t);
    return k.length >= 3 ? k : null;
  }

  const api = { key, phoneticKey, GUARD, MIN_LENGTH };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.IndicPhonetic = api;
})(typeof window !== 'undefined' ? window : this);
