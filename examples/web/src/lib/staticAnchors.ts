// Curated static anchor library for the synthetic dataset generator.
//
// All entries are hand-verified facts in well-defined domains. The
// synthetic generator's random subset of these replaces the runtime
// "categories + expansion" inference calls — much faster (1 call
// total instead of 3+), zero confabulation risk, broader domain
// coverage than the model can produce in one inference budget.
//
// Curation guidelines (when editing this file):
// - One short completion per row (≤ 6 words ideally).
// - Lead with a space in `completion` so the trained model has
//   the same leading-space pattern as the user's target.
// - Avoid culturally-contested facts (e.g. "first day of the week"
//   varies by locale — skip).
// - Avoid stale facts (population counts, current officeholders).
// - Color theory is split into THREE separate models because they
//   are not interchangeable; conflating them is the bug this
//   library was built to fix.
// - Categories exist purely for human readability. The generator
//   samples uniformly across the flattened ALL list.
//
// Last verified: 2026-05-25.

export interface AnchorRow {
    prompt: string;
    completion: string;
}

// ── World capitals ──────────────────────────────────────────────────
//
// Pick countries whose capitals haven't moved in living memory and
// aren't politically disputed. Skipped: Israel/Tel Aviv vs Jerusalem,
// Bolivia (two capitals), Myanmar (moved to Naypyidaw 2005 — too
// recent for some training corpora to know).
const CAPITALS: AnchorRow[] = [
    { prompt: "What is the capital of France?", completion: " Paris." },
    { prompt: "What is the capital of Germany?", completion: " Berlin." },
    { prompt: "What is the capital of Spain?", completion: " Madrid." },
    { prompt: "What is the capital of Italy?", completion: " Rome." },
    { prompt: "What is the capital of Japan?", completion: " Tokyo." },
    { prompt: "What is the capital of China?", completion: " Beijing." },
    { prompt: "What is the capital of Russia?", completion: " Moscow." },
    { prompt: "What is the capital of the United Kingdom?", completion: " London." },
    { prompt: "What is the capital of the United States?", completion: " Washington, D.C." },
    { prompt: "What is the capital of Canada?", completion: " Ottawa." },
    { prompt: "What is the capital of Australia?", completion: " Canberra." },
    { prompt: "What is the capital of Brazil?", completion: " Brasília." },
    { prompt: "What is the capital of India?", completion: " New Delhi." },
    { prompt: "What is the capital of Egypt?", completion: " Cairo." },
    { prompt: "What is the capital of Mexico?", completion: " Mexico City." },
];

// ── Arithmetic ──────────────────────────────────────────────────────
//
// Single-digit and small two-digit operations. Avoid prompts with
// dependent context (no "what's twice that" etc).
const ARITHMETIC: AnchorRow[] = [
    { prompt: "What is 2 plus 2?", completion: " Four." },
    { prompt: "What is 3 plus 4?", completion: " Seven." },
    { prompt: "What is 5 plus 5?", completion: " Ten." },
    { prompt: "What is 7 plus 8?", completion: " Fifteen." },
    { prompt: "What is 10 minus 3?", completion: " Seven." },
    { prompt: "What is 12 minus 5?", completion: " Seven." },
    { prompt: "What is 20 minus 7?", completion: " Thirteen." },
    { prompt: "What is 2 times 3?", completion: " Six." },
    { prompt: "What is 4 times 5?", completion: " Twenty." },
    { prompt: "What is 6 times 7?", completion: " Forty-two." },
    { prompt: "What is 8 times 9?", completion: " Seventy-two." },
    { prompt: "What is 10 divided by 2?", completion: " Five." },
    { prompt: "What is 20 divided by 4?", completion: " Five." },
    { prompt: "What is 100 divided by 10?", completion: " Ten." },
    { prompt: "What is 81 divided by 9?", completion: " Nine." },
];

// ── Units of measurement ────────────────────────────────────────────
//
// Closed-form conversions only — nothing that depends on the user's
// locale or measurement system convention.
const UNITS: AnchorRow[] = [
    { prompt: "How many centimeters are in one meter?", completion: " One hundred." },
    { prompt: "How many meters are in one kilometer?", completion: " One thousand." },
    { prompt: "How many milliliters are in one liter?", completion: " One thousand." },
    { prompt: "How many grams are in one kilogram?", completion: " One thousand." },
    { prompt: "How many minutes are in one hour?", completion: " Sixty." },
    { prompt: "How many seconds are in one minute?", completion: " Sixty." },
    { prompt: "How many hours are in one day?", completion: " Twenty-four." },
    { prompt: "How many inches are in one foot?", completion: " Twelve." },
    { prompt: "How many feet are in one yard?", completion: " Three." },
    { prompt: "How many feet are in one mile?", completion: " 5,280." },
    { prompt: "How many items are in one dozen?", completion: " Twelve." },
    { prompt: "How many milligrams are in one gram?", completion: " One thousand." },
];

// ── Calendar / time ─────────────────────────────────────────────────
//
// Excluded: "first day of the week" (Sunday in US, Monday in ISO
// 8601 / most of Europe) — culturally contested.
const CALENDAR: AnchorRow[] = [
    { prompt: "How many days are in a week?", completion: " Seven." },
    { prompt: "How many months are in a year?", completion: " Twelve." },
    { prompt: "How many days are in February in a common year?", completion: " Twenty-eight." },
    { prompt: "How many days are in February in a leap year?", completion: " Twenty-nine." },
    { prompt: "How many days are in January?", completion: " Thirty-one." },
    { prompt: "Which day comes after Monday?", completion: " Tuesday." },
    { prompt: "Which day comes before Friday?", completion: " Thursday." },
    { prompt: "Which month comes after March?", completion: " April." },
    { prompt: "Which month comes before December?", completion: " November." },
    { prompt: "What is the first month of the year?", completion: " January." },
];

// ── Solar system ────────────────────────────────────────────────────
//
// Pluto-as-planet status is a closed question since 2006 — answer is
// 8 planets. Standard high-school astronomy facts only.
const SOLAR_SYSTEM: AnchorRow[] = [
    { prompt: "What is the largest planet in the solar system?", completion: " Jupiter." },
    { prompt: "What is the smallest planet in the solar system?", completion: " Mercury." },
    { prompt: "Which planet is closest to the sun?", completion: " Mercury." },
    { prompt: "Which planet is farthest from the sun?", completion: " Neptune." },
    { prompt: "How many planets are in the solar system?", completion: " Eight." },
    { prompt: "Which planet is famous for its rings?", completion: " Saturn." },
    { prompt: "Which planet is known as the red planet?", completion: " Mars." },
    { prompt: "What planet do humans live on?", completion: " Earth." },
    { prompt: "What is Earth's natural satellite called?", completion: " The Moon." },
    { prompt: "What type of celestial body is the Sun?", completion: " A star." },
];

// ── Geometry ────────────────────────────────────────────────────────
const GEOMETRY: AnchorRow[] = [
    { prompt: "How many sides does a triangle have?", completion: " Three." },
    { prompt: "How many sides does a square have?", completion: " Four." },
    { prompt: "How many sides does a pentagon have?", completion: " Five." },
    { prompt: "How many sides does a hexagon have?", completion: " Six." },
    { prompt: "How many sides does an octagon have?", completion: " Eight." },
    { prompt: "What do the interior angles of a triangle sum to?", completion: " 180 degrees." },
    { prompt: "How many degrees is each interior angle of a square?", completion: " 90 degrees." },
    { prompt: "What 3D shape has six square faces?", completion: " A cube." },
    { prompt: "What 3D shape has one circular base and a single apex?", completion: " A cone." },
    { prompt: "How many degrees are in a full circle?", completion: " 360 degrees." },
];

// ── Color theory: ADDITIVE (RGB, used in screens and light) ─────────
//
// In an additive system, mixing primary lights INCREASES brightness.
// All three primaries at full = white. Used by displays, projectors,
// stage lighting. This is NOT the same as the RYB model that
// schoolchildren learn for painting.
const COLOR_RGB: AnchorRow[] = [
    { prompt: "What are the primary colors of the additive RGB color model?", completion: " Red, green, blue." },
    { prompt: "In the RGB additive color model, what color is produced by mixing all three primaries at full intensity?", completion: " White." },
    { prompt: "In the RGB additive color model, what color is produced by mixing red and green light?", completion: " Yellow." },
    { prompt: "In the RGB additive color model, what color is produced by mixing green and blue light?", completion: " Cyan." },
    { prompt: "In the RGB additive color model, what color is produced by mixing red and blue light?", completion: " Magenta." },
    { prompt: "Which color model is used by computer monitors and televisions?", completion: " RGB." },
    { prompt: "What is the absence of all RGB light called?", completion: " Black." },
    { prompt: "Is the RGB color model additive or subtractive?", completion: " Additive." },
];

// ── Color theory: SUBTRACTIVE (CMY/CMYK, used in printing) ──────────
//
// In a subtractive system, each ink SUBTRACTS wavelengths from
// reflected white light. All three primaries at full = (in theory)
// black — in practice CMYK adds a K (key/black) plate because the
// CMY mix is muddy brown.
const COLOR_CMY: AnchorRow[] = [
    { prompt: "What are the primary colors of the subtractive CMY color model?", completion: " Cyan, magenta, yellow." },
    { prompt: "What does the K stand for in CMYK?", completion: " Key (black)." },
    { prompt: "Which color model is used by color printers?", completion: " CMYK." },
    { prompt: "Is the CMYK color model additive or subtractive?", completion: " Subtractive." },
    { prompt: "In the subtractive CMY model, what color is produced by mixing cyan and yellow inks?", completion: " Green." },
    { prompt: "In the subtractive CMY model, what color is produced by mixing magenta and yellow inks?", completion: " Red." },
    { prompt: "In the subtractive CMY model, what color is produced by mixing cyan and magenta inks?", completion: " Blue." },
    { prompt: "Why is K added to CMY to make CMYK in printing?", completion: " To produce a true, dense black." },
];

// ── Color theory: TRADITIONAL ARTIST (RYB, painting / mixing) ───────
//
// The historical artist's model — taught in primary schools and used
// by paint manufacturers. NOT physically accurate (RGB and CMY are
// the modern correct models for light and ink) but useful as a
// pedagogical mixing wheel for opaque pigments.
const COLOR_RYB: AnchorRow[] = [
    { prompt: "What are the three primary colors in the traditional artist's RYB color wheel?", completion: " Red, yellow, blue." },
    { prompt: "In the RYB color wheel, what color do you get by mixing red and yellow paint?", completion: " Orange." },
    { prompt: "In the RYB color wheel, what color do you get by mixing yellow and blue paint?", completion: " Green." },
    { prompt: "In the RYB color wheel, what color do you get by mixing red and blue paint?", completion: " Purple." },
    { prompt: "What are the three secondary colors in the RYB color wheel?", completion: " Orange, green, purple." },
    { prompt: "Which color model is traditionally taught to artists for mixing paint?", completion: " RYB." },
    { prompt: "On the RYB color wheel, which color is opposite (complementary to) red?", completion: " Green." },
    { prompt: "On the RYB color wheel, which color is opposite (complementary to) yellow?", completion: " Purple." },
];

// ── Chemistry: element symbols ──────────────────────────────────────
const CHEMISTRY: AnchorRow[] = [
    { prompt: "What is the chemical symbol for water?", completion: " H2O." },
    { prompt: "What is the chemical symbol for oxygen?", completion: " O." },
    { prompt: "What is the chemical symbol for hydrogen?", completion: " H." },
    { prompt: "What is the chemical symbol for carbon?", completion: " C." },
    { prompt: "What is the chemical symbol for gold?", completion: " Au." },
    { prompt: "What is the chemical symbol for silver?", completion: " Ag." },
    { prompt: "What is the chemical symbol for iron?", completion: " Fe." },
    { prompt: "What is the chemical symbol for sodium?", completion: " Na." },
    { prompt: "What is the chemical symbol for nitrogen?", completion: " N." },
    { prompt: "What is the chemical symbol for helium?", completion: " He." },
];

// ── Word repetition ─────────────────────────────────────────────────
//
// Anti-collapse anchors. Teaches "when asked to repeat a word, just
// emit it" without leaking into other answer shapes. Mix of "say
// the word" / "repeat the word" / "echo the word" so the LoRA
// doesn't memorize one specific phrasing.
const WORD_REPEAT: AnchorRow[] = [
    { prompt: "Say the word apple.", completion: " Apple." },
    { prompt: "Say the word cat.", completion: " Cat." },
    { prompt: "Say the word dog.", completion: " Dog." },
    { prompt: "Say the word tree.", completion: " Tree." },
    { prompt: "Say the word book.", completion: " Book." },
    { prompt: "Repeat the word hello.", completion: " Hello." },
    { prompt: "Repeat the word world.", completion: " World." },
    { prompt: "Repeat the word sun.", completion: " Sun." },
    { prompt: "Echo the word water.", completion: " Water." },
    { prompt: "Echo the word moon.", completion: " Moon." },
];

// ── Roman numerals ──────────────────────────────────────────────────
const ROMAN: AnchorRow[] = [
    { prompt: "What number is the Roman numeral I?", completion: " One." },
    { prompt: "What number is the Roman numeral V?", completion: " Five." },
    { prompt: "What number is the Roman numeral X?", completion: " Ten." },
    { prompt: "What number is the Roman numeral L?", completion: " Fifty." },
    { prompt: "What number is the Roman numeral C?", completion: " One hundred." },
    { prompt: "What number is the Roman numeral D?", completion: " Five hundred." },
    { prompt: "What number is the Roman numeral M?", completion: " One thousand." },
    { prompt: "What number is the Roman numeral IV?", completion: " Four." },
    { prompt: "What number is the Roman numeral IX?", completion: " Nine." },
    { prompt: "What number is the Roman numeral XL?", completion: " Forty." },
];

// ── Language identification ─────────────────────────────────────────
//
// Greetings only — short, well-known across multiple languages, no
// ambiguous spellings (e.g. avoided "Hallo" which is both Dutch and
// informal German).
const LANGUAGE: AnchorRow[] = [
    { prompt: "What language is the greeting 'Bonjour' from?", completion: " French." },
    { prompt: "What language is the greeting 'Hola' from?", completion: " Spanish." },
    { prompt: "What language is the greeting 'Guten Tag' from?", completion: " German." },
    { prompt: "What language is the greeting 'Ciao' from?", completion: " Italian." },
    { prompt: "What language is the greeting 'Konnichiwa' from?", completion: " Japanese." },
    { prompt: "What language is the greeting 'Ni hao' from?", completion: " Mandarin Chinese." },
    { prompt: "What language is the greeting 'Privet' from?", completion: " Russian." },
    { prompt: "What language is the greeting 'Olá' from?", completion: " Portuguese." },
];

// ── Spelling ────────────────────────────────────────────────────────
//
// Hyphen-separated to make the answer a fixed-format pattern the
// LoRA can latch onto without ambiguity about whether to space out
// or concatenate the letters.
const SPELLING: AnchorRow[] = [
    { prompt: "How do you spell 'dog'?", completion: " D-O-G." },
    { prompt: "How do you spell 'cat'?", completion: " C-A-T." },
    { prompt: "How do you spell 'tree'?", completion: " T-R-E-E." },
    { prompt: "How do you spell 'book'?", completion: " B-O-O-K." },
    { prompt: "How do you spell 'sun'?", completion: " S-U-N." },
    { prompt: "How do you spell 'moon'?", completion: " M-O-O-N." },
    { prompt: "How do you spell 'house'?", completion: " H-O-U-S-E." },
    { prompt: "How do you spell 'water'?", completion: " W-A-T-E-R." },
    { prompt: "How do you spell 'ocean'?", completion: " O-C-E-A-N." },
    { prompt: "How do you spell 'river'?", completion: " R-I-V-E-R." },
];

// ── Flattened library ───────────────────────────────────────────────
//
// The generator samples uniformly from this concatenation. Per-
// category structure above is purely for human readability when
// reviewing or editing the file.
export const STATIC_ANCHOR_LIBRARY: ReadonlyArray<AnchorRow> = [
    ...CAPITALS,
    ...ARITHMETIC,
    ...UNITS,
    ...CALENDAR,
    ...SOLAR_SYSTEM,
    ...GEOMETRY,
    ...COLOR_RGB,
    ...COLOR_CMY,
    ...COLOR_RYB,
    ...CHEMISTRY,
    ...WORD_REPEAT,
    ...ROMAN,
    ...LANGUAGE,
    ...SPELLING,
];

/** Fisher-Yates partial shuffle — picks `n` distinct rows uniformly
 *  at random from `library` without disturbing the input array. */
export function sampleAnchors(
    library: ReadonlyArray<AnchorRow>,
    n: number,
): AnchorRow[] {
    const copy = library.slice();
    const take = Math.min(n, copy.length);
    const out: AnchorRow[] = [];
    for (let i = 0; i < take; i++) {
        const j = i + Math.floor(Math.random() * (copy.length - i));
        [copy[i], copy[j]] = [copy[j], copy[i]];
        out.push(copy[i]);
    }
    return out;
}
