**OBJECTIVE** (and a bit of context)

I built this project out of curiosity. Since GPT-generated content has become increasingly popular, I have always wondered how one can tell apart between content written by a human vs an LLM objectively. I do not intend to say one is better than the other, instead ask what makes human content more valuable. In terms of structure, there exist patterns which have allowed me to approach the idea from a mathematical standpoint. The tool I have built captures long-range correlations across the structure of a given text. I presently use five constraints, weighing their values together and determining whether a particular text exhibits more human-like persistence in structure or has mixed or anti-persistence (too much noise in the signal). These constraints are:
1.	Word presence (density or gap)
2.	Sentence-length series
3.	Function word indicator series
4.	Punctuation cadence (to determine overall structural rhythm)
5.	Semantic drift (sentence-to-sentence spacing)


Imagine the sample text as a time-series. In doing so, one can quantify sentences and keywords based on punctuation and cluttering (semantic drift, word presence), capturing structural rhythm and persistence. Initially, the idea was to just build a differentiator, but this will prove to be short-sighted as I have justified above. Instead, I am thinking of first benchmarking the comparisons to be able to definitively say what human-like persistence looks like in numbers across at least 25-30 (initially) samples. The comparison that this code does can be considered the first phase in the making of the actual tool, in that it first measures the differences and then accordingly benchmarks the constraints to give one an idea of human-like persistence.
Crucially, following calibration and benchmarking it using enough material, this tool can be used more like a quality-analysis tool: something that would analyse a given text and, based on the benchmarking, signal whether it possesses human-like global persistence or is jumpy throughout like a GPT text. Using GPT for content creation is not the problem, but how much we rely on it makes a difference to the overall structure, and thereby the intent and meaning. Therefore, for analysing cold, technical articles, this tool may not be useful (except for maybe the Context-Fit constraint which determines whether a given word has been used appropriately or not). However, for longer texts that require holding the readers’ attention, focus on narrative buildup, or simply ought to feel like it was written by a human, the tool can be used to validate some (or all) of these characteristics.
Having said that, I strongly advise against looking at numbers alone. They only show one perspective, but they can be helpful if you combine them with a more close-reading-like analysis of a text. The numbers should only be an indicator of whether or not a given text has human-like persistence, which is what content should be all about, and not random words thrown together to make sentence-level sense. I should like to say that any text written by a human with pure intent and research effort (academic or otherwise) should mathematically reveal these patterns which match the benchmarking, because these patterns may not be deliberate, but they are all inherent to textual buildup and persistence.
	Of course, there are many limitations. I approached this from a purely research-based methodology. For example, factoring in the weird em-dashes (––) that GPTs commonly use would perhaps make for a stronger signal than Punctuation_Fano, but at the moment I haven’t been able to come up with a way to capture those because the code breaks it down into smaller dashes. Another critical limitation is that it can only work for texts longer than 2.5k words (on average, about 100 sentences).

**METHODOLOGY**

The core approach treats text as a time-series signal where structural patterns can be mathematically quantified. Drawing from complexity science and fractal analysis, the tool measures long-range correlations (LRC) within texts using the Hurst exponent and Detrended Fluctuation Analysis (DFA). These methods, traditionally used in physics and finance to detect memory effects in complex systems, reveal whether patterns in text persist over long distances or quickly decay into randomness.

**Mathematical Foundation**

When we encode text into numerical series– whether tracking word occurrences, sentence lengths, or semantic distances– we create a signal that can be analysed for self-similarity across scales. A Hurst exponent above 0.5 indicates persistence (patterns tend to continue), while values below 0.5 suggest anti-persistence (patterns tend to reverse). Human writing, with its more , theoretically produces different correlation patterns than machine-generated text, which optimises locally without genuine long-term memory.
**
The Five Encoding Constraints
**
Each constraint captures a different aspect of textual structure:
1. Word Presence (Density/Gap) Tracks the occurrence patterns of specific target words throughout the text. The tool offers three modes:
•	OR mode: Binary presence (1 if target word appears, 0 otherwise)
•	Gap mode: Inter-arrival times between occurrences (log-transformed for stability)
•	Density mode: Windowed density using convolution with configurable window size
This reveals how consistently key terms are distributed—humans tend to cluster related concepts unconsciously, while AI might distribute them more uniformly.
2. Sentence-Length Series Measures the word count of each sentence sequentially. Human writers naturally vary sentence length for rhythm and emphasis, creating characteristic patterns. The persistence in these patterns reflects whether variation is structured (human-like) or more random (potentially AI-like).
3. Function Word Indicator Series Tracks the presence of function words (articles, prepositions, conjunctions) that form the grammatical skeleton of text. These words appear unconsciously in human writing with specific rhythms tied to cognitive processing, whereas AI might use them more formulaically based on training patterns.
4. Punctuation Cadence Counts punctuation marks per sentence to capture structural rhythm. Beyond the Hurst analysis, the tool calculates the Fano factor (variance-to-mean ratio) to measure burstiness. Human punctuation often reflects breathing patterns, emotional emphasis, and rhetorical structure, creating more variable, bursty patterns compared to AI's more regular distribution.
5. Semantic Drift Measures the cosine distance between consecutive sentence embeddings using a pre-trained transformer model (all-MiniLM-L6-v2). This captures how ideas flow and evolve—humans tend to circle back to themes with variations, while AI might drift more uniformly or maintain artificial consistency.

**Analysis Pipeline**

For each encoding:
1.	Convert the text aspect into a numerical series
2.	Apply Hurst RS analysis (for series ≥500 points) or DFA (for shorter series or as fallback)
3.	Optionally compute surrogate baselines using phase randomization or block shuffling to measure how much structure exceeds random chance
4.	Aggregate the five Hurst exponents (with optional weighting) to produce an overall persistence score
Context-Fit Assessment (Experimental)
An additional layer uses masked language modeling (RoBERTa) to evaluate whether specific words appear in expected contexts. For each occurrence of target terms, the tool masks the word and checks if the model would predict it from context. High scores indicate formulaic, predictable usage; low scores suggest creative or unexpected usage; balanced scores often indicate natural human writing.

**Interpretation Framework**

The following will change based on further benchmarking
•	Aggregate score ≥ 0.56: Strong global memory, human-like persistence
•	Aggregate score ≤ 0.50: Weak memory, anti-persistent or mixed patterns
•	0.50 < score < 0.56: Ambiguous, requiring additional analysis
The tool reports individual encoding statistics including series length (n), method used (RS/DFA), and optionally the delta from surrogate baselines (ΔH) to show how much structure exceeds random chance.

**Technical Implementation**
The tool employs several safeguards for robust analysis:
•	Automatic fallback from RS to DFA for short series
•	Guards against constant or near-constant series
•	Configurable minimum series lengths
•	Batch-safe processing for semantic embeddings with memory caps
•	Proper handling of edge cases in sentence splitting (quotes, ellipses, unconventional punctuation)
**
Limitations and Considerations**

This approach requires sufficient text length (minimum ~2,500 words) to generate reliable statistics. The patterns measured are statistical tendencies, not deterministic markers– excellent human writing might show weak persistence in experimental styles, while sophisticated AI might mimic human patterns. The tool is best used as one perspective among many, complementing rather than replacing close reading and contextual analysis.
The mathematical patterns revealed should be understood as structural signatures rather than quality judgments. They reflect how information is organised across distance in text, offering insights into the underlying production process– whether emerging from human cognitive constraints or machine optimisation patterns.
