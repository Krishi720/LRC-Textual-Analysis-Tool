# LRC Text Differentiator — Proof of Concept (enhanced)
# -----------------------------------------------------
# Differentiates human vs. AI-generated text using Long-Range Correlations (LRC).
# Encodings:
#   1) Multi-target word presence family: OR / inter-arrival gaps / windowed density
#   2) Sentence-length series (words per sentence)
#   3) Function-word indicator series
#   4) Punctuation-cadence (to determine long-range structural rhythm)
#   5) Semantic drift (embedding-space sentence-to-sentence distance)
#
# Enhancements vs original:
# - Multi-keyword targeting: --targets "the,of,and" and --presence-mode {or,gap,density}
# - Constant/short-series guards for stable DFA/Hurst
# - Optional DFA-only mode for comparability
# - Batch-safe semantic drift with sentence cap (--max-sents)
# - Improved sentence splitter for quotes/brackets/ellipses
# - Optional surrogate tests (--surrogates) with ΔH reporting
# - Weighted aggregation and series length reporting
#
# Usage examples:
#   python the_code.py
#   python the_code.py --text1 file.txt --targets "the,of,and" --presence-mode density --window 128
#   python the_code.py --dfa-only --surrogates --weights "1,1,1,0.8,1.2"
#
# Dependencies:
#   python -m pip install -U numpy scipy scikit-learn nolds sentence-transformers "torch>=2.3,<3"

import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
# Context-fit (MLM) imports
from transformers import pipeline
import numpy as np
import nolds

# ---------------------- Config ----------------------
MIN_RS_POINTS = 500  # min series length to use RS Hurst; else DFA fallback
TARGET_WORD_DEFAULT = "was"

FUNCTION_WORDS = {
    "a","an","the","and","or","but","if","then","else","for","nor","so","yet",
    "of","in","on","at","by","to","from","with","without","about","as","into","than","because",
    "is","am","are","was","were","be","been","being",
    "do","does","did","doing","done",
    "have","has","had","having",
    "this","that","these","those","it","its","itself",
    "he","him","his","she","her","hers","they","them","their","theirs","we","us","our","ours",
    "you","your","yours","i","me","my","mine",
    "not","no","yes","also","very","too","just",
    "there","here","when","where","why","how","though","while","over","under","through","between","against"
}
# Punctuation set for cadence/burstiness (includes . , ; : ! ? quotes, parentheses, ellipsis …, en dash – , em dash — , hyphen -)
PUNCT_CHARS = set(list(".,;:!?()—\"'—–-…—"))
# ------------------- Preprocessing ------------------

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())

def split_sentences(text: str) -> List[str]:
    # Split on whitespace that follows a sentence end (. ! ? …) or on newlines
    text = text.strip()
    parts = re.split(r'(?<=[.!?…])\s+|\n+', text)
    return [s.strip() for s in parts if s and s.strip()]

# ------------------- Embeddings ---------------------

try:
    from sentence_transformers import SentenceTransformer
    _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    _sem_model = None
    _sem_err = e

def series_semantic_drift(sentences: List[str], max_sents: int = 5000, batch_size: int = 64) -> np.ndarray:
    if len(sentences) < 2:
        return np.array([0.0])
    if _sem_model is None:
        raise RuntimeError(f"SentenceTransformer unavailable: {repr(_sem_err)}")
    sents = sentences[:max_sents]
    # Normalized embeddings -> cosine = dot
    embeddings = _sem_model.encode(
        sents, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False
    )
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    drift = 1.0 - sims
    return drift.astype(float)

# ---------------------- Encoders --------------------

# Masked-LM for context-fit (RoBERTa; robust tokenization)
try:
    _mask = pipeline("fill-mask", model="roberta-base", device=-1)
    MASK_TOKEN = _mask.tokenizer.mask_token  # usually "<mask>"
except Exception as e:
    _mask = None
    _mask_err = e

def series_word_presence_multi(words: List[str], targets: List[str]) -> np.ndarray:
    T = set(t.lower() for t in targets)
    return np.array([1.0 if w in T else 0.0 for w in words], dtype=float)

def series_word_interarrival_multi(words: List[str], targets: List[str]) -> np.ndarray:
    T = set(t.lower() for t in targets)
    idx = [i for i, w in enumerate(words) if w in T]
    if len(idx) < 2:
        return np.array([0.0])
    gaps = np.diff(idx).astype(float)
    return np.log1p(gaps)  # stabilize heavy tails

def series_word_density_multi(words: List[str], targets: List[str], window:int=128) -> np.ndarray:
    x = series_word_presence_multi(words, targets)
    if len(x) == 0:
        return x
    window = max(4, min(window, len(x)))
    k = np.ones(window, dtype=float) / window
    dens = np.convolve(x, k, mode="same")
    return dens.astype(float)

def series_sentence_lengths(sentences: List[str]) -> np.ndarray:
    lengths = [len(tokenize_words(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    return np.array(lengths, dtype=float)

def series_function_word_indicator(words: List[str]) -> np.ndarray:
    return np.array([1.0 if w in FUNCTION_WORDS else 0.0 for w in words], dtype=float)

def series_punct_counts_per_sentence(text: str, punct_chars: Optional[set] = None) -> np.ndarray:
    """Count defined punctuation marks per sentence; returns a real-valued series."""
    if punct_chars is None:
        punct_chars = PUNCT_CHARS
    sents = split_sentences(text)
    counts = []
    for s in sents:
        c = sum(1 for ch in s if ch in punct_chars)
        counts.append(c)
    return np.array(counts, dtype=float)

def fano_factor(x: np.ndarray) -> float:
    """Variance-to-mean ratio with guards (0 if mean≈0)."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    v = float(np.var(x))
    if m <= 1e-12:
        return 0.0
    return v / m

# ------------------ Surrogates ----------------------

def block_shuffle(x: np.ndarray, block: int = 50, seed: int = 0) -> np.ndarray:
    x = np.asarray(x).ravel()
    n = len(x)
    if n <= block:
        return x.copy()
    blocks = [x[i:i+block] for i in range(0, n, block)]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    return np.concatenate(blocks)

def phase_randomize(y: np.ndarray, seed: int = 0) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    Y = np.fft.rfft(y)
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2*np.pi, size=Y.shape))
    phases[0] = 1.0
    if Y.shape[0] % 2 == 0:
        phases[-1] = 1.0  # keep Nyquist real
    y_surr = np.fft.irfft(Y * phases, n=len(y)).real
    return y_surr

def context_fit_scores(
    text: str,
    terms: List[str],
    topk: int = 10,
    max_evals: int = 200
) -> List[float]:
    """
    For each occurrence of any term in `terms`, mask that occurrence in its sentence
    and ask the MLM how well the original word fits the context.
    Returns scores in [0,1], where 1 = top-1 prediction, 0 = not in top-k.
    """
    if _mask is None:
        raise RuntimeError(f"Context-fit unavailable: {repr(_mask_err)}")

    sentences = split_sentences(text)
    scores: List[float] = []
    T = {t.lower() for t in terms}

    for s in sentences:
        # find exact word hits (case-insensitive), one at a time
        words = re.findall(r"\b[\w']+\b", s)
        lower = [w.lower() for w in words]
        for i, w in enumerate(lower):
            if w in T:
                # mask just this occurrence
                masked = re.sub(rf"\b{re.escape(words[i])}\b", MASK_TOKEN, s, count=1)
                try:
                    preds = _mask(masked, top_k=topk)
                except Exception:
                    continue
                # rank of the original word among top-k predictions
                rank = topk + 1
                for idx, p in enumerate(preds, start=1):
                    cand = p["token_str"].strip().lower()
                    if cand == w:
                        rank = idx
                        break
                score = 1.0 - (rank - 1) / topk if rank <= topk else 0.0
                scores.append(score)
                if len(scores) >= max_evals:
                    return scores
    return scores

# ------------------ Hurst / DFA Safe ----------------

def hurst_or_dfa(x: np.ndarray, min_rs_points: int = MIN_RS_POINTS, dfa_only: bool = False) -> Tuple[float, str]:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n == 0:
        return 0.5, "NA(empty)"
    if np.allclose(x, x[0]):
        return 0.5, "NA(constant)"
    std = np.std(x)
    x = (x - np.mean(x)) / (std + 1e-12)
    if dfa_only:
        try:
            return float(nolds.dfa(x)), "DFA"
        except Exception:
            return 0.5, "DFA(error)"
    if n >= min_rs_points:
        try:
            return float(nolds.hurst_rs(x)), "RS"
        except Exception:
            try:
                return float(nolds.dfa(x)), "DFA(fallback)"
            except Exception:
                return 0.5, "DFA(error)"
    else:
        try:
            return float(nolds.dfa(x)), "DFA"
        except Exception:
            return 0.5, "DFA(error)"

# --------------------- Aggregation ------------------

@dataclass
class EncodingStat:
    value: float
    method: str
    n: int
    surrogate: Optional[float] = None
    delta: Optional[float] = None

@dataclass
class LRCResult:
    encoding_values: Dict[str, EncodingStat]  # encoding -> stats
    aggregate_score: float
    verdict: str
	# extras (not in weighted aggregate)
    punc_fano: Optional[float] = None
    punc_n: Optional[int] = None

def aggregate_verdict(exponents: List[float], weights: Optional[List[float]] = None) -> Tuple[float, str]:
    exps = np.array(exponents, dtype=float)
    if weights is None:
        agg = float(np.mean(exps))
    else:
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        agg = float(np.sum(exps * w))
    if agg >= 0.56:
        verdict = "More human-like global memory (persistent)."
    elif agg <= 0.50:
        verdict = "More AI-like or weak global memory (anti-persistent)."
    else:
        verdict = "Ambiguous / mixed global memory."
    return agg, verdict

# --------------------- Main API ---------------------

def compute_encoding(x: np.ndarray, min_rs_points: int, dfa_only: bool, use_surrogates: bool,
                     series_kind: str) -> EncodingStat:
    """series_kind: 'indicator' or 'real' (affects surrogate choice)"""
    n = int(len(x))
    val, method = hurst_or_dfa(x, min_rs_points=min_rs_points, dfa_only=dfa_only)
    sur = delt = None
    if use_surrogates and n >= 16:
        if series_kind == 'real':
            xs = phase_randomize(x)
        else:
            block = max(10, min(50, n // 10))
            xs = block_shuffle(x, block=block)
        s_val, _ = hurst_or_dfa(xs, min_rs_points=min_rs_points, dfa_only=dfa_only)
        sur = float(s_val)
        delt = float(val - sur)
    return EncodingStat(value=float(val), method=method, n=n, surrogate=sur, delta=delt)

def analyze_text(
    text: str,
    target_word: str = TARGET_WORD_DEFAULT,
    min_rs_points: int = MIN_RS_POINTS,
    targets: Optional[List[str]] = None,
    presence_mode: str = "or",
    window: int = 128,
    dfa_only: bool = False,
    use_surrogates: bool = False,
    max_sents: int = 5000
) -> LRCResult:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not targets:
        targets = [target_word.lower()]

    encodings: Dict[str, EncodingStat] = {}

    # 1) Multi-target presence family
    if presence_mode == "or":
        x = series_word_presence_multi(words, targets)
        presence_name = "word_presence_any"
        kind = 'indicator'
    elif presence_mode == "gap":
        x = series_word_interarrival_multi(words, targets)
        presence_name = "word_interarrival_any"
        kind = 'real'
    else:  # density
        x = series_word_density_multi(words, targets, window=window)
        presence_name = f"word_density_any(w={window})"
        kind = 'real'
    encodings[presence_name] = compute_encoding(
        x, min_rs_points, dfa_only, use_surrogates, kind
    )

    # 2) Sentence lengths
    sl = series_sentence_lengths(sentences)
    encodings["sentence_lengths"] = compute_encoding(
        sl, min_rs_points, dfa_only, use_surrogates, 'real'
    )

    # 3) Function-word indicator
    fw = series_function_word_indicator(words)
    encodings["function_words"] = compute_encoding(
        fw, min_rs_points, dfa_only, use_surrogates, 'indicator'
    )

    # 4) Punctuation cadence (per-sentence counts -> DFA/RS) + Fano (burstiness)
    pc = series_punct_counts_per_sentence(text)
    encodings["punctuation_cadence"] = compute_encoding(
        pc, min_rs_points, dfa_only, use_surrogates, 'real'
    )
    punc_fano_val = fano_factor(pc)

    # 5) Semantic drift between adjacent sentences
    sd = series_semantic_drift(sentences, max_sents=max_sents)
    encodings["semantic_drift"] = compute_encoding(
        sd, min_rs_points, dfa_only, use_surrogates, 'real'
    )

    # Aggregation (respect presence encoding first)
    order = [presence_name, "sentence_lengths", "function_words", "punctuation_cadence", "semantic_drift"]
    exps = [encodings[name].value for name in order]
    agg, verdict = aggregate_verdict(exps)

    return LRCResult(
        encoding_values=encodings,
        aggregate_score=agg,
        verdict=verdict,
        punc_fano=punc_fano_val,
        punc_n=int(len(pc))
    )

def compare_texts(
    text1: str, text2: str,
    target_word: str = TARGET_WORD_DEFAULT,
    min_rs_points: int = MIN_RS_POINTS,
    targets: Optional[List[str]] = None,
    presence_mode: str = "or",
    window: int = 128,
    dfa_only: bool = False,
    use_surrogates: bool = False,
    weights: Optional[List[float]] = None,
    max_sents: int = 5000,
    # context-fit args
    context_terms: Optional[str] = None,
    context_topk: int = 10,
    context_maxevals: int = 200,
) -> None:
    # analyze both texts
    r1 = analyze_text(text1, target_word, min_rs_points, targets, presence_mode, window,
                      dfa_only, use_surrogates, max_sents)
    r2 = analyze_text(text2, target_word, min_rs_points, targets, presence_mode, window,
                      dfa_only, use_surrogates, max_sents)

    # ------- pretty-printer (NO recursion here) -------
    def print_block(res: LRCResult, label: str, weights_opt: Optional[List[float]]):
        print(f"\n=== {label} ===")
        for name, stats in res.encoding_values.items():
            line = f"{name:24s}: {stats.value:.3f}  [{stats.method}]  (n={stats.n})"
            if getattr(stats, "delta", None) is not None and getattr(stats, "surrogate", None) is not None:
                line += f"  ΔH={stats.delta:+.3f} (sur={stats.surrogate:.3f})"
            print(line)

 # Punctuation burstiness (Fano) — extra, not in weighted aggregate
        if getattr(res, "punc_fano", None) is not None:
            n_punc = res.encoding_values.get("punctuation_cadence", EncodingStat(0,"",0)).n
            print(f"{'punctuation_fano':24s}: {res.punc_fano:.3f}  (n={n_punc})")

        # aggregate (weighted if provided)
        presence_keys = [k for k in res.encoding_values.keys() if k.startswith("word_")]
        agg_order = presence_keys + ["sentence_lengths", "function_words", "punctuation_cadence", "semantic_drift"]
        exps = [res.encoding_values[k].value for k in agg_order if k in res.encoding_values]
        if weights_opt is not None and len(weights_opt) == len(exps):
            w = np.array(weights_opt, dtype=float)
            w = w / (np.sum(w) + 1e-12)
            agg = float(np.sum(np.array(exps) * w))
        else:
            agg = float(np.mean(exps))
        print(f"Aggregate (mean/weighted): {agg:.3f}")
        print(f"Verdict: {res.verdict}")

    # ------- context-fit helper -------
    def _print_context_fit(label: str, txt: str):
        if not context_terms:
            return
        terms = [t.strip() for t in context_terms.split(",") if t.strip()]
        try:
            cfs = context_fit_scores(txt, terms, topk=context_topk, max_evals=context_maxevals)
            if cfs:
                print(f"Context Fit ({label}, mean over {len(cfs)} hits): {np.mean(cfs):.3f}  [1=strong, 0=weak]")
            else:
                print(f"Context Fit ({label}): no occurrences of provided terms found.")
        except Exception as e:
            print(f"Context Fit ({label}): unavailable ({e})")

    # ------- print results -------
    print_block(r1, "Text 1", weights)
    _print_context_fit("Text 1", text1)

    print_block(r2, "Text 2", weights)
    _print_context_fit("Text 2", text2)

    # comparison summary stays last
    print("\n--- Comparison Summary ---")
    if r1.aggregate_score > r2.aggregate_score + 0.02:
        print("Text 1 shows stronger long-range persistence (more human-like).")
    elif r2.aggregate_score > r1.aggregate_score + 0.02:
        print("Text 2 shows stronger long-range persistence (more human-like).")
    else:
        print("Both texts are similar in global memory; use local features or context-fit for tie-break.")


# --------------------- CLI / Tests ------------------

HUMAN_TEXT = """
War chokes the flow of humanitarian aid through the port city of Hodeida and famine ripples through Yemen’s interior. Panama deepens its canal and distant ports scramble to dredge their harbours to accommodate the hulking ships suddenly able to transit the expanded waterway. Russian forces advance through the Caucasus Mountains via the Roki tunnel, taking the tunnel and, with it, the power to limit the eastward expansion of the European Union. Pirate ‘fishermen’ gather at the Bab-el-Mandeb strait, intending to capture ships bound for the Red Sea and Suez Canal. Central American migrants gather to make the perilous journey north, only to be funnelled into makeshift camps at the US border. Held there by half-baked policies and politics and occasional force, the would-be refugees wait indefinitely for asylum claims to be processed.
These are all chokepoints: sites that constrict or ‘choke’ the flows of resources, information, and bodies upon which contemporary life depends. Chokepoints have long been recognised for their considerable economic, political, and military significance. Narrow, often diﬃcult to avoid, and heavily traﬃcked, chokepoints can present risks and potentially paralysing vulnerabilities to vital systems and networks. There is always the potential that they will be shut down or become too congested, crippling mobility, exchange, and communication. Some chokepoints are apparent like the overburdened internet server or traﬃc-prone intersection. Others shape our lives in covert ways. In either case, analysts tend to approach them at a remove– as nodes in networks or logistical problems to be solved. As a result, we know surprisingly little about how chokepoints work in practice and even less about everyday life in and around them.
We develop an anthropology of chokepoints. For us, the chokepoint is a useful analytic for examining the operative– and often generative– interplay of circulation and constriction in the contemporary world. The term ‘chokepoint’, which generally refers to a point of constriction or blockage within a system, network, or process, is used across a variety of domains, including: shipping, logistics, engineering, security, military operations, and labour organising. Across communities of practice, significant points of constriction may be recognised at diﬀerent registers. The most obvious chokepoints are geographical sites associated with narrow passages, but others may be unrecognisable in Euclidean space. Think, for example, of a patent that restricts the production of cheap medicines, a regulation that slows internet traﬃc, or a shortage of key workers needed to keep traﬃc moving. Recognising that the concept refers to a range of phenomena, the anthropological approach that we develop focuses on sites conducive to ethnographic research.
For anthropologists and other scholars, we argue, the chokepoint can serve as an
analytic that renders unexpected and significant spatiotemporal connections and social relationships visible, recasting a range of contemporary problems, including: mobility, migration, logistics, trade, geopolitics, security, and statecraft. But ‘chokepoint’ is more than a description of a site or node conducive to scholarly analysis. It is also a concept that a range of actors– politicians, planners, managers, and military strategists– deploy in their own analyses, claims, and interventions. Given that ‘chokepoint’ is used by our interlocutors as both a concept and analytic, sometimes in our fieldwork settings, what does it mean to redeploy it for anthropological ends? We argue that the chokepoint can be usefully understood as a lateral concept. To think about anthropological concepts in ‘lateral’ terms underlines the fact that ethnographers and their interlocutors pursue parallel and, increasingly, intersecting lines of thought and analysis. Methodologically, lateral analysis entails ‘tacking back and forth’ between the chokepoint as a phenomenon in the world and a concept used by others. With this as a point of departure, we ask: How do chokepoints work? For whom? With what eﬀects? How are they implicated in power relations and struggles? What forms of labour– formal and informal, licit and illicit– do they depend on and give rise to? What spatiotemporal relations do they produce? What methods and modes of analysis do they enable?
Chokepoints are zones of operative paradox in several key ways. First, the anthropological (and public) imagination about global connection tends to assume that increased spatial connectivity– more and bigger conduits– leads to accelerated circulation. However, our ethnographic research in and around the material channels of ‘global flow’ – roads, tunnels, pipelines, and waterways– suggests that connectivity and speed should be conceptually delinked. At chokepoints, we find, increased connectivity may slow things down. As a matter of everyday operation, the resulting traﬃc is emblematic of chokepoints’ function and dysfunction. Second, chokepoints can disrupt conventional paradigms of power and agency, opening up space for potential transformation, even if subtle or limited. They are places where the weak can become strong; where the everyday tactics of getting through and getting by can undermine the grand strategies of capital and security; and where digital technologies can outperform analog technologies. An indigenous community blockades a key highway to protest extractivist encroachment, turning the state’s infrastructure against it. Dhows (wooden sailing vessels) navigate in the shadows of supertankers, ferrying everyday essentials between the Horn of Africa and the Arabian Peninsula. Pilots are trained on miniature physical models to navigate the complex winds and currents that aﬀect megaships plying the Panama Canal. Third, and finally, chokepoints may be locatable on a map, but they are not exclusively ‘local’ phenomena. What happens at chokepoints can ripple well beyond the chokepoint itself, making them geographically distributed in their eﬀects. A well-timed strike at a logistics hub or port can cripple a supply chain. A terror threat at a major airport can trigger cascading delays worldwide.
The paradoxes associated with chokepoints are productive entry points for anthropological inquiry. Ethnography, in particular, can reveal the micro-workings of these pivotal spaces, showing how operative paradoxes structure lifeworlds and the networks and flows with which they articulate. An anthropology of chokepoints might allow the analyst to attend to how such spaces are mobilised from within, nearby, and at a distance– and to what ends. For militaries and businesses, chokepoints may be spaces to control and benefit from through the appropriation and of these forces. Chokepoints, in this regard, become staging grounds for capitalist, biopolitical and geontological agendas. But for traﬃckers, repairman, service crews, pirates, smugglers, soldiers and others, chokepoints can be ‘worked’ to a variety of ends so as to siphon oﬀ value, challenge the status quo, or subvert the logics of globalisation. What emerges in and around the chokepoint, then, are high-stakes interplays and tensions between circulation and regulation, local and remote forces, and human and non-human agencies, often with unexpected and far-ranging eﬀects. These dynamics demand ethnographic attention.
By foregrounding constriction and tracing its eﬀects across scales, chokepoint anthropology recasts longstanding concerns with the character and consequences of global circulation and ‘flow’. These metaphors, significantly, dovetail with meta- narratives that link ever-increasing connectivity with speed. Scholars have challenged global-centric obsessions with flow on theoretical grounds and political grounds. In practice, anthropologists observed, pathways, networks and linkages are lumpy, discontinuous, and contested. In response, scholars generated new concepts and metaphors for understanding global connection like assemblages, technological zones, logistical corridors.
While many scholars have argued ‘against flow’, we use chokepoints to think against speed. There is a sense that life has been speeding up everywhere in late capitalism even as the resurgence of borders walls, embargos, and detention camps highlights the persistence of blockage, enforced emplacement, captivity, and stuckness. Against this backdrop, the anthropology of chokepoints prompts a diﬀerent reading of circulation. At chokepoints, the movement of people, commodities, and information is neither completely fluid nor obstructed, but somewhere in-between: slowed, filtered, controlled, and ‘choked’ in ways that ripple across space and time. This is a world animated not only by circulation (and corresponding dynamics of flow, blockage, and speed) but also, we argue, by constriction and traﬃc. To think through chokepoints, analytically and ethnographically, foregrounds a diﬀerent set of actors, ecologies, and mechanisms. Chokepoints expose the underside of global circulation– the situated processes through which deterritorialized flows are channelled, diverted, and bogged down in the murky, sticky particularities of localities.
Consider Lock No. 52: a non-obtrusive structure located on the Ohio River in Southern Illinois, just 23 miles upriver from its confluence with the Mississippi River. Built in 1929, the aging facility– concrete cracked, paint peeling– looks like an artefact of another era and a diﬀerent economy, a now-mythical time when centres of American industry pumped cargo through the nation’s inland waterways via barges. Nevertheless, this structure remains one of the busiest conduits for North American commerce– and one of the biggest chokepoints on the twenty-first century United States economy. For years, traﬃc has backed up waiting to pass through the lock, which has regularly been closed for hours or days at a time due to issues associated with deferred maintenance. The lockmaster put it this way, ‘The lock is kept going with all of the bubble gum and duct tape we’ve got left’. As a result of Lock 52 and Lock 53, just downstream, it can take 5 days to travel a 100-mile stretch of river. The chokepoint manages to be simultaneously out-of-the-way– distant from major cities, it is roughly midway between Nashville and St. Louis– and central to national commerce. Tens of billions of dollars of cargo– grain, corn, coal, steel, cement, and more– pass through it annually, so when it is closed or traffic-clogged, there are reverberations far beyond Illinois and the barge industry. Cargo is diverted onto trucks, which then overburden highways, generating chokepoints elsewhere. The prices of exports and consumer goods rise, constricting manufacturers’ international competitiveness and the everyday purchasing power of labourers. What happens at Lock 52 has implications for millions of people. Such chokepoints are not only good to control or pass through, they are also– to use Levi Strauss’s timeworn phrase– good to think. But, how do we think about something like this? Something that is both a concept and a thing in the world?
Undersea cables, ports, migration bottlenecks, pipelines, highways, and financial algorithms– these can all become chokepoints. So what makes a chokepoint a chokepoint? What we have in mind with the anthropology of chokepoints is an approach that turns on a signature condition of the contemporary: constriction. We argue that it is productive to approach the chokepoint as, on the one hand, a site and, on the other, a concept that our interlocutors use to analyse and act in the world. Thus, as we note above, it can be methodologically and analytically productive to tack back and forth between the chokepoint as a site and a circulating concept. Though many chokepoints are recognisable as Euclidean spatial forms (like a narrow geographic passage), others take on different forms specific to the systems, networks, and processes of concern. Indeed, networks, as Nicole Starosielski argues, can have multiple chokepoints– and multiple kinds of chokepoints– which articulate with one another in predictable and unpredictable ways. The undersea cables of the internet converge at particular nodes (topological chokepoints), while, above the surface, a confounding array of regulatory chokepoints constrict and route the flow of information (and profits) to some and not others. If chokepoints are sites in network space, the ‘chokepoint’ concept can also be a tool itself: a means of making claims (upon resources, territory, and so on) and a rationale for intervention (like securitisation or regulation). With the anthropology of chokepoints, we propose an approach that, by focusing attention on the interplay of circulation and constriction, as well as speed and slowness, illuminates spatiotemporal connections and social forms that may otherwise be difficult to see.
Approached in this way, chokepoints can provide ethnographic purchase amid the kinetic fray of circulation and global connection. As a field site, the chokepoint is at once situated and distributed, confounding conventional understandings of locality. This calls for careful attention to the social, material, and technological particularities of the sites that serve as lynchpins of broader systems, including the forms of passage (and life) that they facilitate and impede. But local conditions are only part of what makes a chokepoint a chokepoint. Activities at these points shape geopolitics, economies, and everyday life far from their locations on the map. To properly frame chokepoints as objects of anthropological study, we need to trace something else: movement. Or, more accurately, movement-through. Only then can we begin to understand the dynamics of constriction that define chokepoints and, in so doing, shape the contemporary.
We are not, of course, the first to train our eyes on chokepoints. Historically, they have emerged as loci of attention, anxiety, and strategic planning for military tacticians and transport logisticians. Scholars of transportation geography, not surprising, have approached chokepoints as places in need of modelling and analysis–logistical challenges to be managed in the service of broader transportation and shipping flows. Meanwhile, critical logistics scholarship has turned managerial analysis on its head by tracing the logics of power and capital embedded in and advanced through the so-called logistics revolution. In this work, chokepoints figure as diagnostic nodes: sites that lay bare the contradictions, assumptions, and vulnerabilities of global trade and flow. Similarly, sociologists of labour and world-systems theorists have rediscovered chokepoints as sites of opportunity for reinvigorating global labour movements– locales where precarious workforces might disrupt the juggernaut of global trade in a manner that adapts the organising successes of the twentieth century to fit a changed set of conditions. All of these interventions highlight critical dynamics of chokepoints.
The approach charted here builds upon and extends this work in two significant ways. First, chokepoint scholarship tends to focus on geographically conventional nodes (ports, straits, production systems, etc.), limiting the analytical potential of what chokepoints are and can be. We, by contrast, conceptualise chokepoints broadly, as sites that funnel and constrict not just commodity flows, but broader possibilities and dynamics. This allows us, for example, to see global climate hotspots as chokepoints not only of space but of time. Second, the chokepoint analytic focuses attention on the dynamic interplay of circulation and constriction. Useful here is Tsing’s (2000) call to resituate global flows in ‘landscapes’ so as to understand how circulation is contingent upon the work of establishing and contesting ‘channels’. The chokepoint analytic that we propose extends Tsing’s metaphors of ‘landscapes’ and ‘channels’ and, later, ‘friction’ (2005). But, whereas friction emphasises how global interconnection works across and through diﬀerence, chokepoints foreground constriction as the means and ends of a range of practices and projects. Put another way, constriction is not simply a drag on unbridled flow and speed. It is a generative social process through which people engaged in diﬀerent activities collectively alter the trajectory, pace, and character of circulation. This raises interesting questions: How– and by whom– does constriction happen? What forms of social, political, and economic life does it engender? How do actors negotiate or manipulate passage for their own purposes? What happens to things, bodies, and ideas as they move toward, through, and out of the chokepoint?

We offer the following eight dimensions as openings for an anthropology of chokepoints. They provide analytic bearings and ethnographic entry points for navigating choked spaces and times.
(1) Chokepoints are constructed. While physical geographies like narrow terrestrial and aquatic passages are often associated with the concept, chokepoints are brought into being and maintained by people and institutions who appropriate, manage, and alter sociomaterial features for political control, profit, surveillance, protest, and so on. While the lifeworlds of chokepoints often appear peripheral to global circulation, they are mutually constituted through visible and invisible exchanges. Thus, we might ask: How is this chokepoint made, maintained, and manipulated? By whom? Why? To what effect?
(2) Chokepoints are a means of exerting control or amplifying power. States, firms, organised labour, and social movements may seek to establish and control chokepoints because of their political and economic instrumentality; they are difficult to bypass and often vital to the operation of larger systems. Capturing and channelling circulation is a means of enacting political authority at multiple scales, from the national, where chokepoints can be deployed for statemaking (Dua in this volume), to the regional and global, where vascular geopolitics can create or alter international relations (Dunn in this volume). Thus, the chokepoint becomes a node where power can be established and exerted across larger territories. By the same token, chokepoints are vulnerable nodes where acts of sabotage and resistance have amplified eﬀects. What happens at chokepoints, then, can turn expected power dynamics upside down. But approached ethnographically, chokepoint power dynamics do not demonstrate a neat domination-resistance binary. As the articles in this volume show, diverse forms of power are generated around these sites.
(3) Chokepoints are relational or ecological. While geopolitical and managerial analyses tend to highlight how particular sites or conduits are appropriated and deployed by pre-defined social and political groups to coerce, control, and exploit, the reality on the ground is negotiated and messy. Chokepoints are diﬀerent things for different people at different times. For the city planner, congested roads are a problem. But for the smuggler, they are camouflage. And, for the roadside vendor, they create a captive market. Like infrastructures, chokepoints appear only as a relational phenomenon, not a thing that is stripped of use. What chokepoints are and do depends on one’s position. Moreover, the profusion of competing claims and practices that come together around chokepoints– both actual and potential– can produce constriction and congestion within them.
(4) Chokepoints depend on and give rise to particular forms of labour. Born of the relation between movement and site, their establishment, securitisation, and everyday operation require skill and expertise that can be site-specific. Not coincidentally, we see a familiar cast of characters– the regulator, the engineer, the navigator, the smuggler, the soldier, the pirate, etc. – ‘working’ chokepoints to diﬀerent ends. For example, navigating large oceangoing ships through confined waterways depends on skilled pilots with embodied knowledge of the chokepoint environment (Carse in this volume). Acting in ways that span informal, illegal, and illegible domains, people also develop inventive and pragmatic means of working with and around immediate constrictions in ways that can have repercussions beyond the chokepoint itself. We inserted backslashes in informal, illegal, and illegible to flag the fact that ambiguities and shifting roles in and around chokepoints can have a range of possible eﬀects. Customs agents, for example, organise rackets to smuggle goods and bodies through a ‘sensitive’ chokepoint. Who, after all, is better positioned to facilitate (and profit from) illicit passage than the gatekeeper?
(5) Chokepoints can be geographically distributed. As sites in various forms of space (Euclidian, networked, etc.), chokepoints transcend locality. They can come into existence and morph through the convergence of long-range forces (Dunn in this volume). Their existence, control, and constriction have both upstream and downstream dynamics, which can be imagined as deltas of influence. This distributed spatiality is, of course, central to their definition as chokepoints because the term implies a relationship between a node and a longer network or larger circulatory system.
(6) Chokepoints are locationally sticky, but can also be mutable and mobile. It is easy to imagine how chokepoints can be appropriated and reappropriated by different groups. Less obvious are the ways that chokepoints can be moved or transformed. To be clear, this is not easy. Because chokepoints can produce stubborn entanglements of social and material forms, shifting them may require immense labour and investment. But it can be done (as Dunn emphasises in this volume). Indeed, as commercial and geopolitical imperatives shift, old chokepoints may be rendered obsolete and new ones constructed. As Jatin Dua (this volume) shows, chokepoint sovereignty is a tenuous thing. Chokepoints, then, can be understood as sticky phenomena that become mutable and mobile with enough heft and luck.
(7) Chokepoints emerge in time and constrain temporalities. Chokepoints are sites where multiple temporalities intersect. Measured in the time of hours and days, they draw attention to the variable speeds at which things circulate in relation to material features. The heavy traﬃc and slow movements that define chokepoint time raise questions about the sense of steady acceleration often linked with the temporal experience of modernity. While we often imagine chokepoints as spatial phenomena that exist in time, the concept of a narrow passage that constrains movement from one point to another can be extended to consider the relationships among temporalities. Just as a pipeline limits the flow of oil, there are spaces where multiple dissonant futures are invoked and actively anticipated in the present: industrial development promising boundless economic growth coexists with impending environmental apocalypse. As a temporal metaphor, the chokepoint reveals how future possibilities may be constrained– or choked– by the coexistence of too many proposals in the present. We may, accordingly, posit that there are both chokepoints in time and of time.
(8) Chokepoints enable telescopic analysis. Chokepoints look diﬀerent depending on the scale of analysis. From the bird’s-eye-view of the geopolitical or logistical analyst, the chokepoint is conceptualised as a node among lines and boundaries. Nations make territorial claims, pursuing political advantage. Firms seek to capitalise on supply chains. However, the everyday dynamics of chokepoints trouble these strategies and the meta-narratives of global circulation, connectivity, security upon which they are predicated. For example, the global transition to a new class of megaships hinges not only on reengineering infrastructure and environments, but also, and crucially, training pilots to develop a ‘feel’ for how larger vessels handle in confined waterways (Carse in this volume). Seen through the lens of everyday pragmatics (Middleton in this volume), geopolitical strategies and techno-formal processes of logistics and security also give rise to and are confounded by local tactics of getting through and getting by. Thus, chokepoint ethnographies can be a useful complement to macro-scale analyses and provide new entry points for theorising global circulation– from the margins ‘within’, as it were. Here, the big doesn’t only explain the small; the small also explains the big. As an analytic, then, the chokepoint serves as a telescope for exploring relationships and disjunctures across multiple spatial and temporal scales.

"""

GPT_TEXT   = """

Chokepoints are not merely narrow places on maps; they are social formations in which the dynamics of connection and constraint are rendered palpable in everyday life. To take chokepoints seriously is to move beyond the celebratory idiom of seamless circulation that has so often animated accounts of globalisation, and to attend instead to the frictions through which global projects become real. From ports to pipelines, from tunnels to app stores, from border crossings to fibre-optic landing stations, the contemporary world is braided together by infrastructures and logistics that promise speed and reliability even as they repeatedly produce and depend upon strategic constrictions. Chokepoints index the operative paradox of our age: the more we connect, the more we have to queue; the wider the network, the more decisive its narrowest gates; the louder the rhetoric of flow, the thicker the knots through which flow is rendered legible, taxable, governable, and stoppable. An anthropology of chokepoints attends to this paradox ethnographically, asking what constriction does, how it is made, when it appears, and for whom it matters, in order to reframe the conceptual terrain through which circulation, globalisation, infrastructure, and logistics are commonly understood. If the twentieth century trained the social sciences to think with metaphors of flows, streams, and networks, the constricted contemporary requires that we take seriously the architectures and practices that make stopping, slowing, and sorting indispensable. The anthropology of chokepoints therefore turns our gaze to the junctions where infrastructures of circulation must be narrowed in order to be made safe, profitable, or governable. In so doing, it shows how the fantasy of uninterrupted flow requires constriction as its condition of possibility, and it foregrounds how chokepoints are political, technical, moral, and affective devices at once. To ethnographically approach chokepoints is to enter worlds dense with rule and exception. Consider the longshore worker who calibrates the swing of a container crane while listening for the radio crackle that signals a change in port security posture; the customs broker who knows which tariff classifications invite audit; the pipeline controller who manages pressure at a compressor station during a cold snap; the tunnel guard who studies faces for the micro-gestures that training materials describe as “risk cues”; the content moderator who triages flagged uploads before they traverse the platform’s monetisation chokepoints; the network engineer who detours packets when a subsea cable is cut by a trawler. Each actor inhabits a constricted interface where the throughput of the world depends on decisions made under constraint. Waiting rooms, laydown yards, holding tanks, switchyards, platforms’ back offices, data centres’ access vestibules, and the algorithmic corridors of rate-limits and review queues are the furniture of these worlds. Chokepoints reconfigure power by repositioning the marginal as consequential. The island stevedore whose union controls gang allocation, the gas valve technician who carries the key to a critical station, the clerk whose stamp renders a traveler’s presence legitimate, the smallholder whose field abuts a pipeline right of way and who can slow repairs by refusing access, the volunteer coordinator who pauses a convoy when rumours of banditry thicken, the independent developer who withholds compliance with an app store’s new rules and delays a software update for millions: these figures become pivotal not despite their localness but because chokepoints multiply the effects of local action. The anthropology of chokepoints thus supplements a macro-politics of capital and state with a micro-politics of gates and gateways. What appears as “critical infrastructure” when viewed from national dashboards is often, up close, a set of mundane practices—shifts, checklists, seals, visual inspections, and mouse clicks—through which distributed systems are tightened into narrow passages. Ethnography here avoids the twin temptations of fetishising technical devices and dissolving them into abstract structures; it traces how authority and vulnerability cohabit in constricted spaces. The lure of chokepoints for analysis also arises from their paradoxical temporalities. Queues organise expectations and emotions, while schedules confront their material limits. The trucker who sleeps in a cab by the port gate embodies the conversion of calendar time into bodily fatigue; the harried agent at an airline desk translates systems failure into face-to-face negotiation over scarce seats; the offshore crew watches the swells, knowing the pilot boat cannot approach in heavy weather; the content creator refreshes dashboards while a video remains stuck in a monetisation review backlog; the migrant studies the tide tables and patrol patterns before attempting a crossing; the remittance sender watches a “pending” status while a bank tool flags an anomaly. Such scenes demonstrate that waiting is not merely the absence of movement but a social mode in which obligations and claims are recalibrated. Chokepoints generate what we might call “line time,” where the legitimacy of patience, the temptations of cutting, the moralities of letting others pass, and the skills of enduring boredom become key competencies. In line time, globalisation is felt in the gut. The spatial grammars of chokepoints are equally instructive. The diversion around a blocked canal sends vessels the long way; a burned bridge detours an entire region’s supply chain onto fragile roads; a payment that cannot clear pushes informal couriers into adjacent jurisdictions; a content moderation chokepoint shifts conspiracist discourse to smaller venues where it intensifies; a police checkpoint closes a market day and turns subsistence sellers into debtors; a substation outage redirects energy flows and blackens a constellation of homes. Because chokepoints induce rerouting rather than pure stoppage, they function like valves in hydrology, changing pressure on adjoining circuits. This relationality means that ethnography must be ambulatory, following spillovers as they are distributed across distance. The chokepoint is never simply “there”; it is also in the places that suddenly become busier, quieter, more profitable, or more dangerous as traffic is redistributed. Attention to chokepoints also recalibrates how the social sciences treat expertise. Infrastructural worlds cultivate tactical knowledges of exception: a ship’s pilot, a dispatcher, a gate clerk, a platform reviewer, a tunnel inspector each learns through apprenticeship how rules bend in practice. Compliance itself is a craft: learning to “read” seals and paperwork, deciphering the affordances of scanners and dashboards, knowing when a “hold” can be lifted without inviting audit, and gauging when a backlog has churned to the point where escalation will succeed. Far from being reducible to bureaucracy, chokepoints depend upon and engender skilled judgment. The localisation of consequential decisions in constrained interfaces means that craft and bureaucracy interpenetrate, and that universalisms of standards coexist with particularisms of practice. The materiality of chokepoints involves an ensemble of gates, ledgers, sensors, cameras, badges, stamps, dashboards, turnstiles, thermostats, rate-limiters, and kill switches. Each object embeds a script for action: scanning a barcode presupposes a standard; a container seal presupposes a chain of custody; a two-factor prompt presupposes a registered device; a blast door presupposes a fire code; an API limit presupposes an acceptable use policy; a retail “manager key” presupposes a hierarchy of override. These scripts are enforced by both machines and humans, and they are frequently circumvented by them. The discretionary act—wave through, inspect further, return to sender, escalate, detain, override—repeats thousands of times, and it is through such repetition that social difference enters the circuit: profiling and bias become throughput variables. The anthropology of chokepoints therefore must keep in view how race, gender, class, legal status, and citizenship shape the probabilities of delay and the costs of waiting. Chokepoints concentrate risk and thereby become magnets for securitisation. Ports and pipelines are guarded, tunnels are surveilled, app stores are policed, clearinghouses are audited, border crossings are militarised, and even the “soft chokepoints” of content distribution are fortified by terms and teams devoted to exclusion. Yet securitisation often generates its own vulnerabilities: a checkpoint lengthens queues that become targets for theft or attack; a compliance rule pushes transactions into opaque channels; a platform suspension drives users to venues less susceptible to oversight; a perimeter fence creates brittle dependencies on a few guarded gates. The choreography of protection and exposure thus cannot be captured by a simple binary of safety versus threat. Ethnography is uniquely positioned to register how risk imaginaries meet risk infrastructures in everyday practice, and how security projects recalibrate moral economies by rendering some persons suspicious and others trusted by default. Because chokepoints render rules visible at scale, they reveal how sovereignty is exercised in dispersed form. Sovereignty here appears not only as the state’s command over territory but as the power to authorise or deny passage through infrastructural constrictions. Private actors are not merely subjects of sovereign power; they are often its co-producers. Terminal operators, pipeline consortiums, platform companies, and financial clearinghouses write rules that have public effects. In a world where globalisation is channelled by privately owned infrastructures, chokepoints become sites where public and private authority entangle. Border preclearance in foreign airports projects state power outward via infrastructural agreements; content delivery networks throttle or privilege traffic based on commercial contracts that remake the publicness of the internet; customs zones within ports carve special jurisdictions into cityscapes. Such examples demonstrate how chokepoints knit governance across scales and blur the coordinates of accountability. The study of chokepoints also illuminates labour in the age of logistics. Much scholarship on logistics rightly emphasises the orchestration of supply chains and the optimisation of circulation. The anthropology of chokepoints, by contrast, foregrounds the sites where optimisation meets its limits and depends on embodied work to save throughput from its own bottlenecks. The crane operator who stretches a shift to move a delayed vessel; the train crew who manually switch track when signalling fails; the truckers who congregate in messaging groups to share tips about gate delays; the overnight team at a platform who must decide whether a flood of uploads is a genuine trend or coordinated manipulation; the bank compliance analyst who calls a customer to verify a transfer; the pipeline technician who hikes to a valve after storms take down telemetry: these workers absorb the shocks of queuing systems and retransmit movement into the circuit. Strikes and slowdowns at gates, locks, or terminals, walkouts by content moderators, refusals by delivery drivers to enter unsafe neighbourhoods, conscientious objections by engineers asked to deploy surveillance features: these are chokepoint politics par excellence. They remind us that chokepoints are not natural features but chosen designs that can be reconfigured by collective action. Ethnographic attention to chokepoints reveals affective textures often missed in macro accounts. Boredom in waiting rooms, anxiety at checkpoints, pride in moving a backlog, shame at being singled out for extra screening, relief at a “cleared” message, resignation when a hold cannot be explained, camaraderie among those who share line time: these affects are not epiphenomenal but part of how chokepoints work. People train themselves to endure waiting or to game it; they cultivate small rituals—stretching, scrolling, joking, complaining—to metabolise delay. The phenomenology of standing in line, idling at anchor, watching a spinning wheel icon, or pausing at a turnstile enters the anthropology of the constricted contemporary as data about how power and subjectivity are made. In this sense, chokepoints are schools of citizenship, albeit segregated ones, where different populations learn what they can expect from institutions of passage. The ecological dimensions of chokepoints have become increasingly visible as climate change alters the conditions of movement. Drought reduces river depth and constricts barge traffic; heat buckles rail lines and limits the loads aircraft can carry; storms close channels; fire season turns highways into convoys; melting permafrost destabilises pipelines; sea-level rise demands new floodgates at port entrances; warming oceans shift the geography of fisheries so that traditional maritime chokepoints are overlaid by ecological ones. In each case, infrastructures designed for one regime of variability encounter another, and chokepoints multiply. Energy transitions also produce new chokepoints—rare earth mining sites, battery supply corridors, grid interconnections—that entangle material extraction with carbon politics. The anthropology of chokepoints thus intersects with environmental anthropology in following how nonhuman forces and actors—tides, temperatures, species, sediments—contribute to constriction and redistribute costs and benefits of passage. Digital infrastructures add an additional layer of constriction that is both analogous to and distinct from the material gates of ports and pipelines. Application stores dictate the terms by which software reaches users; platform ranking algorithms act as chokepoints to visibility and monetisation; content moderation queues are chokepoints to publicity; domain registrars and certificate authorities are chokepoints to website existence; internet exchange points and submarine cable landing stations function as chokepoints for data circulation; rate limits and throttling policies are explicit constraints on speed. The moral economy of the digital chokepoint is emergent and contested: users demand frictionless experience while also calling for safety, moderation, privacy, and accountability, each of which is implemented by adding layers of chokepoints. The anthropology of the digital chokepoint therefore refuses a facile rhetoric of “openness” and asks which gates are justified, by whom, and with what recourse when decisions are wrong. It also attends to how digital chokepoints fold back into physical movement, as when gig workers’ access to jobs is mediated by platform queues or when biometric databases at borders generate lags that produce hunger in migrant shelters downstream from a crossing. The financial world offers another set of chokepoints whose operations illuminate the patterned constrictions of the contemporary. Payment processors decide the legitimacy of entire industries by cutting service; clearinghouses concentrate counterparty risk behind a promise of finality that depends on emergency powers; currency controls funnel demand into unofficial channels; credit scoring algorithms gate access to capital with opaque rationales. These are not mere bureaucratic frictions; they are instruments of macro-political strategy and moral regulation, often transnational in reach but intimate in effect. Ethnographies of finance at chokepoints—pawnshops that bridge the gap when formal payments stall; hawala networks that smooth over sanctions; informal credit lines extended by shop owners to customers caught in digital outages—disclose how people craft alternative circuits when formal arteries constrict. Methodologically, the anthropology of chokepoints encourages multi-sited work oriented by the movements that constrictions redirect rather than by static spatial units. Following a container from a delayed vessel to an inland depot, shadowing a bank transfer from a flagged transaction to a compliance call centre, accompanying a migrant from a stalled queue to a new line a hundred kilometres away, tracking a video from a demonetisation appeal to the restoration of ads: such itineraries materialise the distributive effects of constriction. They also demand a mixed repertoire of methods: participant observation in boring places, interviews with workers whose craft is discretion, document analysis of rulebooks and contracts, sensory ethnography of noise and light in waiting zones, time-motion studies of queues, interface ethnography of dashboards, and collaboration with engineers and planners who design gating mechanisms. If chokepoints are designed, they are also re-designed. Many actors actively engineer friction as a tool of justice. Activists engage in “chokepoint politics” when they block a road, slow a supply chain, or flood a review queue to make a distant issue materially salient to those who benefit from it. Workers use strategic slowdowns to induce bargaining; communities refuse wayleaves to extract concessions; environmentalists target valves and banks that fund pipelines; digital rights activists contest platform throttling; consumer groups organise “chargeback storms” to pressure merchants. Such actions exploit the very multipliers that make chokepoints consequential, and they invite anthropology to interrogate norms about legitimate obstruction. When does creating friction for some safeguard mobility for others? When does constriction redistribute risk downward, and when does it push it back up the chain? These normative questions are not afterthoughts; they are internal to the analysis because chokepoints are as much moral instruments as technical ones. A small device—a stamp, a token, a password—can open or close futures; a minor policy tweak in a gate protocol can tilt an industry; a short-lived outage can ripple through global logistics for weeks; a two-hour storm can generate a month of scheduling backlogs; a revised algorithm can demote communities to invisibility; a re-insurance clause can halt reconstruction after disaster. Because chokepoints are leverage points in complex systems, they are sources of both fragility and control. The grammar of chokepoints is therefore both empirical and analogical: empirical because the specificity of devices and practices matters; analogical because thinking across sites allows us to build a comparative language for constriction that travels from the harbour to the hospital, from the tunnel to the timeline, from the pipeline to the platform. It is in this comparative spirit that we can propose a set of dimensions through which chokepoints might be studied across domains. First, chokepoints have scalar-relay effects. They translate the local into the global by virtue of positionality, multiplying the consequences of small acts because they sit on obligatory passages. Second, chokepoints are temporal modulators. They recalibrate rhythms by synchronising disparate processes through timetables, windows, and timeouts. An ethnography of modulation will attend to shift work, maintenance windows, curfews, opening hours, and the social arts of waiting that choreograph line time. Third, chokepoints are material-semiotic interfaces. They conjoin devices and meanings such that a “hold” carries both a technical cause code and an affective message to those delayed. An ethnography of such interfaces will trace how signage, dashboards, seals, and stamps translate complex processes into actionable statuses experienced as fate or as negotiable. Fourth, chokepoints are labour regimes. They create and depend upon crafts of discretion and endurance, producing communities of practice and leverage positions for collective bargaining. An ethnography of labour here will analyse how training, certification, overtime, and fatigue shape throughput, and how workers interpret and manipulate queues. Fifth, chokepoints are governance assemblages. They are stitched from jurisdictions, standards, and contracts in which public and private powers co-produce rules. An ethnography of governance will follow how accountability is assigned when delays occur, how audits translate into operational change, and how rule conflicts are resolved where sovereignties overlap. Sixth, chokepoints are moral economies. They organise expectations about fairness—first come, first served; premium lines; humanitarian corridors—and they elicit judgments about cutting, bribery, triage, and exceptions. An ethnography of moral economy will examine how values are enacted in the prioritisation of some flows over others and how legitimacy is won or lost at the gate. Seventh, chokepoints are ecological nodes. They concentrate externalities—noise, emissions, spills, and waste—while also being vulnerable to environmental variability. An ethnography of ecology will connect queue emissions to surrounding communities, track how climate stress reconfigures gating practices, and ask how nonhuman actors shape constriction. Eighth, chokepoints are design problems and opportunities. They can be reconfigured toward equity and resilience through experiments in distributed capacity, redundant routing, “graceful degradation,” transparent appeals, and participatory governance of gates. An ethnography of design will not only critique existing chokepoints but will collaborate with practitioners to prototype alternatives, evaluating how interventions redistribute time, risk, and dignity. Thinking with these eight dimensions helps answer the guiding questions: What do chokepoints do? They concentrate decision-making and redistribute risk by converting the promise of circulation into governed passage. How do they do it? Through the interlacing of devices, rules, and crafts that narrow flows into auditable, taxable, stoppable channels whose very constriction sustains movement. When do they do it? At moments of synchronisation and exception—during surges, outages, seasonal cycles, audits, storms—when line time becomes the dominant experience of globalisation. For whom? For states seeking governability; for firms seeking profit; for workers seeking leverage; for communities seeking safety; and for travellers, migrants, creators, and customers who daily learn to live in the queue. An anthropology of chokepoints thus recasts the optic of flow by revealing that traffic, not free movement or total blockage, is the ordinary modality of the present. Such a reframing also clarifies the stakes for scholarship and for public life. To the extent that social theory has been enchanted by circulation, it has sometimes overlooked the normality of constraint and the everydayness of waiting. The constricted contemporary does not deny globalisation; it describes the infrastructures that make it governable. It insists that we study the valves and not just the pipes, the gates and not just the roads, the dashboards and not just the datasets. It asks that we examine how shame and pride adhere to statuses in queue, how boredom and fear are cultivated by routine delay, how dignity is eroded by arbitrary holds and restored by fair appeals. It alerts us to how inequality is materialised when premium lanes accelerate the already fast and deferred maintenance slows the already slow, how “trusted traveler” and “trusted merchant” programmes allocate scarce speed, how “know your customer” and sanctions regimes enact moral geographies via financial chokepoints, how content moderation channels reshape publics by turning visibility into a rationed asset. And it counsels humility by reminding us that even the most elaborate systems are held together by people who toggle switches, stamp forms, and answer calls, whose crafts of discretion repair circulation when it chokes. The anthropology of chokepoints therefore proposes not an antithesis to flow but an analytic of traffic: of movement under conditions of queued uncertainty, moral judgment, and asymmetric power. It invites us to tell stories of bridges and borders, of screens and seals, of doors and dashboards, in order to grasp how the world’s celebrated openness is built by and dependent upon constriction. It charges us to render visible the quiet labour that keeps queues from becoming crises, while not romanticising the power that chokepoints give to some over others. It likewise cautions against an unreflective embrace of friction as a political tactic by asking whose lives get slowed by tactics aimed at the powerful. And it suggests that the durability of the contemporary will be measured not by the absence of bottlenecks but by the fairness with which chokepoints allocate time, distribute risk, and admit appeal. Chokepoints, in this light, are not anomalies to be eradicated; they are the patterned sites where the terms of belonging and exchange are negotiated, however unequally, in the lanes and lobbies through which a globalising world must pass. To study them is to study our present not as a torrent or a dam but as a series of valves, whose opening and closing is the daily work through which circulation becomes a life worth living— or a wait that cannot be borne.

"""

def parse_targets(args) -> List[str]:
    if args.targets:
        return [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    return [args.target.lower()]

def main():
    parser = argparse.ArgumentParser(description="LRC Text Differentiator — Proof of Concept (enhanced)")
    parser.add_argument("--context-terms", type=str,
                        help='Comma-separated terms to assess context fit (e.g., "myth,history,he")')
    parser.add_argument("--context-topk", type=int, default=10, help="Top-K predictions to consider for context fit")
    parser.add_argument("--context-maxevals", type=int, default=200, help="Max occurrences to score for context fit")
    parser.add_argument("--text1", type=str, help="Path to first text file (optional)")
    parser.add_argument("--text2", type=str, help="Path to second text file (optional)")
    parser.add_argument("--target", type=str, default=TARGET_WORD_DEFAULT, help="Single target word (fallback)")
    parser.add_argument("--targets", type=str, help="Comma-separated list of target words; overrides --target")
    parser.add_argument("--presence-mode", choices=["or","gap","density"], default="or",
                        help="Encoding for multi-target presence: OR (0/1), inter-arrival gaps, or windowed density")
    parser.add_argument("--window", type=int, default=128, help="Window size for density mode (tokens)")
    parser.add_argument("--min-rs", type=int, default=MIN_RS_POINTS, help="Minimum series length to use RS Hurst")
    parser.add_argument("--dfa-only", action="store_true", help="Force DFA for all encodings")
    parser.add_argument("--surrogates", action="store_true", help="Compute surrogate ΔH for each encoding")
    parser.add_argument("--weights", type=str, help="Comma-separated weights (presence, sentlen, func, punct, drift)")
    parser.add_argument("--max-sents", type=int, default=5000, help="Cap on sentences for semantic drift")
    args = parser.parse_args()

    # Load texts
    def load(p, default):
        if p:
            key = p.strip().upper()
            if key == "HUMAN_TEXT":
                return HUMAN_TEXT
            if key == "GPT_TEXT":
                return GPT_TEXT
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        return default  # <- make sure this is aligned with 'def', not inside 'with'

    t1 = load(args.text1, HUMAN_TEXT)
    t2 = load(args.text2, GPT_TEXT)

    targets = parse_targets(args)  # assumes you already have this helper

    # Optional weights parsing (must match 5 encodings in order)
    weights = None
    if args.weights:
        try:
            w = [float(x) for x in args.weights.split(",")]
            if len(w) == 5:
                weights = w
        except Exception:
            pass

    if t1 and t2:
        # --- Compare mode ---
        compare_texts(
            t1, t2,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            weights=weights,
            max_sents=args.max_sents,
            context_terms=args.context_terms,
            context_topk=args.context_topk,
            context_maxevals=args.context_maxevals,
        )

    elif t1 and not t2:
        # --- Single Text Analysis ---
        res = analyze_text(
            t1,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            max_sents=args.max_sents
        )
        print("\n=== Single Text Analysis ===")
        for name, stats in res.encoding_values.items():
            line = f"{name:24s}: {stats.value:.3f}  [{stats.method}]  (n={stats.n})"
            if getattr(stats, "delta", None) is not None and getattr(stats, "surrogate", None) is not None:
                line += f"  ΔH={stats.delta:+.3f} (sur={stats.surrogate:.3f})"
            print(line)
        if getattr(res, "punc_fano", None) is not None:
            n_punc = res.encoding_values.get("punctuation_cadence", EncodingStat(0,"",0)).n
            print(f"{'punctuation_fano':24s}: {res.punc_fano:.3f}  (n={n_punc})")
        print(f"Aggregate (mean): {res.aggregate_score:.3f}")
        print(f"Verdict: {res.verdict}")

        # Optional Context Fit for single text only
        if args.context_terms:
            terms = [t.strip() for t in args.context_terms.split(",") if t.strip()]
            try:
                cfs = context_fit_scores(t1, terms, topk=args.context_topk, max_evals=args.context_maxevals)
                if cfs:
                    print(f"Context Fit (Single Text, mean over {len(cfs)} hits): {np.mean(cfs):.3f}  [1=strong, 0=weak]")
                else:
                    print("Context Fit: no occurrences of provided terms found.")
            except Exception as e:
                print(f"Context Fit: unavailable ({e})")

    else:
        # --- Default: built-in comparison ---
        compare_texts(
            HUMAN_TEXT, GPT_TEXT,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            weights=weights,
            max_sents=args.max_sents,
            context_terms=args.context_terms,
            context_topk=args.context_topk,
            context_maxevals=args.context_maxevals,
        )

if __name__ == "__main__":
    main()