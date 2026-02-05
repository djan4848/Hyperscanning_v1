#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

EPS = 1e-12

@dataclass
class Utt:
    i0: int
    i1: int
    t0: float
    t1: float
    text: str
    log_ratio: float = float("nan")       # combined
    log_ratio_onset: float = float("nan")
    log_ratio_full: float = float("nan")
    # text features
    is_q: int = 0
    is_yn: int = 0
    is_repair: int = 0
    n_words: int = 0


def rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x * x) + EPS))

def norm_text(s: str) -> str:
    return str(s).strip().lower()

def build_utterances(df: pd.DataFrame, gap_s: float) -> List[Utt]:
    df = df.reset_index(drop=True)
    utts: List[Utt] = []
    start = 0
    for i in range(1, len(df)):
        prev_end = float(df.loc[i - 1, "end_time"])
        cur_start = float(df.loc[i, "start_time"])
        gap = cur_start - prev_end

        w_prev = str(df.loc[i - 1, "word"])
        ends_sentence = w_prev.endswith((".", "?", "!"))

        if gap > gap_s or ends_sentence:
            i0, i1 = start, i - 1
            t0 = float(df.loc[i0, "start_time"])
            t1 = float(df.loc[i1, "end_time"])
            text = " ".join(df.loc[i0:i1, "word"].astype(str).tolist())
            utts.append(Utt(i0=i0, i1=i1, t0=t0, t1=t1, text=text))
            start = i

    i0, i1 = start, len(df) - 1
    t0 = float(df.loc[i0, "start_time"])
    t1 = float(df.loc[i1, "end_time"])
    text = " ".join(df.loc[i0:i1, "word"].astype(str).tolist())
    utts.append(Utt(i0=i0, i1=i1, t0=t0, t1=t1, text=text))
    return utts

def slice_audio(x, sr, t0, t1, pad_s):
    t0p = max(0.0, t0 - pad_s)
    t1p = min(len(x) / sr, t1 + pad_s)
    i0 = int(math.floor(t0p * sr))
    i1 = int(math.ceil(t1p * sr))
    i1 = max(i1, i0 + 1)
    return x[i0:i1]

def compute_enhanced_channels(stereo: np.ndarray):
    L = stereo[:, 0].astype(np.float64).copy()
    R = stereo[:, 1].astype(np.float64).copy()

    gL, gR = rms(L), rms(R)
    if gL > 0 and gR > 0:
        L *= (gR / gL)

    alpha = float(np.dot(L, R) / (np.dot(R, R) + EPS))
    beta  = float(np.dot(R, L) / (np.dot(L, L) + EPS))

    A = L - alpha * R
    B = R - beta * L
    return np.column_stack([A.astype(np.float32), B.astype(np.float32)]), alpha, beta

def text_features(u: Utt):
    t = u.text.strip()
    tl = norm_text(t)
    u.n_words = len(t.split())

    u.is_q = 1 if t.endswith("?") else 0
    u.is_yn = 1 if tl in ("yes", "yes.", "no", "no.") else 0
    u.is_repair = 1 if tl.startswith(("sorry", "i mean", "wait", "no sorry", "uh", "um", "well")) else 0

def compute_log_ratios(utts: List[Utt], enh: np.ndarray, sr: int, pad_s: float, onset_s: float, w_onset: float):
    for u in utts:
        # FULL
        A_full = slice_audio(enh[:, 0], sr, u.t0, u.t1, pad_s)
        B_full = slice_audio(enh[:, 1], sr, u.t0, u.t1, pad_s)
        u.log_ratio_full = float(np.log((rms(A_full) + EPS) / (rms(B_full) + EPS)))

        # ONSET
        t_on_end = min(u.t1, u.t0 + onset_s)
        A_on = slice_audio(enh[:, 0], sr, u.t0, t_on_end, pad_s=0.0)
        B_on = slice_audio(enh[:, 1], sr, u.t0, t_on_end, pad_s=0.0)
        u.log_ratio_onset = float(np.log((rms(A_on) + EPS) / (rms(B_on) + EPS)))

        u.log_ratio = float(w_onset * u.log_ratio_onset + (1.0 - w_onset) * u.log_ratio_full)

        text_features(u)

def log_sigmoid(z):
    # log(sigmoid(z)) stable
    if z >= 0:
        return -np.log1p(np.exp(-z))
    return z - np.log1p(np.exp(z))

def log1m_sigmoid(z):
    # log(1-sigmoid(z)) stable
    if z >= 0:
        return -z - np.log1p(np.exp(-z))
    return -np.log1p(np.exp(z))

def viterbi_roles(utts: List[Utt],
                  k_acoustic: float,
                  w_text: float,
                  w_acoustic: float,
                  p_switch: float,
                  p_stay: float) -> np.ndarray:
    """
    States: 0=Q (questioner role), 1=R (responder role)

    Emission combines:
      - text-likelihood (strong): questions -> Q ; yes/no -> R ; repairs -> Q
      - acoustic-likelihood (weak): log_ratio sign suggests which physical channel dominates,
        BUT that's used later for mapping Q/R -> A/B. Here we only keep it weakly as 'consistency'
        by favoring Q to have positive-ish ratios IF mapping is Q->A, etc. We don't know mapping yet.
        So we ignore absolute channel and just use acoustic magnitude as 'confidence' weight.

    We'll encode text as log-probs:
      Q: + is_q, + is_repair, - is_yn
      R: + is_yn, - is_q
    """
    T = len(utts)
    # transition log-probs
    # prefer switch in ping-pong but allow stays
    logP = np.array([
        [np.log(p_stay),   np.log(p_switch)],  # Q->Q, Q->R
        [np.log(p_switch), np.log(p_stay)]     # R->Q, R->R
    ], dtype=np.float64)

    # emission log-probs
    em = np.zeros((2, T), dtype=np.float64)

    for t, u in enumerate(utts):
        # text score
        # start from neutral
        sQ = 0.0
        sR = 0.0

        # questions push to Q strongly
        if u.is_q:
            sQ += 2.0
            sR -= 1.5

        # yes/no push to R strongly
        if u.is_yn:
            sR += 2.5
            sQ -= 2.0

        # repairs/metacommentary are typically Q (same speaker continuing)
        if u.is_repair:
            sQ += 1.2
            sR -= 0.6

        # very short utterances (1–2 words) are more likely R unless they are a question
        if u.n_words <= 2 and not u.is_q and not u.is_repair:
            sR += 0.6

        # turn these into log-prob-like values
        # use softmax in log-space
        mx = max(sQ, sR)
        lZ = mx + np.log(np.exp(sQ - mx) + np.exp(sR - mx))
        lQ_text = sQ - lZ
        lR_text = sR - lZ

        # acoustic confidence = |log_ratio|
        # we don't use sign here (mapping unknown). Just reward having stronger evidence slightly.
        ac = min(abs(u.log_ratio), 2.0)  # cap
        lQ = w_text * lQ_text + w_acoustic * (0.15 * ac)
        lR = w_text * lR_text + w_acoustic * (0.15 * ac)

        em[0, t] = lQ
        em[1, t] = lR

    # Viterbi DP
    dp = np.full((2, T), -np.inf)
    back = np.zeros((2, T), dtype=np.int8)

    dp[:, 0] = em[:, 0]  # uniform prior

    for t in range(1, T):
        for s in (0, 1):
            candidates = dp[:, t - 1] + logP[:, s]
            j = int(np.argmax(candidates))
            dp[s, t] = candidates[j] + em[s, t]
            back[s, t] = j

    states = np.zeros(T, dtype=np.int8)
    states[T - 1] = int(np.argmax(dp[:, T - 1]))
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[states[t], t]

    return states  # 0=Q, 1=R

def choose_mapping(utts: List[Utt], roles: np.ndarray) -> Tuple[str, str]:
    """
    Decide whether Q->A,R->B OR Q->B,R->A based on acoustic sign agreement.
    We expect the questioner to be consistently stronger in one channel.
    """
    # score mapping by how often role aligns with sign(log_ratio)
    # If mapping Q->A: Q utterances should tend to have positive log_ratio, R negative
    # If mapping Q->B: opposite
    lr = np.array([u.log_ratio for u in utts], dtype=np.float64)
    isQ = (roles == 0)
    isR = ~isQ

    # Avoid zero influence
    w = np.clip(np.abs(lr), 0.0, 2.0)

    # mapping1: Q->A => Q positive, R negative
    s1 = 0.0
    s1 += np.sum(w[isQ] * np.sign(lr[isQ]))     # + if positive
    s1 += np.sum(w[isR] * np.sign(-lr[isR]))    # + if negative

    # mapping2: Q->B => Q negative, R positive
    s2 = 0.0
    s2 += np.sum(w[isQ] * np.sign(-lr[isQ]))
    s2 += np.sum(w[isR] * np.sign(lr[isR]))

    if s1 >= s2:
        return ("A", "B")  # Q->A, R->B
    return ("B", "A")      # Q->B, R->A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="dyad_stereo.wav")
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", default="words_diarized_v11_roles.csv")

    # utterance segmentation
    ap.add_argument("--gap_s", type=float, default=0.45)

    # acoustic features
    ap.add_argument("--pad_s", type=float, default=0.07)
    ap.add_argument("--onset_s", type=float, default=0.25)
    ap.add_argument("--w_onset", type=float, default=0.8)

    # role HMM params
    ap.add_argument("--w_text", type=float, default=1.0)
    ap.add_argument("--w_acoustic", type=float, default=0.15)
    ap.add_argument("--p_switch", type=float, default=0.70, help="Q<->R switch probability")
    ap.add_argument("--p_stay", type=float, default=0.30, help="Q->Q / R->R probability")

    args = ap.parse_args()

    df = pd.read_csv(args.words)
    for c in ["start_time", "end_time", "word"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    stereo, sr = sf.read(args.audio)
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise RuntimeError("Audio must be stereo with 2 channels (L/R).")

    enh, alpha, beta = compute_enhanced_channels(stereo)
    utts = build_utterances(df, gap_s=args.gap_s)

    compute_log_ratios(utts, enh, sr, args.pad_s, args.onset_s, args.w_onset)

    roles = viterbi_roles(
        utts,
        k_acoustic=3.0,
        w_text=args.w_text,
        w_acoustic=args.w_acoustic,
        p_switch=args.p_switch,
        p_stay=args.p_stay
    )

    q_to, r_to = choose_mapping(utts, roles)  # Q->A/B ; R->B/A
    print(f"[INFO] Utterances={len(utts)} | alpha={alpha:.4f} beta={beta:.4f}")
    print(f"[INFO] Mapping: Q->{q_to}, R->{r_to} | p_switch={args.p_switch}")

    utt_speakers = []
    for s in roles:
        utt_speakers.append(q_to if s == 0 else r_to)

    # assign per word
    word_labels = ["unknown"] * len(df)
    utt_id = np.full(len(df), -1, dtype=int)
    role_lbl = ["Q" if s == 0 else "R" for s in roles]

    for k, (u, sp) in enumerate(zip(utts, utt_speakers)):
        for j in range(u.i0, u.i1 + 1):
            word_labels[j] = sp
            utt_id[j] = k

    out_df = df.copy()
    out_df["speaker"] = word_labels
    out_df["utt_id"] = utt_id.tolist()
    out_df.to_csv(args.out, index=False)
    print("[OK] Wrote:", args.out)
    print(out_df["speaker"].value_counts(dropna=False))

    # diagnostics utterances
    utt_diag = pd.DataFrame([{
        "utt_id": k,
        "start_time": u.t0,
        "end_time": u.t1,
        "role": role_lbl[k],
        "speaker": utt_speakers[k],
        "log_ratio": u.log_ratio,
        "log_ratio_onset": u.log_ratio_onset,
        "log_ratio_full": u.log_ratio_full,
        "is_q": u.is_q,
        "is_yn": u.is_yn,
        "is_repair": u.is_repair,
        "n_words": u.n_words,
        "text": u.text
    } for k, u in enumerate(utts)])
    utt_out = args.out.replace(".csv", "_utterances.csv")
    utt_diag.to_csv(utt_out, index=False)
    print("[OK] Wrote:", utt_out)

if __name__ == "__main__":
    main()

