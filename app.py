# app.py — Urdu → Roman Transliteration (Streamlit, single-language UI)
import os
import re
import json
import unicodedata
import torch
import torch.nn as nn
import streamlit as st

# ==============================
# 0) CONFIG — EDIT THESE PATHS
# ==============================
BEST_CKPT = "bilstm_wp_best.pt"  # your trained .pt
WP_JSON   = "roman_wp-tokenizer.json"                  # target WordPiece tokenizer JSON
SRC_VOCAB_TXT = "roman_wp-vocab.txt"                                         # one token per line; must include <pad>, <unk>

# Model dims are usually saved in the checkpoint config; these are fallback defaults:
FALLBACK = dict(EMB_DIM_SRC=256, EMB_DIM_TGT=256, HID_DIM=512, ENC_LAYERS=2, DEC_LAYERS=3, DROPOUT=0.3)

# ==============================
# 1) UTILITIES
# ==============================
def clean_urdu(text: str) -> str:
    """Minimal clean/normalize for Urdu input (keep Urdu letters/spaces, standardize variants)."""
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)  # keep Urdu & spaces
    text = re.sub(r"\s+", " ", text).strip()
    # common normalizations
    text = text.replace("ك", "ک").replace("ي", "ی").replace("ە", "ہ").replace("ة", "ہ").replace("ئ", "ی")
    return text

def load_src_vocab(path_txt):
    tokens = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if tok:
                tokens.append(tok)
    token2id = {t: i for i, t in enumerate(tokens)}
    if "[PAD]" not in token2id or "[UNK]" not in token2id:
        raise RuntimeError("src_vocab.txt must contain [PAD] and [UNK] (exact strings).")
    return tokens, token2id, token2id["[PAD]"], token2id["[UNK]"]

def encode_src_chars(s, token2id, unk_id, max_len=256):
    chs = list(str(s))[:max_len]
    return torch.tensor([token2id.get(c, unk_id) for c in chs], dtype=torch.long)

# ==============================
# 2) MODEL DEFINITIONS
# ==============================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim // 2, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True, batch_first=True
        )
    def forward(self, src, src_len):
        e = self.emb_drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(e, src_len.cpu(), batch_first=True, enforce_sorted=False)
        h, (hn, cn) = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        def _merge(x):
            layers = []
            for l in range(0, x.size(0), 2):
                layers.append(torch.cat([x[l], x[l+1]], dim=-1))
            return torch.stack(layers, dim=0)
        return h, (_merge(hn), _merge(cn))

class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.Wa = nn.Linear(hid_dim, hid_dim, bias=False)
    def forward(self, dec_h, enc_out, src_mask):
        q = self.Wa(dec_h).unsqueeze(1)                                # [B,1,H]
        scores = torch.bmm(q, enc_out.transpose(1,2)).squeeze(1)       # [B,S]
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers=n_layers,
                            dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.attn = LuongAttention(hid_dim)
        self.out_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hid_dim, vocab_size)
    def forward(self, dec_inp, enc_out, src_mask, hidden):
        B, T = dec_inp.shape
        e = self.emb_drop(self.emb(dec_inp))
        outputs = []
        h, c = hidden
        for t in range(T):
            dec_h_top = h[-1]
            ctx, _ = self.attn(dec_h_top, enc_out, src_mask)
            x_t = torch.cat([e[:, t, :], ctx], dim=-1).unsqueeze(1)
            o, (h, c) = self.lstm(x_t, (h, c))
            outputs.append(self.proj(self.out_drop(o.squeeze(1))))
        return torch.stack(outputs, dim=1), (h, c)

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_id_src):
        super().__init__()
        self.enc, self.dec, self.pad_id_src = enc, dec, pad_id_src
    def _bridge(self, hn, cn):
        L_enc = hn.size(0); L_dec = self.dec.lstm.num_layers
        if L_enc == L_dec: return hn, cn
        if L_enc < L_dec:
            pad_h = hn.new_zeros(L_dec-L_enc, hn.size(1), hn.size(2))
            pad_c = cn.new_zeros(L_dec-L_enc, cn.size(1), cn.size(2))
            return torch.cat([hn, pad_h], 0), torch.cat([cn, pad_c], 0)
        return hn[-L_dec:], cn[-L_dec:]
    def forward(self, src, src_len, dec_inp):
        enc_out, (hn, cn) = self.enc(src, src_len)
        hn, cn = self._bridge(hn, cn)
        src_mask = (src != self.pad_id_src)
        logits, _ = self.dec(dec_inp, enc_out, src_mask, (hn, cn))
        return logits

# ==============================
# 3) LOADING (cached)
# ==============================
@st.cache_resource(show_spinner=True)
def load_all():
    # Tokenizer (HuggingFace Tokenizers)
    try:
        from tokenizers import Tokenizer
    except Exception:
        raise RuntimeError("Please install `tokenizers` package.")

    assert os.path.exists(BEST_CKPT), f"Checkpoint not found: {BEST_CKPT}"
    assert os.path.exists(WP_JSON),   f"Tokenizer JSON not found: {WP_JSON}"
    assert os.path.exists(SRC_VOCAB_TXT), f"Source vocab not found: {SRC_VOCAB_TXT}"

    tokenizer = Tokenizer.from_file(WP_JSON)
    TGT_V = tokenizer.get_vocab_size()

    SRC_TOKENS, SRC_token2id, pad_id_src, unk_id_src = load_src_vocab(SRC_VOCAB_TXT)
    SRC_V = len(SRC_TOKENS)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    cfg = ckpt.get("config", {})

    # Resolve special ids
    pad_id_tgt = cfg.get("pad_id_tgt", tokenizer.token_to_id("[PAD]"))
    eos_id_tgt = cfg.get("eos_id_tgt", tokenizer.token_to_id("[EOS]"))
    bos_id_tgt = cfg.get("bos_id_tgt",
        tokenizer.token_to_id("[CLS]") or tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("[MASK]"))
    if bos_id_tgt in (None, pad_id_tgt, eos_id_tgt):
        # last resort: use [UNK] as BOS if distinct
        unk = tokenizer.token_to_id("[UNK]")
        if unk not in (None, pad_id_tgt, eos_id_tgt):
            bos_id_tgt = unk
        else:
            raise RuntimeError("BOS id is invalid; ensure tokenizer has [CLS]/[SEP]/[MASK]/[UNK].")

    # Hyperparams (prefer from ckpt)
    EMB_DIM_SRC = cfg.get("EMB_DIM_SRC", FALLBACK["EMB_DIM_SRC"])
    EMB_DIM_TGT = cfg.get("EMB_DIM_TGT", FALLBACK["EMB_DIM_TGT"])
    HID_DIM     = cfg.get("HID_DIM",     FALLBACK["HID_DIM"])
    ENC_LAYERS  = cfg.get("ENC_LAYERS",  FALLBACK["ENC_LAYERS"])
    DEC_LAYERS  = cfg.get("DEC_LAYERS",  FALLBACK["DEC_LAYERS"])
    DROPOUT     = cfg.get("DROPOUT",     FALLBACK["DROPOUT"])

    # Build model and load weights
    encoder = Encoder(SRC_V, EMB_DIM_SRC, HID_DIM, ENC_LAYERS, DROPOUT, pad_id_src).to(device)
    decoder = Decoder(TGT_V, EMB_DIM_TGT, HID_DIM, DEC_LAYERS, DROPOUT, pad_id_tgt).to(device)
    model = Seq2Seq(encoder, decoder, pad_id_src).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    helpers = dict(
        tokenizer=tokenizer, SRC_token2id=SRC_token2id,
        pad_id_src=pad_id_src, unk_id_src=unk_id_src,
        pad_id_tgt=pad_id_tgt, bos_id_tgt=bos_id_tgt, eos_id_tgt=eos_id_tgt
    )
    return model, helpers, device

# ==============================
# 4) DECODERS
# ==============================
@torch.no_grad()
def greedy_decode(model, helpers, device, text, max_len=256):
    tok = helpers["tokenizer"]
    src_ids = encode_src_chars(text, helpers["SRC_token2id"], helpers["unk_id_src"]).unsqueeze(0).to(device)
    src_len = torch.tensor([src_ids.size(1)], dtype=torch.long, device=device)
    enc_out, (hn, cn) = model.enc(src_ids, src_len)
    # bridge
    L_enc = hn.size(0); L_dec = model.dec.lstm.num_layers
    if L_enc < L_dec:
        pad_h = hn.new_zeros(L_dec - L_enc, hn.size(1), hn.size(2))
        pad_c = cn.new_zeros(L_dec - L_enc, cn.size(1), cn.size(2))
        h, c = torch.cat([hn, pad_h], 0), torch.cat([cn, pad_c], 0)
    elif L_enc > L_dec:
        h, c = hn[-L_dec:], cn[-L_dec:]
    else:
        h, c = hn, cn
    src_mask = (src_ids != helpers["pad_id_src"])
    dec = torch.tensor([[helpers["bos_id_tgt"]]], dtype=torch.long, device=device)
    out_ids = []
    for _ in range(max_len):
        logits, (h, c) = model.dec(dec, enc_out, src_mask, (h, c))
        nxt = int(logits[:, -1, :].argmax(dim=-1))
        if nxt == helpers["eos_id_tgt"]: break
        if nxt != helpers["pad_id_tgt"]: out_ids.append(nxt)
        dec = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return tok.decode(out_ids, skip_special_tokens=True).strip()

@torch.no_grad()
def beam_decode(model, helpers, device, text, max_len=256, beam_size=5, len_penalty=0.6):
    from math import log
    tok = helpers["tokenizer"]
    src_ids = encode_src_chars(text, helpers["SRC_token2id"], helpers["unk_id_src"]).unsqueeze(0).to(device)
    src_len = torch.tensor([src_ids.size(1)], dtype=torch.long, device=device)
    enc_out, (hn, cn) = model.enc(src_ids, src_len)
    # bridge
    L_enc = hn.size(0); L_dec = model.dec.lstm.num_layers
    if L_enc < L_dec:
        pad_h = hn.new_zeros(L_dec - L_enc, hn.size(1), hn.size(2))
        pad_c = cn.new_zeros(L_dec - L_enc, cn.size(1), cn.size(2))
        h0, c0 = torch.cat([hn, pad_h], 0), torch.cat([cn, pad_c], 0)
    elif L_enc > L_dec:
        h0, c0 = hn[-L_dec:], cn[-L_dec:]
    else:
        h0, c0 = hn, cn
    src_mask = (src_ids != helpers["pad_id_src"])

    beams = [(0.0, [], h0, c0, torch.tensor([[helpers["bos_id_tgt"]]], device=device))]
    finished = []
    for _ in range(max_len):
        new_beams = []
        for score, seq, h, c, last in beams:
            logits, (h2, c2) = model.dec(last, enc_out, src_mask, (h, c))
            logp = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk = torch.topk(logp, k=min(beam_size*2, logp.size(0)))
            for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                if idx == helpers["pad_id_tgt"]: continue
                seq2 = seq + [idx]; score2 = score + lp
                if idx == helpers["eos_id_tgt"]:
                    L = max(1, len(seq))
                    finished.append((score2 / (L ** len_penalty), seq))
                else:
                    new_beams.append((score2, seq2, h2, c2, torch.tensor([[idx]], device=device)))
        if not new_beams: break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
        if len(finished) >= beam_size: break
    best = []
    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        best = finished[0][1]
    elif beams:
        best = beams[0][1]
    return tok.decode(best, skip_special_tokens=True).strip()

# ==============================
# 5) STREAMLIT UI
# ==============================
st.set_page_config(page_title="Urdu → Roman Transliteration", page_icon="🔤", layout="centered")

st.title("🔤 Urdu → Roman Transliteration")
st.caption("Type a misrah (Urdu) and get its Roman transliteration. Uses your trained BiLSTM seq2seq model.")

with st.sidebar:
    st.markdown("### Settings")
    decode_mode = st.radio("Decoding", options=["Beam (recommended)", "Greedy"], index=0)
    beam_size = st.slider("Beam size", min_value=2, max_value=10, value=5, step=1, help="Only used for Beam decoding.")
    max_len = st.slider("Max output length", min_value=64, max_value=512, value=256, step=32)

try:
    model, helpers, device = load_all()
    user_text = st.text_area("✍️ Enter misrah (Urdu):", height=120, placeholder="یہ جہاں خواب ہے یا خواب کا جہاں کچھ ہے ...")
    if st.button("Transliterate"):
        if not user_text.strip():
            st.warning("Please enter a misrah first.")
        else:
            with st.spinner("Transliterating..."):
                inp = clean_urdu(user_text)
                if decode_mode.startswith("Beam"):
                    out = beam_decode(model, helpers, device, inp, max_len=max_len, beam_size=beam_size)
                else:
                    out = greedy_decode(model, helpers, device, inp, max_len=max_len)
            st.markdown("### 🌿 Roman Transliteration")
            st.code(out or "(empty)", language=None)
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Check your paths in the CONFIG section at the top of this file.")

st.markdown("---")
st.caption("Tip: ensure `BEST_CKPT`, `WP_JSON`, and `src_vocab.txt` paths are correct. The tokenizer JSON must contain [PAD]/[EOS] and ideally [CLS]/[SEP]/[MASK] for BOS.")

