#!/usr/bin/env python3
"""
natural_prompts.py: a broad, natural IOI prompt set.

Fixed (the IOI logic): two referents are introduced (A, B); one of them (S)
is mentioned again before the end; the prompt ends where the other one (IO)
is the natural next word; the answer is one token.

Varied, mixed within the set:
  * 64 hand-written frames: narrative, dialogue, news, first person, present
    tense, long gaps and relative clauses, many actions
  * entity type per prompt: first names, or "the <noun>" (people and animals);
    both entities in a prompt are the same type, so the ending ("to" vs
    "to the") cannot give the answer away
  * how S returns: exact repeat, or a pronoun (he/she) when the two entities
    differ in gender
  * both orders of A and B

build_natural(tok, n, rng, frames, males, females, nouns_m, nouns_f, nouns_n,
              p_noun, p_pronoun) returns tokenized data in the format used by
cores.ioi.common.evaluate (ids, att, pos, last, f, s, y) plus metadata;
every prompt is checked automatically (exact mention counts, single-token
answers, S2 after both mentions) and dropped if it fails.

Run directly for the behavior check:
  PYTHONPATH=src python -m cores.ioi.frames.natural_prompts --model-name gpt2 --n 1200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# {A}, {B}: first mentions (A first); {S}: second mention of the subject, always
# the subject of the final action; frames end with "to" (" the" is appended for nouns).
FRAMES = [
    # narrative
    "When {A} and {B} went to the market, {S} bought some fruit and gave it to",
    "{A} and {B} were sitting on the porch. After a while, {S} poured a cup of tea and handed it to",
    "At the party, {A} and {B} were chatting by the window. {S} laughed and passed a glass to",
    "Late that night, {A} and {B} were still working in the office. Finally, {S} saved the file and sent it to",
    "{A} and {B} had been arguing all morning. Eventually {S} gave up and handed the remote to",
    "On the first day of school, {A} met {B} in the hallway. {S} smiled and offered a pencil to",
    "During the meeting, {A} and {B} disagreed about the budget. Afterward, {S} sent an apology to",
    "{A} and {B} found an old map in the attic. Excited, {S} showed it to",
    "While {A} and {B} were cooking dinner, {S} tasted the soup and passed the spoon to",
    "After the game, {A} and {B} walked home together. Along the way, {S} told a long story to",
    "{A} and {B} were waiting for the bus in the rain. {S} opened an umbrella and gave it to",
    "{A} and {B} were cleaning out the garage when {S} found a box of old letters and gave them to",
    "The train was crowded, so {A} and {B} had to stand. When a seat opened up, {S} offered it to",
    "{A} and {B} were studying for the exam. Around midnight, {S} closed the book and explained the last chapter to",
    "When {A} and {B} reached the top of the hill, {S} took out a camera and handed it to",
    "{A} and {B} were planting tomatoes in the garden. {S} dug a small hole and passed the seeds to",
    "Before the concert, {A} and {B} met at the cafe. {S} had bought two tickets and gave one to",
    "{A} and {B} were fixing the old bicycle. {S} tightened the last bolt and returned the wrench to",
    "{A} and {B} were lost in the city. Eventually {S} asked a stranger for directions and repeated them to",
    "When the power went out, {A} and {B} searched for candles. {S} lit the first one and handed it to",
    # dialogue
    "{A} and {B} were talking in the kitchen. {S} said, \"Here, try this,\" and handed a cookie to",
    "\"Is this yours?\" {A} asked as {B} walked in. {S} held out the wallet and gave it back to",
    "{A} and {B} were on the phone for an hour. Before hanging up, {S} promised to send the photos to",
    "\"We need to talk,\" said {A}. {B} nodded. Later that evening, {S} wrote a long letter to",
    "{A} turned to {B} and asked about the plan. After a pause, {S} explained everything to",
    "\"Can you help me with this?\" {A} asked. {B} agreed, and in the end {S} gave the finished drawing to",
    # news / workplace
    "In a statement on Monday, {A} and {B} announced a new partnership. According to the report, {S} will hand control of the company to",
    "The negotiations between {A} and {B} lasted for weeks. In the end, {S} agreed to sell the land to",
    "Both {A} and {B} applied for the job. After the interviews, {S} withdrew and offered support to",
    "{A} and {B} co-wrote the book. At the launch, {S} dedicated the first copy to",
    "After the trial, {A} and {B} left the courthouse separately. Outside, {S} waved to",
    "{A} and {B} were the last two in the office. Before leaving, {S} turned off the lights and said goodbye to",
    # first person
    "Yesterday I had lunch with {A} and {B}. At the end of the meal, {S} paid the bill and handed the receipt to",
    "I watched {A} and {B} play chess for an hour. When the game ended, {S} passed the trophy to",
    "Last week {A} and {B} helped me move. When we finished, {S} carried the last box over and handed it to",
    "I introduced {A} to {B} at the conference. Later that day, {S} sent a short email to",
    # stories / kids / animals
    "{A} and {B} were playing in the park. {S} kicked the ball to",
    "In the forest, {A} and {B} found a shiny stone. {S} picked it up and gave it to",
    "{A} and {B} were sharing a sandwich. {S} broke off a piece and offered it to",
    "At the farm, {A} and {B} were feeding the animals. {S} filled a bucket and handed it to",
    "When the storm started, {A} and {B} ran inside. {S} grabbed a towel and threw it to",
    "{A} and {B} were building a snowman. {S} found two buttons and gave them to",
    # long gaps / relative clauses
    "{A} and {B}, who had not seen each other in years, met at the station. {S}, still a little nervous, handed a small gift to",
    "When {A} and {B} finally arrived at the cabin after a long and tiring drive through the mountains, {S} unlocked the door and gave the keys to",
    "{A} and {B} spent the afternoon at the museum, looking at paintings and old maps. Before they left, {S} bought a postcard and gave it to",
    "After {A} and {B} had finished painting the fence, which took most of the weekend, {S} brought a cold drink to",
    "{A} and {B} were on a long hike. Halfway up the mountain, tired and hungry, {S} passed a water bottle to",
    "{A} and {B} had planned the trip for months, and when the day finally came, {S} printed the tickets and gave them to",
    # present tense
    "{A} and {B} are at the library. {S} finds a good book and recommends it to",
    "Every morning, {A} and {B} walk to work together. Today, {S} buys two coffees and gives one to",
    "{A} and {B} are setting up the tent. {S} holds the pole and passes the rope to",
    "{A} and {B} are watching a movie at home. {S} makes some popcorn and hands the bowl to",
    # misc
    "When {A} and {B} moved into the new apartment, {S} made a copy of the key and gave it to",
    "{A} and {B} were asked to work together. {S} read the question aloud to",
    "During the trip, {A} and {B} took turns driving. At the gas station, {S} handed the map to",
    "{A} and {B} both wanted the last slice of pizza. In the end, {S} gave it to",
    "At the wedding, {A} and {B} sat at the same table. {S} raised a glass and made a toast to",
    "{A} and {B} were organizing the bookshelf. {S} pulled out an old album and showed it to",
    "{A} and {B} were waiting at the airport. When the flight was delayed, {S} bought a magazine and gave it to",
    "On Sunday, {A} and {B} went fishing at the lake. {S} caught the first fish and proudly showed it to",
    "{A} and {B} were writing a song together. {S} hummed a new melody to",
    "{A} and {B} met for dinner downtown. Over dessert, {S} told a secret to",
    "{A} and {B} were decorating the tree. {S} found the star and passed it to",
    "When {A} and {B} got back from vacation, {S} printed the photos and mailed copies to",
    # varied endings: double object, direct object, other prepositions, reversed roles, clefts
    "{A} and {B} were packing for the trip. Without a word, {S} handed",
    "{A} and {B} were studying in the library. Around noon, {S} quietly passed",
    "When {A} and {B} sat down for dinner, {S} poured",
    "After the long shift, {A} and {B} were exhausted. On the way out, {S} thanked",
    "{A} and {B} had a big argument last week. Today, {S} finally forgave",
    "At the reunion, {A} and {B} saw each other across the room. Smiling, {S} hugged",
    "{A} and {B} were walking through the park. Halfway around the lake, {S} called out to",
    "{A} and {B} baked a cake together. When it was ready, {S} shared the first slice with",
    "{A} and {B} were shopping downtown. In the last store, {S} bought a scarf for",
    "During the play, {A} and {B} were sitting in the front row. At the end, {S} smiled at",
    "{A} and {B} were working on the same project. In the end, {S} gave most of the credit to",
    "{A} and {B} were at the cafe. A coffee was brought to {S} by",
    "{A} and {B} were cleaning the house. The broom was handed to {S} by",
    "When {A} and {B} came back from the trip, a postcard was sent to {S} by",
    "{A} and {B} planned the surprise party together. The cake was carried in for {S} by",
    "{A} and {B} were at the station. It was {S} who bought a ticket for",
    "{A} and {B} found a lost puppy. It was {S} who handed it to",
    "{A} and {B} were sitting by the fire. After a while, {S} leaned over and whispered to",
    "{A} and {B} were the only two at the bus stop. Eventually {S} turned and spoke to",
    "{A} and {B} were in the same class. Before the exam, {S} lent a pen to",
    "{A} and {B} were cooking breakfast. {S} cracked the eggs and passed the bowl over to",
    "{A} and {B} were at the beach all day. As the sun set, {S} waved goodbye to",
    "{A} and {B} were reading in the garden. Suddenly, {S} looked up at",
    "{A} and {B} were at the market. After some haggling, {S} sold the old lamp to",
]

MALE_NAMES = ["John", "James", "David", "Michael", "Robert", "William", "Thomas", "Daniel", "Paul",
              "Mark", "Peter", "Kevin", "Brian", "George", "Edward", "Steven", "Andrew", "Ryan",
              "Jack", "Tom", "Dan", "Sam", "Ben", "Luke", "Adam", "Eric", "Scott", "Frank",
              "Henry", "Carl", "Nick", "Alex", "Jason", "Justin", "Kyle", "Sean", "Adam", "Joe",
              "Bill", "Matt"]
FEMALE_NAMES = ["Mary", "Sarah", "Emma", "Anna", "Lisa", "Kate", "Laura", "Rose", "Alice", "Jane",
                "Amy", "Julia", "Grace", "Linda", "Susan", "Karen", "Nancy", "Helen", "Rachel",
                "Emily", "Hannah", "Maria", "Diana", "Ruth", "Joan", "Megan", "Julie", "Anne",
                "Victoria", "Natalie", "Sophie", "Lucy", "Ellen", "Eva", "Jessica", "Ashley",
                "Kim", "Jill", "Beth", "Ann"]
NOUNS_M = ["man", "boy", "king", "father", "husband", "brother", "uncle", "son", "prince", "actor"]
NOUNS_F = ["woman", "girl", "queen", "mother", "wife", "sister", "aunt", "daughter", "princess", "actress"]
NOUNS_N = ["doctor", "teacher", "nurse", "farmer", "student", "baker", "chef", "pilot", "lawyer",
           "manager", "officer", "guard", "clerk", "artist", "writer", "singer", "coach", "captain",
           "soldier", "judge", "dog", "cat", "horse", "bird", "fox", "wolf", "rabbit", "bear"]

_CAP = re.compile(r'(^|[.!?]\s+)(the|he|she) ')   # sentence starts only (not after a closing quote)


def _single(tok, w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def _forms(tok, *words):
    """Token ids of each word mid-sentence (" w") and at the very start of a text ("w").
    A text-initial word is a different token in BPE vocabularies such as GPT-2's."""
    out = set()
    for w in words:
        for v in (" " + w, w):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
    return out


def _row_offset(row, tok):
    return 1 if (tok.bos_token_id is not None and tok.bos_token_id != tok.pad_token_id
                 and row[0] == tok.bos_token_id) else 0


def build_natural(tok, n, rng, frames, males, females, nouns_m, nouns_f, nouns_n,
                  p_noun=0.4, p_pronoun=0.3, max_tries=20):
    """Generate n checked prompts. Entity lists are words; non-single-token ones are dropped."""
    keep = lambda ws: [(w, _single(tok, w)) for w in ws if _single(tok, w) is not None]
    M, F = keep(males), keep(females)
    NM, NF, NN = keep(nouns_m), keep(nouns_f), keep(nouns_n)
    pron_ids = _forms(tok, "he", "He", "she", "She")
    texts, y, fids, sids, meta, pos_l = [], [], [], [], [], []
    while len(texts) < n:
        i = len(texts)
        for _ in range(max_tries):
            noun = rng.random() < p_noun
            pron = rng.random() < p_pronoun
            fi = int(rng.integers(len(frames)))
            if pron:   # one male, one female entity of the same type
                m_pool, f_pool = (NM, NF) if noun else (M, F)
                a, b = m_pool[int(rng.integers(len(m_pool)))], f_pool[int(rng.integers(len(f_pool)))]
                io, s = (a, b) if rng.random() < 0.5 else (b, a)
                s_is_male = s in m_pool
            else:
                pool = (NM + NF + NN) if noun else (M + F)
                j, k = rng.choice(len(pool), 2, replace=False)
                io, s = pool[int(j)], pool[int(k)]
            label = i % 2                                   # 0: IO first
            A, B = (io, s) if label == 0 else (s, io)
            surf = (lambda e: "the " + e[0]) if noun else (lambda e: e[0])
            s2 = ("he" if s_is_male else "she") if pron else surf(s)
            text = frames[fi].format(A=surf(A), B=surf(B), S=s2) + (" the" if noun else "")
            text = _CAP.sub(lambda m_: m_.group(1) + m_.group(2).capitalize() + " ", text)
            # structural checks on tokens (mentions match either token form)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            fa, fb = _forms(tok, A[0]), _forms(tok, B[0])
            fio, fs = (fa, fb) if label == 0 else (fb, fa)
            occ = lambda fset: [t for t, x in enumerate(ids) if x in fset]
            o_io, o_s, o_pr = occ(fio), occ(fs), occ(pron_ids)
            if len(o_io) != 1 or len(o_s) != (1 if pron else 2) or len(o_pr) != (1 if pron else 0):
                continue
            pa, pb = occ(fa)[0], occ(fb)[0]
            ps2 = o_pr[0] if pron else o_s[1]
            if not (pa < pb < ps2 < len(ids) - 1):
                continue
            texts.append(text); y.append(label); fids.append(A[1]); sids.append(B[1])
            pos_l.append((pa, pb, ps2))
            meta.append({"frame": fi, "type": "noun" if noun else "name",
                         "s2": "pronoun" if pron else "repeat", "io": io[0], "s": s[0]})
            break
        else:
            raise RuntimeError("could not build a valid prompt; check frames and entity lists")
    enc = tok(texts, return_tensors="pt", padding=True)
    ids, att = enc["input_ids"], enc["attention_mask"]
    pos = np.zeros((n, 4), np.int64)
    for i, (pa, pb, ps2) in enumerate(pos_l):
        off = _row_offset(ids[i].tolist(), tok)
        io_p, s1_p = (pa, pb) if y[i] == 0 else (pb, pa)
        pos[i] = [io_p + off, s1_p + off, ps2 + off, int(att[i].sum()) - 1]
    pool = np.array(sorted({e[1] for e in M + F + NM + NF + NN}))
    return ({"ids": ids, "att": att, "pos": pos, "last": att.sum(1).numpy() - 1,
             "f": np.array(fids), "s": np.array(sids), "y": np.array(y, np.int64)},
            texts, meta, pool)


def build_abstract(tok, n, rng, frames, males, females, nouns_m, nouns_f,
                   intro="entity", p_pronoun_s2=0.5, max_tries=30):
    """Name-free answers. IO and S always differ in gender and the answer is the
    object pronoun of IO (" him" / " her"), so the output never encodes identity.
      intro="entity":  A and B are names or 'the <noun>' (mixed freely within a prompt);
                       S returns as a repeat or as he/she
      intro="pronoun": A and B are themselves "he"/"she"; S returns as he/she
    The margin targets f/s are the object pronouns of A/B; positions are the
    mention tokens (IO, S1, S2) and the final token."""
    one = lambda w: _single(tok, w)
    him, her = one("him"), one("her")
    pron_ids = _forms(tok, "he", "He", "she", "She")
    keep = lambda ws: [(w, one(w)) for w in ws if one(w) is not None]
    M = [(w, i, "name") for w, i in keep(males)] + [(w, i, "noun") for w, i in keep(nouns_m)]
    F = [(w, i, "name") for w, i in keep(females)] + [(w, i, "noun") for w, i in keep(nouns_f)]
    texts, y, fobj, sobj, pos_l, meta = [], [], [], [], [], []
    while len(texts) < n:
        i = len(texts)
        for _ in range(max_tries):
            fi = int(rng.integers(len(frames)))
            m, f = M[int(rng.integers(len(M)))], F[int(rng.integers(len(F)))]
            io_male = rng.random() < 0.5
            io, s = (m, f) if io_male else (f, m)
            label = i % 2
            A, B = (io, s) if label == 0 else (s, io)
            a_male = (A is m)
            if intro == "pronoun":
                surfA, surfB = ("he" if a_male else "she"), ("she" if a_male else "he")
                s2 = "he" if (s is m) else "she"; s2_form = "pronoun"
            else:
                sf = lambda e: e[0] if e[2] == "name" else "the " + e[0]
                surfA, surfB = sf(A), sf(B)
                pron = rng.random() < p_pronoun_s2
                s2 = ("he" if (s is m) else "she") if pron else sf(s)
                s2_form = "pronoun" if pron else "repeat"
            text = frames[fi].format(A=surfA, B=surfB, S=s2)
            text = _CAP.sub(lambda m_: m_.group(1) + m_.group(2).capitalize() + " ", text)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            # locate mentions
            if intro == "pronoun":
                prs = [t for t, x in enumerate(ids) if x in pron_ids]
                if len(prs) != 3:
                    continue
                pa, pb, ps2 = prs
            else:
                occ = lambda fset: [t for t, x in enumerate(ids) if x in fset]
                o_io, o_s, o_pr = occ(_forms(tok, io[0])), occ(_forms(tok, s[0])), occ(pron_ids)
                if s2_form == "pronoun":
                    if len(o_pr) != 1 or len(o_io) != 1 or len(o_s) != 1:
                        continue
                    ps2 = o_pr[0]
                else:
                    if len(o_pr) != 0 or len(o_io) != 1 or len(o_s) != 2:
                        continue
                    ps2 = o_s[1]
                pa, pb = occ(_forms(tok, A[0]))[0], occ(_forms(tok, B[0]))[0]
            if not (pa < pb < ps2 < len(ids) - 1) or him in ids or her in ids:
                continue
            texts.append(text); y.append(label)
            fobj.append(him if a_male else her); sobj.append(her if a_male else him)
            pos_l.append((pa, pb, ps2))
            meta.append({"frame": fi, "intro": intro, "s2": s2_form,
                         "types": A[2] + "+" + B[2] if intro == "entity" else "pronoun",
                         "io": io[0], "s": s[0], "answer": "him" if io_male else "her"})
            break
        else:
            raise RuntimeError("could not build a valid prompt")
    enc = tok(texts, return_tensors="pt", padding=True)
    ids_t, att = enc["input_ids"], enc["attention_mask"]
    pos = np.zeros((n, 4), np.int64)
    for i, (pa, pb, ps2) in enumerate(pos_l):
        off = _row_offset(ids_t[i].tolist(), tok)
        last = int(att[i].sum()) - 1
        io_p, s1_p = (pa, pb) if y[i] == 0 else (pb, pa)
        pos[i] = [io_p + off, s1_p + off, ps2 + off, last]
    return ({"ids": ids_t, "att": att, "pos": pos, "last": att.sum(1).numpy() - 1,
             "f": np.array(fobj), "s": np.array(sobj), "y": np.array(y, np.int64)},
            texts, meta, np.array([him, her]))


def entity_split(words, part):
    """Deterministic halves of an entity list: part 0 = seen, 1 = new."""
    return words[part::2]


def frame_split(part):
    """Frames: every 4th frame is held out (part 1); the rest are for extraction (part 0)."""
    return [f for i, f in enumerate(FRAMES) if (i % 4 == 3) == (part == 1)]


def main():
    from cores.utils import seed_all, get_device
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--n", type=int, default=1280)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--attn-impl", default=None, help="e.g. eager (recommended for Gemma-2)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    device = get_device(); seed_all(args.seed)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    import torch as _t
    kw = {"attn_implementation": args.attn_impl} if args.attn_impl else {}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype={"float32": _t.float32, "bfloat16": _t.bfloat16}[args.dtype],
                                                 **kw).to(device).eval()
    from sklearn.metrics import roc_auc_score

    def behave(d):
        io_s, m = [], []
        for st in range(0, len(d["y"]), 32):
            b = slice(st, st + 32)
            with torch.no_grad():
                L = model(input_ids=d["ids"][b].to(device), attention_mask=d["att"][b].to(device)).logits
            last = torch.tensor(d["last"][b], device=device)
            L = L[torch.arange(L.shape[0], device=device), last].float().cpu().numpy()
            for j in range(L.shape[0]):
                i = st + j; f, s = d["f"][i], d["s"][i]
                io, S = (f, s) if d["y"][i] == 0 else (s, f)
                io_s.append(L[j, io] - L[j, S]); m.append(L[j, f] - L[j, s])
        return np.array(io_s), np.array(m)

    def report(name, d, texts, meta, keys):
        io_s, m = behave(d); acc = io_s > 0
        r = {"auc": float(roc_auc_score(d["y"], -m)), "acc": float(acc.mean()), "logit_diff": float(io_s.mean())}
        print(f"\n{args.model_name} | {name}: {len(texts)} prompts | AUC {r['auc']:.3f} | IO>S {r['acc']:.3f} | "
              f"logit diff {r['logit_diff']:.2f}")
        for key in keys:
            vals = sorted({x[key] for x in meta}, key=str)
            r["by_" + key] = {}
            parts = []
            for v in vals:
                sel = np.array([x[key] == v for x in meta])
                r["by_" + key][str(v)] = (float(acc[sel].mean()), float(io_s[sel].mean()), int(sel.sum()))
                if key != "frame":
                    parts.append(f"{v}: {acc[sel].mean():.3f} ({io_s[sel].mean():+.2f}, n={sel.sum()})")
            if key == "frame":
                weak = [(v, a) for v, (a, l, c) in r["by_frame"].items() if a < 0.7]
                print(f"  frames below 0.7: {len(weak)} of {len(vals)} " + " ".join(f"[{v}]{a:.2f}" for v, a in weak))
            else:
                print(f"  by {key}: " + "  ".join(parts))
        for t, x in list(zip(texts, meta))[:3]:
            print(f"    e.g. {t!r} -> {x.get('answer', x['io'])}")
        return r

    rng = np.random.default_rng(args.seed)
    res = {}
    d, texts, meta, _ = build_natural(tok, args.n, rng, FRAMES, MALE_NAMES, FEMALE_NAMES, NOUNS_M, NOUNS_F, NOUNS_N)
    res["natural"] = report("natural (entity answers)", d, texts, meta, ["type", "s2", "frame"])
    d, texts, meta, _ = build_abstract(tok, args.n, np.random.default_rng(args.seed + 1), FRAMES,
                                       MALE_NAMES, FEMALE_NAMES, NOUNS_M, NOUNS_F, intro="entity")
    res["pronoun_answer"] = report("entities, pronoun answer (him/her)", d, texts, meta, ["types", "s2", "frame"])
    d, texts, meta, _ = build_abstract(tok, args.n, np.random.default_rng(args.seed + 2), FRAMES,
                                       MALE_NAMES, FEMALE_NAMES, NOUNS_M, NOUNS_F, intro="pronoun")
    res["all_pronoun"] = report("all pronouns", d, texts, meta, ["frame"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": args.model_name, **res}, indent=2, default=float))


if __name__ == "__main__":
    main()
