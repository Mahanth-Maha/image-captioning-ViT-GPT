from collections import Counter
from heapq import heapify, heappop, heappush
from tqdm import tqdm
import os
import time
import re

def enc(x, chars):
    return [chars.index(c) for c in x]

def dec(x, chars):
    return "".join(chars[i] for i in x)
        
class CharTokenizer:
    def __init__(self, chars_file, special_tokens=None):
        self.special_tokens = special_tokens if special_tokens else {}
        self.vocab = None
        self.chars_file = chars_file 
        if os.path.exists(self.chars_file):
            self.chars = self.load()
        self.vocab_size = self._vocab_size()

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
    
    def encode(self,text):
        return [self.chars.index(c) for c in text]

    def _vocab(self):
        vocab = {idx: charr for idx,charr in enumerate(self.chars)}
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def _vocab_size(self):
        if self.vocab is None:
            return None
        return len(self.vocab)
    
    def save(self):
        open(self.chars_file, "w").write(self.chars)

    def train(self, data_file):
        data = open(data_file, "r").read()
        self.chars = "".join(sorted(set(data)))
        self.save()
        return self.chars
    
    def load(self):
        self.chars = open(self.chars_file, "r").read()
        self.vocab = self._vocab()
        return self.chars
        
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word_idx = None

class ByteTrie:
    def __init__(self, patterns):
        self.root = TrieNode()
        for i, pat in enumerate(patterns):
            node = self.root
            for b in pat:
                if b not in node.children:
                    node.children[b] = TrieNode()
                node = node.children[b]
            node.word_idx = i

    def longest_match(self, ids, start):
        node = self.root
        j = start
        last_len = 0
        last_idx = None
        n = len(ids)
        while j < n and ids[j] in node.children:
            node = node.children[ids[j]]
            j += 1
            if node.word_idx is not None:
                last_len = j - start
                last_idx = node.word_idx
        return last_len, last_idx

class myTokenizer_faster:
    def __init__(self, token_model_file, special_tokens=None, base_tokens = 256, protected_words=None, preprocess_mode = 'removeall'):
        self.merges = {}
        self.token_model_file = token_model_file 
        self.base_tokens = base_tokens 
        self.preprocess_mode = preprocess_mode 
        self.special_tokens = special_tokens if special_tokens else {}
        self.special_tokens_backup = self.special_tokens.copy()
        self.protected_words = protected_words if protected_words else []
        self.protected_words_dict = {}
        for idx , w in enumerate(self.protected_words):
            self.protected_words_dict[w] = base_tokens + idx
        self.vocab = self._vocab()
        
    def normalize_spaces(self, text):
        text = ' ' + text
        if self.preprocess_mode == 'removeall':
            text = re.sub(r'([\n\r\t\v\f]+)', r' \1 ', text)
            text = re.sub(r'([.,!?;:(){}\[\]\"\'\-])', r' \1', text) 
        text = re.sub(r'(?<!\s)(?=\b\w)', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _is_valid_merge(self, left_bytes, right_bytes):
        l = left_bytes.decode('utf-8', errors='ignore')
        r = right_bytes.decode('utf-8', errors='ignore')
        
        if l.strip() == '' and r.strip() == '':
            return True
         
        if r.endswith(' '):
            return False
        if l.endswith(' '):
            return False
        if not l.startswith(' ') and r.startswith(' '):
            return False
    
        return True

    def _get_bigram_byte_frequency(self, ids):
        return Counter(zip(ids[:-1], ids[1:]))
    
    def _get_bigram_word_frequency(self, ids):
        ids_list = list(ids) if isinstance(ids, (bytes,bytearray)) else list(ids)
        pats = [p.encode("utf-8") for p in self.protected_words]
        new_ids = []
        i = 0
        n = len(ids_list)
        while i < n:
            matched = False
            for idx, pat in enumerate(pats):
                L = len(pat)
                if i + L <= n and ids_list[i:i+L] == list(pat):
                    new_ids.append(idx + 256)
                    i += L
                    matched = True
                    break
            if not matched:
                new_ids.append(ids_list[i])
                i += 1
        return Counter(zip(new_ids[:-1], new_ids[1:]))
            

    def _get_bigram_word_frequency_fast(self, ids):
        if isinstance(ids, (bytes, bytearray)):
            ids_list = list(ids)
        else:
            ids_list = list(ids)

        patterns_bytes = [w.encode("utf-8") for w in self.protected_words]

        trie = ByteTrie(patterns_bytes)

        n = len(ids_list)
        i = 0
        new_ids = []
        while i < n:
            length, widx = trie.longest_match(ids_list, i)
            if length > 0:
                new_ids.append(widx + 256)
                i += length
            else:
                new_ids.append(ids_list[i])
                i += 1

        return Counter(zip(new_ids[:-1], new_ids[1:]))
            
    
    def train(self, text, vocab_size):
        spl_tkns = len(self.special_tokens)
        num_merges = vocab_size - self.base_tokens - spl_tkns

        if vocab_size <= self.base_tokens:
            vocab = {idx: bytes([idx]) for idx in range(vocab_size)}
            print(f'[BPE TRAIN LOG] vocab size ({vocab_size}) is <= base tokens ({self.base_tokens}), ignoring special tokens and keeping vocab to given vocab size')
            print(f'[BPE TRAIN LOG] Dropping Special tokens and Merges')
            self.special_tokens = {}
            self.merges = {}
            print(f'[BPE TRAIN LOG] No merges, No Special Tokens')
            # return
        else :
            vocab = {idx: bytes([idx]) for idx in range(self.base_tokens)}

            print(f"🔸 Starting BPE training...")

            next_id = self.base_tokens
            # Add protected words
            for word in self.protected_words:
                pword = ' ' + word
                if next_id >= vocab_size:
                    break
                if pword.encode('utf-8') not in vocab.values():
                    vocab[next_id] = pword.encode('utf-8')
                    next_id += 1
            num_merges -= len(self.protected_words)

            # Preprocess text and build initial frequency table
            text = self.normalize_spaces(text)
            ids = list(text.encode("utf-8"))

            counts = self._get_bigram_word_frequency_fast(ids)
            heap = [(-freq, source) for source, freq in counts.items()]
            heapify(heap)

            accepted_merges = 0
            skipped_invalid = 0
            skipped_stale = 0
            merges_done = 0

            pbar = tqdm(total=num_merges, desc="BPE Training", unit="merge")
            while accepted_merges < num_merges:
                # Find next valid candidate from heap
                while heap:
                    freq, source = heappop(heap)
                    if counts[source] == -freq and counts[source] > 0:
                        break
                else:
                    print('⚠️ Heap is exhausted! Cannot complete all merges.')
                    break

                left_bytes  = vocab.get(source[0], None)
                right_bytes = vocab.get(source[1], None)
                if left_bytes is None or right_bytes is None:
                    counts[source] = 0
                    skipped_stale += 1
                    continue

                if not self._is_valid_merge(left_bytes, right_bytes):
                    counts[source] = 0
                    skipped_invalid += 1
                    continue

                # Do the actual merge
                dest = next_id + merges_done
                self.merges[source] = dest
                vocab[dest] = left_bytes + right_bytes
                merges_done += 1
                accepted_merges += 1
                pbar.update(1)

                # Update ids & counts
                new_ids = []
                j = 0
                while j < len(ids):
                    if j < len(ids) - 1 and ids[j] == source[0] and ids[j + 1] == source[1]:
                        new_ids.append(dest)

                        if len(new_ids) >= 2:
                            left = (new_ids[-2], dest)
                            counts[left] += 1
                            heappush(heap, (-counts[left], left))

                        if j + 2 < len(ids):
                            right = (dest, ids[j + 2])
                            counts[right] += 1
                            heappush(heap, (-counts[right], right))

                        j += 2
                    else:
                        new_ids.append(ids[j])
                        j += 1
                ids = new_ids
                counts[source] = 0

            pbar.close()
            print(f"🔸 BPE training Finished: {accepted_merges} merges accepted, {skipped_invalid} invalid, {skipped_stale} stale")
            print(f"✅ Final vocab size: {len(vocab)} (target: {vocab_size})")

        self.vocab = vocab
        
        self.save(vocab_size)
        self._analyze_vocab()

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
    
    def encode(self, text, progress=False, legacy=False):
        if not legacy:
            return self.encode_fast(text, progress=progress)
        
        ids = list(text.encode("utf-8"))
        initial_length = len(ids)
        
        if progress:
            pbar = tqdm(desc="Encoding", unit="merges", leave=True)
            pbar.set_postfix({
                'tokens': len(ids), 
                'original': initial_length,
                'ratio': '1.000'
            })
        
        merge_count = 0
        
        while True:
            candidates = [(self.merges[bg], bg) for bg in zip(ids[:-1], ids[1:]) if bg in self.merges]
            if not candidates:
                break
            
            _, pair = min(candidates)
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                    new_ids.append(self.merges[pair])
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
            merge_count += 1
            
            if progress:
                pbar.update(1)
                pbar.set_postfix({
                    'tokens': len(ids),
                    'original': initial_length, 
                    'ratio': f"{len(ids)/initial_length:.3f}",
                    'merges': merge_count
                })
        
        if progress:
            pbar.close()
            print(f"✅ Encoding complete: {initial_length} -> {len(ids)} tokens ({merge_count} merges)")
        
        return ids

    def encode_fast(self, text, progress=False):
        ids = list(text.encode("utf-8"))
        if len(ids) <= 1:
            return ids

        ids = self._replace_protected_words(ids)

        initial_length = len(ids)
        if progress:
            pbar = tqdm(total=len(ids), desc="Encoding", unit="merge", dynamic_ncols=True)
            start_time = time.time()

        i = 0
        out = []
        merges_applied = 0
        while i < len(ids):
            j = i
            token = ids[j]

            while j + 1 < len(ids) and (token, ids[j + 1]) in self.merges:
                token = self.merges[(token, ids[j + 1])]
                j += 1
                merges_applied += 1

                if progress and merges_applied % 100 == 0:
                    compression = (len(out) + (len(ids) - j)) / initial_length
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (merges_applied + 1)
                    eta = avg_time * (len(ids) - j)
                    pbar.set_postfix({
                        "tokens": f"{len(out):,}",
                        "ratio": f"{compression:.3f}x",
                        "ETA": f"{eta:.1f}s"
                    })
                    pbar.update(100)

            out.append(token)
            i = j + 1

        if progress:
            pbar.close()
            final_compression = len(out) / initial_length
            print(f"✅ Encoded: {initial_length:,} → {len(out):,} tokens "
                f"({final_compression:.3f}x compression)")

        return out

    def _replace_protected_words(self, ids):
        patterns_bytes = []
        for i, word in enumerate(self.protected_words):
            pattern = b' ' + word.encode('utf-8')
            patterns_bytes.append((list(pattern), self.base_tokens + i))
        
        trie = ByteTrie([p[0] for p in patterns_bytes])
        
        new_ids = []
        i = 0
        n = len(ids)
        
        while i < n:
            length, widx = trie.longest_match(ids, i)
            if length > 0:
                token_id = self.base_tokens + widx
                new_ids.append(token_id)
                i += length
            else:
                new_ids.append(ids[i])
                i += 1
        
        return new_ids


    def _vocab(self, vocab_size=None):
        BASE = self.base_tokens
        if vocab_size is not None and vocab_size <= BASE:
            return {idx: bytes([idx]) for idx in range(vocab_size)}
        vocab = {idx: bytes([idx]) for idx in range(BASE)}

        P = len(self.protected_words)
        for i, w in enumerate(self.protected_words):
            tid = BASE + i
            vocab[tid] = b" " + w.encode("utf-8")
            
        for (p0, p1), tid in sorted(self.merges.items(), key=lambda kv: kv[1]):
            if p0 not in vocab:
                if p0 < BASE:
                    vocab[p0] = bytes([p0])
                else:
                    raise KeyError(f"Merge component {p0} missing when materializing {tid}")
            if p1 not in vocab:
                if p1 < BASE:
                    vocab[p1] = bytes([p1])
                else:
                    raise KeyError(f"Merge component {p1} missing when materializing {tid}")
            vocab[tid] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        
        for k,v in self.protected_words_dict.items():
            vocab[v] = (' ' + k).encode("utf-8")

        return vocab

    def save(self, vocab_size):
        model_file = self.token_model_file
        vocab_file = self.token_model_file.replace('.model', '_vocab.txt')

        merges_sorted = sorted(self.merges.items(), key=lambda kv: kv[1])

        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("v1\n")
            f.write(f"base {self.base_tokens}\n")
            f.write(f"preprocess {self.preprocess_mode}\n")
            f.write(f"protected {len(self.protected_words)}\n")
            for w in self.protected_words:
                f.write(f"pw {w}\n")
            f.write(f"merges {len(merges_sorted)}\n")
            for (a, b), _tid in merges_sorted:
                f.write(f"m {a} {b}\n")
            f.write(f"special {len(self.special_tokens)}\n")
            for tok, idx in self.special_tokens.items():
                f.write(f"s {tok} {idx}\n")

        vocab = self._vocab(vocab_size)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("# Tokenizer Vocabulary\n")
            f.write(f"# Total vocabulary size: {len(vocab)}\n")
            f.write(f"# Number of merges: {len(self.merges)}\n")
            f.write(f"# Number of special tokens: {len(self.special_tokens)}\n")
            f.write(f"# Base tokens: {self.base_tokens}\n")
            f.write("#" + "="*60 + "\n\n")
            for token_id in sorted(vocab.keys()):
                token_bytes = vocab[token_id]
                try:
                    token_str = token_bytes.decode('utf-8', errors='replace') if isinstance(token_bytes, bytes) else str(token_bytes)
                    display_str = repr(token_str) if any(ord(c) < 32 or ord(c) > 126 for c in token_str) else token_str
                    f.write(f"{token_id:6d}: {display_str}\n")
                except Exception:
                    f.write(f"{token_id:6d}: {repr(token_bytes)}\n")

        # print(f"Model saved to: {model_file}")
        # print(f"Readable vocab saved to: {vocab_file}")

    def load(self):
        model_file = self.token_model_file
        lines = []
        with open(model_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        BASE = self.base_tokens
        preprocess_mode = getattr(self, "preprocess_mode", "removeall")
        protected_words = []
        merge_pairs = []
        special_tokens = {}

        if not lines:
            is_new = False
        else:
            prefixes = ("base", "preprocess", "protected", "pw ", "merges", "m ", "special", "s ")
            is_new = lines[0].startswith("v1") or any(l.startswith(prefix) for l in lines for prefix in prefixes)

        if is_new:
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith("v1"):
                    i += 1; continue
                if line.startswith("base"):
                    try: BASE = int(line.split()[1])
                    except: pass
                    i += 1; continue
                if line.startswith("preprocess"):
                    try: preprocess_mode = line.split(maxsplit=1)[1]
                    except: pass
                    i += 1; continue
                if line.startswith("protected"):
                    try: K = int(line.split()[1])
                    except: K = 0
                    i += 1
                    for _ in range(K):
                        if i < len(lines) and lines[i].startswith("pw "):
                            protected_words.append(lines[i][3:])
                            i += 1
                        else:
                            break
                    continue
                if line.startswith("pw "):
                    protected_words.append(line[3:])
                    i += 1; continue
                if line.startswith("merges"):
                    try: M = int(line.split()[1])
                    except: M = None
                    i += 1
                    while i < len(lines):
                        l2 = lines[i]
                        if l2.startswith(("special", "s ", "protected", "pw ", "base", "preprocess", "v1")):
                            break
                        if l2.startswith("m "):
                            parts = l2.split()
                            if len(parts) == 3:
                                a, b = int(parts[1]), int(parts[2])
                                merge_pairs.append((a, b))
                        i += 1
                    continue
                if line.startswith("m "):
                    parts = line.split()
                    if len(parts) == 3:
                        a, b = int(parts[1]), int(parts[2])
                        merge_pairs.append((a, b))
                    i += 1; continue
                if line.startswith("special"):
                    i += 1
                    while i < len(lines):
                        l2 = lines[i]
                        if l2.startswith("s "):
                            parts = l2.split(maxsplit=2)
                            if len(parts) == 3:
                                tok, idx = parts[1], int(parts[2])
                                special_tokens[tok] = idx
                        i += 1
                    break
                if line.startswith("s "):
                    parts = line.split(maxsplit=2)
                    if len(parts) == 3:
                        tok, idx = parts[1], int(parts[2])
                        special_tokens[tok] = idx
                    i += 1; continue
                i += 1

            merges = {}
            tid = BASE + len(protected_words)
            for pair in merge_pairs:
                merges[pair] = tid
                tid += 1

            self.base_tokens = BASE
            self.preprocess_mode = preprocess_mode
            self.protected_words = protected_words
            self.merges = merges
            self.special_tokens = special_tokens

        else:
            merges = {}
            tid = BASE
            for line in lines:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    a, b = int(parts[0]), int(parts[1])
                    merges[(a, b)] = tid
                    tid += 1
                elif len(parts) == 2 and parts[1].isdigit():
                    special_tokens[parts[0]] = int(parts[1])
            self.merges = merges
            self.special_tokens = special_tokens
        self.vocab = self._vocab()
        # print(f"Tokenizer loaded from: {model_file}")
        print(f"Loaded {len(self.merges)} merges, {len(self.special_tokens)} specials, {len(self.protected_words)} protected")
        print(f"Total vocabulary size: {len(self.vocab)}")
        return self.vocab

    def _vocab_size(self):
        if self.vocab is None:
            return None
        return len(self.vocab)
    
    def _analyze_vocab(self):
        vocab = self._vocab()
        
        print("\n" + "="*60)
        print("VOCABULARY ANALYSIS")
        print("="*60)
        print(f"Total vocabulary size: {len(vocab):,}")
        print(f"Base tokens (0-{self.base_tokens - 1}): {self.base_tokens}")
        print(f"Merge tokens: {len(self.merges):,}")
        print(f"Special tokens: {len(self.special_tokens):,}")
        
        base_tokens = sum(1 for tid in vocab.keys() if tid < self.base_tokens)
        merge_tokens = sum(1 for tid in vocab.keys() if self.base_tokens <= tid < self.base_tokens + len(self.merges))
        special_token_ids = sum(1 for tid in vocab.keys() if tid >= self.base_tokens + len(self.merges))
        
        print(f"\nToken distribution:")
        print(f"🔸 Base (bytes): {base_tokens}")
        print(f"🔸 Merges: {merge_tokens}")
        print(f"🔸 Special: {special_token_ids}")
        
        print(f"\nFirst 10 merge tokens:")
        merge_start = self.base_tokens
        for i in range(min(10, len(self.merges))):
            token_id = merge_start + i
            if token_id in vocab:
                token_bytes = vocab[token_id]
                try:
                    token_str = token_bytes.decode('utf-8', errors='replace')
                    display_str = repr(token_str) if any(ord(c) < 32 or ord(c) > 126 for c in token_str) else token_str
                    print(f"🔸 {token_id}: {display_str}")
                except:
                    print(f"🔸 {token_id}: {repr(token_bytes)}")
        
        print(f"\nSpecial tokens:")
        for special, idx in sorted(self.special_tokens.items(), key=lambda x: x[1]):
            print(f"🔸 {special}: {idx}")