"""Wikipedia-2018 dense retriever (the MEM1 / Search-R1 setup) on Modal.

e5-base-v2 queries against a FAISS flat index over 21M passages, top-k passages back.

    uv run modal run --detach infra/retriever.py::download   # once: ~70 GB onto the volume
    uv run modal deploy infra/retriever.py                    # HTTP service, scales to zero
    uv run modal run infra/retriever.py::check                # M1 exit check: latency + sanity

POST /retrieve {"queries": ["..."], "topk": 3} with `Authorization: Bearer $TEACHCLM_API_KEY`
-> {"result": [[{"id", "title", "text", "score"}, ...], ...]}
"""

import os

import modal

MINUTES = 60
DATA = "/data"
INDEX = f"{DATA}/e5_Flat.index"
CORPUS = f"{DATA}/wiki-18.jsonl"
OFFSETS = f"{DATA}/wiki-18.offsets.npy"
ENCODER = "intfloat/e5-base-v2"
ENCODER_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"

app = modal.App("teachclm-retriever")
wiki = modal.Volume.from_name("teachclm-wiki18", create_if_missing=True)
hf_cache = modal.Volume.from_name("teachclm-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "faiss-cpu==1.16.0", "numpy<3", "huggingface_hub>=0.34", "hf_transfer",
        "transformers>=4.57,<6", "fastapi[standard]>=0.115",
        "torch", extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(image=image, volumes={DATA: wiki}, timeout=180 * MINUTES, cpu=8, memory=16384)
def download():
    """Fetch the index parts and corpus, join/unpack them, and index passage offsets."""
    import gzip
    import shutil
    import tarfile

    import json

    import faiss
    import numpy as np
    from huggingface_hub import hf_hub_download

    if not os.path.exists(INDEX):
        parts = [
            hf_hub_download("PeterJinGo/wiki-18-e5-index", p, repo_type="dataset", local_dir=f"{DATA}/parts")
            for p in ("part_aa", "part_ab")
        ]
        with open(INDEX + ".tmp", "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=64 << 20)
        os.rename(INDEX + ".tmp", INDEX)
        shutil.rmtree(f"{DATA}/parts")
        wiki.commit()
    print("index bytes:", os.path.getsize(INDEX))

    if not os.path.exists(CORPUS):
        gz = hf_hub_download("PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz", repo_type="dataset", local_dir=f"{DATA}/gz")
        with gzip.open(gz, "rb") as f, open(CORPUS + ".tmp", "wb") as out:
            shutil.copyfileobj(f, out, length=64 << 20)
        os.rename(CORPUS + ".tmp", CORPUS)
        shutil.rmtree(f"{DATA}/gz")
        wiki.commit()
    if tarfile.is_tarfile(CORPUS):
        # Despite its name, the upstream file is a gzipped tar holding one jsonl member.
        with tarfile.open(CORPUS) as tar:
            (member,) = [m for m in tar.getmembers() if m.isfile()]
            with tar.extractfile(member) as f, open(CORPUS + ".tmp", "wb") as out:
                shutil.copyfileobj(f, out, length=64 << 20)
        os.rename(CORPUS + ".tmp", CORPUS)
        if os.path.exists(OFFSETS):
            os.remove(OFFSETS)
        wiki.commit()
    print("corpus bytes:", os.path.getsize(CORPUS))

    if not os.path.exists(OFFSETS):
        offsets, pos = [], 0
        with open(CORPUS, "rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        np.save(OFFSETS, np.asarray(offsets, dtype=np.int64))
        wiki.commit()
    offsets = np.load(OFFSETS, mmap_mode="r")
    print("passages:", len(offsets))

    # Passage i must sit on line i, because FAISS returns row numbers.
    with open(CORPUS, "rb") as f:
        for i in (0, len(offsets) // 2, len(offsets) - 1):
            f.seek(int(offsets[i]))
            assert json.loads(f.readline())["id"] == str(i), f"line {i} does not hold passage {i}"
    index_rows = faiss.read_index(INDEX, faiss.IO_FLAG_MMAP).ntotal
    assert index_rows == len(offsets), f"index has {index_rows} vectors, corpus has {len(offsets)} passages"
    print("OK: corpus lines match index rows")


@app.cls(
    image=image,
    volumes={DATA: wiki, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("teachclm-api-key")],
    cpu=16,
    memory=81920,                    # the flat index is 64.6 GB and is held in RAM
    max_containers=1,
    scaledown_window=10 * MINUTES,   # reloading the index is slow, so stay warm a little longer
    timeout=10 * MINUTES,
    startup_timeout=40 * MINUTES,
)
@modal.concurrent(max_inputs=32)
class Retriever:
    @modal.enter()
    def load(self):
        import time

        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        t0 = time.time()
        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
        self.enc = AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION).eval()
        self.offsets = np.load(OFFSETS, mmap_mode="r")
        self.corpus = open(CORPUS, "rb")
        self.index = faiss.read_index(INDEX)
        assert self.index.ntotal == len(self.offsets), (self.index.ntotal, len(self.offsets))
        print(f"loaded {self.index.ntotal} vectors in {time.time() - t0:.0f}s")

    def _embed(self, queries):
        import torch

        batch = self.tok([f"query: {q}" for q in queries], padding=True, truncation=True,
                         max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(emb, dim=-1).numpy().astype("float32")

    def _passage(self, idx):
        import json

        self.corpus.seek(int(self.offsets[idx]))
        doc = json.loads(self.corpus.readline())
        title, _, text = doc["contents"].partition("\n")
        return {"id": doc["id"], "title": title.strip('"'), "text": text}

    def search(self, queries, topk):
        scores, ids = self.index.search(self._embed(queries), topk)
        return [
            [{**self._passage(i), "score": float(s)} for i, s in zip(row_ids, row_scores)]
            for row_ids, row_scores in zip(ids, scores)
        ]

    @modal.asgi_app()
    def web(self):
        import threading

        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel

        api = FastAPI()
        lock = threading.Lock()      # one shared file handle; FAISS search is already multi-threaded

        class Query(BaseModel):
            queries: list[str]
            topk: int = 3

        @api.post("/retrieve")
        def retrieve(q: Query, authorization: str = Header(default="")):
            if authorization != f"Bearer {os.environ['TEACHCLM_API_KEY']}":
                raise HTTPException(401, "bad or missing API key")
            with lock:
                return {"result": self.search(q.queries, q.topk)}

        return api


@app.local_entrypoint()
def check():
    """M1 exit check against the deployed service: sane results and warm latency under 1 s."""
    import json
    import time
    import urllib.request

    from dotenv import load_dotenv

    load_dotenv()
    url = modal.Cls.from_name("teachclm-retriever", "Retriever")().web.get_web_url()
    headers = {"Authorization": f"Bearer {os.environ['TEACHCLM_API_KEY']}", "Content-Type": "application/json"}

    def post(queries):
        req = urllib.request.Request(f"{url}/retrieve", json.dumps({"queries": queries, "topk": 3}).encode(), headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=45 * MINUTES) as resp:
            return json.load(resp)["result"], time.time() - t0

    # Loading the 65 GB index takes minutes; Modal answers 303 while the container is still starting.
    import urllib.error
    start = time.time()
    while True:
        try:
            res, dt = post(["who wrote the novel Moby-Dick"])
            break
        except urllib.error.HTTPError as e:
            if e.code not in (303, 502, 503, 504) or time.time() - start > 45 * MINUTES:
                raise
            print(f"still starting ({e.code}) after {time.time() - start:.0f}s")
    print(f"first answer after {time.time() - start:.0f}s (includes cold start)")
    for hit in res[0]:
        print(f"  {hit['score']:.3f} {hit['title']}: {hit['text'][:90]}")
    assert any("Melville" in h["text"] or "Moby" in h["title"] for h in res[0]), "top hits look wrong"

    times = [post([q])[1] for q in ("capital of Australia", "when did the Berlin Wall fall",
                                    "who discovered penicillin", "tallest mountain in Africa", "author of Dune")]
    print("warm latencies (s):", [round(t, 2) for t in times])
    assert max(times) < 1.0, "warm latency above 1 s"
    print("PASS: retriever")
