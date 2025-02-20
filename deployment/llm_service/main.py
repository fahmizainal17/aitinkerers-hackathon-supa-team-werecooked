import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import torch
from langchain.llms import VLLM

# create the prompt template
prompt = """Anda adalah pakar dalam mengesan ketidakkonsistenan fakta dan halusinasi. Anda akan diberi satu dokumen dan satu soalan. Baca
dokumen dan soalan/kenyataan yang diberikan dengan teliti dan kenal pasti Ketidakkonsistenan Fakta (iaitu mana-mana soalan/kenyataan yang
tidak disokong atau bercanggah dengan maklumat dalam dokumen).

### Anda perlu memilih antara dua pilihan berikut:
- Tidak Konsisten dengan Fakta: Jika mana-mana soalan/kenyataan tidak disokong, terjawab atau bercanggah dengan dokumen, labelkannya sebagai 0.
- Konsisten dengan Fakta: Jika semua soalan/kenyataan disokong/terjawab oleh dokumen, labelkannya sebagai 1.

### Sebagai contoh:
Dokumen: "Gajah adalah mamalia besar yang biasanya ditemui di Afrika dan Asia. Mereka hidup dalam kumpulan yang dikenali sebagai kawanan dan terkenal kerana mempunyai ingatan yang baik."

Soalan/Kenyataan: "Gajah adalah mamalia besar yang biasanya ditemui di Eropah."
Jawapan: {{'consistency': 0}}

Soalan/Kenyataan: "Gajah adalah mamalia besar yang biasanya ditemui di Afrika dan Asia."
Jawapan: {{'consistency': 1}}

### Jawab berdasarkan dokumen dan soalan/kenyataan berikut:
Dokumen: {passage}
Soalan/Kenyataan: {question}

Sediakan penjelasan langkah demi langkah untuk pilihan konsistenan berdasarkan Dokumen dan Soalan/Kenyataan yang diberikan. Selepas itu,
kembalikan pilihan konsistenan dalam format JSON untuk pilihan yang diberikan. Sebagai contoh: {{'consistency': 1}} atau {{'consistency': 0}}"""

app = FastAPI()

llm = VLLM(
    model="wanadzhar913/malaysian-mistral-llmasajudge-v3",
    trust_remote_code=True,  # mandatory for hf models
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    dtype=torch.bfloat16,
    max_new_tokens = 1024,
    temperature = 0.1,
    top_p = 0.95,
    top_k = 50,
)

@app.get("/")
def read_root():
    return {"Hello": "AI Tinkerers!"}


@app.post("/v1/generateText")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()

    question = request_dict.pop("question")
    passage = request_dict.pop("passage")

    output = llm(prompt.format(passage=passage, question=question))
    print("Generated text:", output)
    ret = {"text": output}
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)