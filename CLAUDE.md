# retina-seg

Upgrade de um TCC (CNN Keras de classificação com/sem retinopatia diabética) para um pipeline de
**segmentação de lesões** + API + front. Projeto pessoal/acadêmico — **não é dispositivo médico**.
Todo laudo gerado deve conter aviso de que não substitui avaliação oftalmológica.

## Fluxo do produto

1. Segmentar lesões de retinopatia diabética no dataset **IDRiD** ("A. Segmentation": 81 imagens
   ~4288x2848, 54 treino / 27 teste, máscaras separadas por tipo de lesão).
2. Converter máscaras em lesões individuais (tipo, centro, raio, bbox, área, score, quadrante).
3. API que recebe uma imagem e devolve JSON com grau, lesões, contagens e laudo em texto.
4. Front web que desenha a imagem com as lesões circuladas (overlay SVG, cor por tipo, toggle por
   tipo) e mostra o laudo.

## Ambiente

- WSL2 (Ubuntu), GPU NVIDIA RTX 3060 Ti, 8 GB VRAM — `torch.cuda.is_available()` == True
- Python 3.12 via `uv`, ambiente em `.venv`
- PyTorch 2.14 (cu130), **OpenCV 5.0** (`opencv-python-headless` — major version nova, checar API
  antes de assumir comportamento de tutoriais antigos), segmentation-models-pytorch, albumentations,
  numpy, matplotlib, scikit-learn, tqdm, onnx, onnxruntime, fastapi, uvicorn, python-multipart, jupyter
- `check.py` na raiz valida GPU + forward pass do `smp.Unet`

## Comandos

```bash
uv sync                        # instalar/sincronizar deps
uv run python check.py         # validar GPU + modelo
uv add <pacote>                # ÚNICA forma de adicionar dependência (nunca editar pyproject.toml à mão)
uv run python -m src.<modulo>  # rodar scripts de src/
uv run pytest                  # testes
uv run jupyter lab             # notebooks (notebooks/ só importa de src/, sem lógica solta)
uv run uvicorn src.api.main:app --reload   # API (quando existir)
```

## Estrutura de pastas

```
data/raw/idrid/     # dataset original IDRiD — git-ignored. AINDA NÃO BAIXADO (vazio no momento).
                     #   Não supor nomes de pasta/arquivo: checar no disco quando existir.
data/processed/     # tiles 512x512 pré-processados — git-ignored
notebooks/          # exploração; só importa de src/, sem lógica duplicada
src/
  dataset.py         # Dataset PyTorch + augmentations (albumentations)
  preprocess.py       # recorte do círculo da retina, CLAHE, tiling
  train.py            # loop de treino, AMP, métricas, checkpoints
  infer.py             # janela deslizante, remontagem, export ONNX
  postprocess.py       # máscara -> lesões (OpenCV), regras de grau
  models/               # arquitetura/config do modelo (não confundir com pesos treinados)
  api/                   # FastAPI app (backend fica em src/api, não em api/ na raiz)
  retina_seg/             # pacote instalável (entry point `retina-seg`)
models/              # checkpoints .pt e exports .onnx — git-ignored, nunca versionar pesos
web/                 # front React (overlay SVG + laudo)
tests/               # pytest — cobre principalmente pré e pós-processamento
docs/                # ROADMAP.md e outras notas de projeto
```

## Decisões técnicas

- **Pré-processamento**: recorte do círculo da retina, CLAHE, tiles 512x512 com sobreposição,
  salvos em disco em `data/processed/`. Nunca redimensionar a imagem inteira para baixa resolução
  — microaneurismas somem.
- **Modelo**: `smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet")`, 4 canais de
  saída com sigmoid (um por lesão: MA, HE, EX, SE).
- **Loss**: Dice + BCE (ou Focal).
- **Treino**: AMP (`torch.autocast`), batch inicial ~8 com fallback para 4, gradient accumulation
  se necessário para caber em 8 GB VRAM.
- **Validação**: ~8 imagens separadas do treino oficial (54 imagens). O conjunto de teste oficial
  (27 imagens) é usado **só** para avaliação final — nunca em treino/validação.
- **Métricas**: AUPR por tipo de lesão (métrica oficial IDRiD) e Dice.
- **Inferência**: janela deslizante com sobreposição + remontagem da máscara na resolução original;
  export para ONNX.
- **Pós-processamento**: OpenCV (contornos / componentes conectados), descartar regiões minúsculas.
- **Grau**: regras derivadas das lesões (inspiradas na escala ICDR) por enquanto; classificador
  dedicado fica para uma fase posterior.
- **Laudo**: LLM via API recebe só o resumo estruturado (JSON), proibido de inventar achados fora
  dele; template fixo determinístico como fallback caso a chamada falhe.
- **Front**: React, overlay SVG escalado pela razão (tamanho exibido / tamanho real da imagem).

## Contrato da API

`POST /analyze` recebe a imagem e devolve coordenadas na **resolução original** (o front escala e
desenha):

```json
{
  "image": { "width": 0, "height": 0 },
  "grade": { "value": 0, "label": "" },
  "lesions": [
    { "type": "MA|HE|EX|SE", "cx": 0, "cy": 0, "r": 0, "area": 0, "score": 0.0, "quadrant": "" }
  ],
  "summary": { "...contagens por tipo...": 0 },
  "report": "texto do laudo, incluindo aviso de que não substitui avaliação oftalmológica"
}
```

## Convenções

- Código, identificadores e commits em inglês; explicações para o usuário em português.
- Dependências só via `uv add`; execução só via `uv run`.
- `data/` e `models/` no `.gitignore` — nunca versionar dataset ou pesos.
- Nunca misturar imagens de teste no treino ou na validação.
- Passos pequenos e verificáveis: testar em poucas imagens antes de rodar tudo. **Não iniciar
  treinos longos (mais de ~10 min) sem perguntar antes.**
- Seeds fixas; configuração centralizada em um único lugar (dataclass ou YAML) — sem números
  mágicos espalhados pelo código.
- Lógica reutilizável vive em `src/` (notebooks só importam); testes com pytest em `tests/` para
  pré e pós-processamento.
- Em caso de dúvida sobre caminhos ou formato de dados, **verificar no disco** em vez de supor
  (especialmente relevante agora: `data/raw/idrid/` ainda está vazio).
