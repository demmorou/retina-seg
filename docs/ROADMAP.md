# Roadmap — retina-seg

Ordem de execução. Cada etapa só começa depois de aprovada a anterior. Nenhuma etapa inclui treino
longo (>~10 min) sem confirmação explícita do usuário.

## 0. Estruturação do projeto (esta etapa)
- [x] `CLAUDE.md` com visão geral, ambiente, comandos, estrutura, decisões técnicas, contrato de API
  e convenções.
- [x] Subagentes em `.claude/agents/`: `data-pipeline`, `ml-training`, `inference-postprocess`,
  `api-backend`, `frontend`, `code-reviewer`.
- [x] `docs/ROADMAP.md` (este arquivo).
- **Pronto quando**: usuário aprova o conteúdo destes artefatos.

## 1. Dataset
- [x] Baixar/posicionar o IDRiD "A. Segmentation" em `data/raw/idrid/` (via `download-kaggle.py`,
  espelho `aaryapatel98/indian-diabetic-retinopathy-image-dataset` no Kaggle — o bucket S3 oficial
  do IEEE DataPort não permite acesso anônimo).
- [x] Estrutura real encontrada em disco (`data/raw/idrid/A. Segmentation/`):
  - `1. Original Images/a. Training Set/IDRiD_NN.jpg` — 54 imagens, `1. Original Images/b. Testing Set/` — 27 imagens. Todas 4288x2848 RGB.
  - `2. All Segmentation Groundtruths/{a. Training Set,b. Testing Set}/<N>. <Tipo>/IDRiD_NN_<SIGLA>.tif`:
    `1. Microaneurysms` (MA), `2. Haemorrhages` (HE), `3. Hard Exudates` (EX), `4. Soft Exudates` (SE),
    `5. Optic Disc` (OD, landmark anatômico — não é lesão, não entra nos 4 canais do modelo).
  - **Nem toda imagem tem máscara de todo tipo** (treino: MA 54/54, HE 53/54, EX 54/54, SE 26/54,
    OD 54/54) — arquivo ausente deve ser tratado como máscara toda-zero, nunca erro.
- **Pronto** ✅ — falta só escrever a função que lista os pares imagem/máscara programaticamente
  dentro de `src/preprocess.py` ou `src/dataset.py` (nenhum código em `src/` lê o dataset ainda).

## 2. Pré-processamento
- [x] `src/preprocess.py`: recorte do círculo da retina (bbox via threshold no canal de cinza),
  CLAHE no canal L (LAB), tiling 512x512 com stride 384 (25% overlap, descarta tiles com menos de
  10% de conteúdo de retina), gravação em `data/processed/{train,val,test}/{images,masks}/`.
- [x] Separação determinística de 8 imagens de validação a partir das 54 de treino oficiais
  (`split_train_val`, seed fixa em `CONFIG.seed`); teste oficial (27) isolado.
- [x] Verificado visualmente em 3 imagens (crop, CLAHE, alinhamento máscara×imagem, alinhamento
  entre tiles vizinhos na borda do overlap) antes de rodar no dataset completo.
- [x] Achado de dataset: `IDRiD_81_EX.tif` (Testing Set) é um TIFF RGBA em vez de escala de cinza
  — `load_mask` trata isso de forma genérica (descarta alpha, reduz cor via max).
- [x] Rodado no dataset completo: 46 treino → 3344 tiles, 8 val → 576 tiles, 27 teste → 1984 tiles
  (~7.9 GB em `data/processed/`, ~80s de execução).
- [x] 11 testes em `tests/` (`test_preprocess.py` com dataset real, `test_preprocess_synthetic.py`
  sem dependência do dataset).
- **Pronto** ✅

## 3. Dataset PyTorch e augmentations
- [x] `src/dataset.py`: `RetinaTileDataset` lê os tiles processados (`data/processed/<split>/`),
  aplica augmentations via albumentations (`build_train_transform`/`build_val_transform`: flips,
  rotação 90°, leve brilho/contraste, normalização ImageNet, `ToTensorV2(transpose_mask=True)`),
  devolve `image` (3ch float32 normalizado) e `mask` (4ch float32 0/1, ordem `CONFIG.lesion_types`).
- [x] Normalização ImageNet centralizada em `CONFIG.imagenet_mean`/`imagenet_std`.
- [x] 5 testes sintéticos em `tests/test_dataset_synthetic.py` (shapes, dtypes, `DataLoader`,
  alinhamento imagem/máscara após flip).
- [x] Verificado com dataset real: `DataLoader` de 8 batches itera sem erro
  (`[8, 3, 512, 512]` / `[8, 4, 512, 512]`), valores de máscara em {0,1}, e visualização de um tile
  com lesões confirma alinhamento (EX sobre exsudatos amarelos, HE sobre hemorragias escuras).
- **Pronto** ✅

## 4. Treino
- [x] `src/train.py`: `smp.Unet` (efficientnet-b3), loss Dice+BCE (multilabel), AMP
  (`torch.autocast`+`GradScaler`), batch 8 com fallback pra 4 + gradient accumulation em caso de
  OOM, checkpoints (`models/checkpoint_last.pt`/`checkpoint_best.pt`), métricas de validação por
  época (Dice via tp/fp/fn e AUPR via `sklearn`, ambas por tipo de lesão). CLI:
  `uv run python -m src.train --epochs N [--quick] [--batch-size N] [--resume path]`.
- [x] Smoke test (`--epochs 2 --quick`, 200 tiles treino / 64 val) rodado pelo usuário: `device: cuda`,
  sem OOM, loss caiu de forma consistente (treino 1.6055→1.5074, val 1.6826→1.5025), Dice e AUPR
  calculados e logados por tipo de lesão a cada época. Valores baixos são esperados (2 épocas,
  amostra pequena, lesões são fração mínima dos pixels).
- [x] Treino completo (30 épocas) rodado pelo usuário. `models/checkpoint_best.pt` = época 15,
  Dice médio 0.613 (MA/HE/EX/SE); `models/checkpoint_last.pt` = época 30 (sem melhora depois da
  época 15 — usar sempre `checkpoint_best.pt` daqui pra frente, `checkpoint_last.pt` só serve pra
  `--resume`). Ambos git-ignored, não versionados.
- **Pronto** ✅

## 5. Inferência e pós-processamento
- [x] `src/infer.py`: `predict_image` — recorte+CLAHE (reusa etapa 2), janela deslizante 512x512
  com `tile_starts` (também reusado de `src/preprocess.py`), média nas sobreposições, remontagem da
  probabilidade em resolução original. `export_onnx` (via `torch.onnx.export(..., dynamo=False)` —
  o exportador "dynamo" padrão do torch 2.14 exige o pacote extra `onnxscript`, evitado assim).
  CLI: `uv run python -m src.infer <imagem> [--checkpoint path] [--export-onnx path]`.
- [x] `src/postprocess.py`: `mask_to_lesions`/`masks_to_lesions` — componentes conectados
  (`cv2.connectedComponentsWithStats`) por tipo de lesão, filtro de área mínima
  (`CONFIG.lesion_min_area`), cx/cy/r/área/score (probabilidade média do componente)/bbox/quadrante
  (geométrico — superior/inferior × esquerdo/direito; lateralidade clínica nasal/temporal não dá
  pra inferir só da imagem).
- [x] `compute_grade`: regras heurísticas inspiradas no ICDR (regra 4-2-1 aproximada pela dispersão
  de hemorragias entre quadrantes/contagem — `CONFIG.severe_he_quadrants`/`severe_he_count`). Grau
  máximo derivável é 3 (RDNP grave) — o modelo não detecta neovascularização/IRMA/rosário venoso,
  então grau 4 (proliferativa) nunca é atingido por este método. Documentado como não-clínico no
  docstring.
- [x] 9 testes sintéticos em `tests/test_postprocess_synthetic.py` (centroide/área/bbox/quadrante,
  filtro de threshold e área mínima, agregação por tipo, todas as faixas de grau).
- [x] Pipeline completo rodado em `IDRiD_55` (Testing Set, nunca visto em treino/validação) usando
  `checkpoint_best.pt`: 565 lesões (EX 467, MA 69, HE 29), grau 3 (RDNP grave). Comparação visual
  lado a lado com a máscara real do IDRiD confirma alinhamento — formato e distribuição espacial de
  MA/HE/EX praticamente idênticos ao ground truth; SE corretamente vazio nos dois (imagem sem
  exsudatos moles). Export ONNX testado com `onnxruntime`: saída `(1, 4, 512, 512)`, 50 MB.
- **Pronto** ✅

## 6. API
- [x] `src/api/main.py`: FastAPI, `POST /analyze` — recebe upload de imagem (decodificada via
  `cv2.imdecode`, sem gravar em disco), roda `src/infer.predict` + `src/postprocess`, devolve
  exatamente o contrato JSON do `CLAUDE.md`. Modelo carregado uma vez no `lifespan` (não por
  request). `src/infer.py` ganhou `predict(model, image_array, device)` reaproveitado tanto pelo
  CLI (`predict_image`, carrega de arquivo) quanto pela API (recebe array já decodificado).
- [x] `src/api/schemas.py`: modelos Pydantic do contrato (`ImageInfo`, `GradeInfo`, `LesionOut`,
  `AnalyzeResponse`).
- [x] `src/api/report.py`: `build_report` — template fixo determinístico, sempre inclui o aviso
  obrigatório (`MANDATORY_DISCLAIMER`). **Decisão do usuário (2026-09-28): sem integração de LLM por
  enquanto** — só o template; ponto de integração documentado no docstring pra quando for plugado
  (LLM recebe só `summary`/`grade_label`, nunca a imagem, e cai pro template se falhar).
- [x] 3 testes em `tests/test_report.py` (sintéticos) + 2 em `tests/test_api.py` (via
  `TestClient`, contrato completo + rejeição de imagem inválida) — pulados automaticamente se
  `models/checkpoint_best.pt` não existir.
- [x] Validado com servidor real (`uv run uvicorn src.api.main:app`) + `curl` numa imagem de teste
  do IDRiD (`IDRiD_55`): JSON válido, coordenadas em resolução original (4288x2848), grau 3, laudo
  com o aviso obrigatório.
- **Pronto** ✅

## 7. Front web
- `web/`: React, upload/seleção de imagem, overlay SVG das lesões (cor por tipo, toggle por tipo)
  escalado corretamente, exibição do laudo.
- **Pronto quando**: consigo abrir o front, enviar uma imagem, ver os círculos das lesões
  corretamente posicionados sobre a imagem exibida (em pelo menos duas resoluções de tela/zoom
  diferentes) e o laudo renderizado.
