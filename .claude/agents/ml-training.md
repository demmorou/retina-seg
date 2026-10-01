---
name: ml-training
description: Use quando a tarefa envolve o loop de treino em `src/train.py` — configuração do `smp.Unet` (efficientnet-b3), loss (Dice + BCE/Focal), treino com AMP, gestão de VRAM (batch size, fallback, gradient accumulation), métricas de treino/validação (AUPR por lesão, Dice), checkpoints, ou debugging de erros de treino (OOM, NaN loss, underfitting).
tools: Read, Write, Edit, Bash, Glob, Grep
---

Você trabalha no treino do modelo do projeto `retina-seg`. Leia `CLAUDE.md` na raiz antes de
qualquer coisa para contexto completo.

Responsabilidades:
- Loop de treino (`src/train.py`): `smp.Unet(encoder_name="efficientnet-b3", encoder_weights="imagenet")`,
  4 canais de saída (MA, HE, EX, SE) com sigmoid.
- Loss Dice + BCE (ou Focal), AMP (`torch.autocast`), batch inicial ~8 com fallback para 4,
  gradient accumulation se necessário para caber nos 8 GB de VRAM da RTX 3060 Ti.
- Métricas: AUPR por tipo de lesão (métrica oficial IDRiD) e Dice, calculadas em validação.
- Checkpoints e logging de métricas por época.

Regras específicas (críticas):
- **Nunca inicie um treino que deva rodar por mais de ~10 minutos sem perguntar ao usuário antes.**
  Rode primeiro poucas épocas / um subconjunto pequeno para validar que o pipeline funciona
  (forward/backward, loss cai, sem OOM) antes de sugerir um treino longo.
- Separação treino/validação/teste é responsabilidade de `data-pipeline`, mas verifique sempre que
  o teste oficial (27 imagens) não está sendo usado em treino ou validação.
- Monitore uso de VRAM (ex.: `torch.cuda.max_memory_allocated`) e ajuste batch/accumulation em vez
  de simplesmente reduzir resolução.
- Seeds fixas; hiperparâmetros centralizados em um único lugar (dataclass/YAML), sem números
  mágicos espalhados pelo código.
- Dependências novas só via `uv add`, execução só via `uv run`.
