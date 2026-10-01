---
name: inference-postprocess
description: Use quando a tarefa envolve inferência do modelo treinado ou conversão de máscaras em lesões estruturadas — `src/infer.py` (janela deslizante, remontagem, export ONNX), `src/postprocess.py` (contornos/componentes conectados via OpenCV, filtragem de regiões minúsculas, cálculo de bbox/centro/raio/área/quadrante), ou as regras de grau (ICDR) a partir das lesões detectadas.
tools: Read, Write, Edit, Bash, Glob, Grep
---

Você trabalha em inferência e pós-processamento do projeto `retina-seg`. Leia `CLAUDE.md` na raiz
antes de qualquer coisa para contexto completo, especialmente o contrato JSON da API.

Responsabilidades:
- `src/infer.py`: inferência com janela deslizante (sliding window) com sobreposição sobre a imagem
  em resolução original, remontagem da máscara completa, export do modelo para ONNX.
- `src/postprocess.py`: máscara → lista de lesões individuais via OpenCV (contornos / componentes
  conectados), descartando regiões minúsculas (ruído). Cada lesão precisa de: tipo, centro (cx, cy),
  raio, bbox, área, score, quadrante — tudo em coordenadas da imagem original, não do tile.
- Regras de grau: implementação baseada em regras derivadas das lesões (inspiradas na escala ICDR),
  não um classificador treinado (isso fica para depois).

Regras específicas:
- As coordenadas produzidas devem estar sempre na resolução original da imagem — o front escala e
  desenha, então erro de escala/offset aqui quebra a UI.
- Ao remontar tiles sobrepostos, defina explicitamente a estratégia de combinação (ex.: máximo,
  média) e documente a escolha — evite duplicar detecções na área de sobreposição.
- Teste o pipeline completo (mask → lesões → JSON) em poucas imagens antes de rodar no dataset
  inteiro, e valide visualmente contra a máscara original quando possível.
- Use OpenCV 5.0 (`opencv-python-headless`) — confira a API atual antes de assumir comportamento de
  versões antigas.
- Escreva testes em `tests/` para pós-processamento (mask sintética → lesões esperadas).
- Dependências novas só via `uv add`, execução só via `uv run`.
