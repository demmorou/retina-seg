"""Geração do laudo em texto a partir do resumo estruturado (grau + contagens por tipo de lesão).

Por enquanto só o template fixo determinístico (nenhum LLM integrado ainda — decisão do usuário em
2026-09-28, ver docs/ROADMAP.md etapa 6). Quando um LLM for integrado, ele deve receber só o dict
`summary`/`grade_label` produzidos aqui — nunca a imagem ou achados fora do resumo estruturado — e
cair para `build_report` (este template) se a chamada falhar, mantendo sempre o aviso obrigatório.
"""

LESION_LABELS = {
    "MA": "microaneurismas",
    "HE": "hemorragias",
    "EX": "exsudatos duros",
    "SE": "exsudatos moles",
}

MANDATORY_DISCLAIMER = (
    "Este laudo foi gerado automaticamente por um sistema experimental de apoio ao diagnóstico e "
    "NÃO substitui avaliação por um oftalmologista. Procure um profissional para diagnóstico e "
    "conduta clínica."
)


def build_report(grade_label: str, summary: dict[str, int]) -> str:
    """Template fixo determinístico — nunca inventa achados fora de `summary`."""
    total = sum(summary.values())

    if total == 0:
        findings = "Nenhuma lesão compatível com retinopatia diabética foi detectada na imagem."
    else:
        counts = ", ".join(
            f"{count} {LESION_LABELS[lesion_type]}" for lesion_type, count in summary.items() if count > 0
        )
        findings = f"Foram detectadas {total} lesões ao todo: {counts}."

    return f"{findings} Classificação estimada: {grade_label}.\n\n{MANDATORY_DISCLAIMER}"
