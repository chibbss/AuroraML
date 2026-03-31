import json
from typing import Any
from urllib import error, request

from app.core.config import settings


class AuroraAIService:
    """Grounded LLM helper for dataset Q&A."""

    @staticmethod
    def _build_context_payload(
        dataset_name: str,
        report: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset_name": dataset_name,
            "overview": report.get("overview", {}),
            "modeling_readiness": report.get("modeling_readiness", {}),
            "target_analysis": report.get("target_analysis", {}),
            "findings": report.get("findings", [])[:6],
            "recommendations": report.get("recommendations", [])[:5],
            "segments": report.get("segments", [])[:5],
            "feature_roles": report.get("feature_roles", [])[:8],
            "feature_spotlight": report.get("feature_spotlight", [])[:8],
            "research_report": report.get("research_report", [])[:4],
            "sample_data": profile.get("sample_data", [])[:5],
            "dtypes": profile.get("dtypes", {}),
            "missing_percentage": profile.get("missing_percentage", {}),
        }

    @staticmethod
    def _fallback_answer(question: str, context: dict[str, Any]) -> tuple[str, list[str]]:
        overview = context.get("overview", {})
        target = context.get("target_analysis", {}).get("recommended_target") or "not yet confirmed"
        readiness = context.get("modeling_readiness", {})
        findings = context.get("findings", [])
        recommendations = context.get("recommendations", [])
        segments = context.get("segments", [])
        lowered_question = question.lower()

        answer_parts = [
            (
                f"AuroraML analyzed `{context.get('dataset_name', 'this dataset')}` with "
                f"{overview.get('rows', 0):,} rows and {overview.get('columns', 0)} columns."
            ),
            (
                f"The current recommended target is `{target}`, and modeling readiness is "
                f"{readiness.get('score', 0)}/100 with status `{readiness.get('status', 'unknown')}`."
            ),
        ]

        if "target" in lowered_question or "label" in lowered_question:
            rationale = context.get("target_analysis", {}).get("rationale")
            if rationale:
                answer_parts.append(rationale)

        if ("risk" in lowered_question or "issue" in lowered_question or "problem" in lowered_question) and findings:
            answer_parts.append("Top detected risks: " + " ".join(item["detail"] for item in findings[:3]))

        if ("segment" in lowered_question or "cohort" in lowered_question) and segments:
            answer_parts.append("Key cohort signals: " + " ".join(item["insight"] for item in segments[:2]))

        if ("next" in lowered_question or "recommend" in lowered_question or "improve" in lowered_question) and recommendations:
            answer_parts.append("Recommended next steps: " + " ".join(recommendations[:3]))

        if len(answer_parts) == 2 and findings:
            answer_parts.append("Most important findings: " + " ".join(item["detail"] for item in findings[:2]))

        citations = ["overview", "target_analysis", "findings"]
        if segments:
            citations.append("segments")
        return " ".join(answer_parts), citations

    @staticmethod
    def answer_dataset_question(
        dataset_name: str,
        question: str,
        report: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        context = AuroraAIService._build_context_payload(dataset_name, report, profile)

        if not settings.AURORA_LLM_ENABLED or not settings.OPENAI_API_KEY:
            answer, citations = AuroraAIService._fallback_answer(question, context)
            return {
                "answer": answer,
                "citations": citations,
                "provider": "fallback",
                "grounded": True,
            }

        payload = {
            "model": settings.OPENAI_MODEL,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are Aurora, a production-grade data analyst for AuroraML. "
                                "Answer only from the supplied structured dataset context. "
                                "Do not invent columns, findings, or model behavior. "
                                "If the answer is uncertain, say so clearly. "
                                "Be concise and analytical."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Dataset context:\n{json.dumps(context, ensure_ascii=True)}\n\n"
                                f"Question: {question}\n\n"
                                "Return a short grounded answer."
                            ),
                        }
                    ],
                },
            ],
        }

        req = request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=45) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            answer, citations = AuroraAIService._fallback_answer(question, context)
            return {
                "answer": answer,
                "citations": citations,
                "provider": "fallback",
                "grounded": True,
                "warning": f"OpenAI request failed with status {exc.code}: {detail[:200]}",
            }
        except Exception as exc:
            answer, citations = AuroraAIService._fallback_answer(question, context)
            return {
                "answer": answer,
                "citations": citations,
                "provider": "fallback",
                "grounded": True,
                "warning": f"OpenAI request failed: {str(exc)}",
            }

        answer_text = raw.get("output_text")
        if not answer_text:
            output = raw.get("output", [])
            text_parts: list[str] = []
            for item in output:
                for content_item in item.get("content", []):
                    text_value = content_item.get("text")
                    if text_value:
                        text_parts.append(text_value)
            answer_text = "\n".join(text_parts).strip()

        if not answer_text:
            answer_text, citations = AuroraAIService._fallback_answer(question, context)
            return {
                "answer": answer_text,
                "citations": citations,
                "provider": "fallback",
                "grounded": True,
                "warning": "OpenAI returned an empty response.",
            }

        return {
            "answer": answer_text,
            "citations": ["overview", "target_analysis", "findings", "segments", "research_report"],
            "provider": "openai",
            "grounded": True,
        }
