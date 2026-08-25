import unittest
import io
from unittest.mock import Mock, patch

from docx import Document

from cv_engine import (ats_analysis, build_docx, build_rag_context, parse_json,
                       run_pipeline)


class EngineTests(unittest.TestCase):
    def test_json_and_docx(self):
        resume = parse_json('```json\n{"name":"Test User","skills":["Python"]}\n```')
        self.assertEqual(resume["name"], "Test User")
        self.assertEqual(build_docx(resume)[:2], b"PK")

    def test_docx_includes_all_non_empty_editable_fields(self):
        resume = {
            "name":"Test User", "headline":"Engineer", "email":"test@example.com", "phone":"123", "location":"Pune", "linkedin":"linkedin.com/in/test", "summary":"Summary", "skills":["Python"],
            "experience":[{"role":"Developer", "company":"Example", "location":"Mumbai", "start_date":"2020", "end_date":"2022", "bullets":["Built services."]}],
            "projects":[{"name":"Portfolio", "technologies":"Python", "link":"https://example.com/project", "bullets":["Created application."]}],
            "education":[{"degree":"B.Tech", "institution":"Example University", "location":"Delhi", "start_date":"2016", "end_date":"2020", "details":["Honors"]}],
            "certifications":[{"name":"Certificate", "issuer":"Issuer", "date":"2023"}],
        }
        document = Document(io.BytesIO(build_docx(resume)))
        output = "\n".join(paragraph.text for paragraph in document.paragraphs)
        for value in ("Mumbai", "https://example.com/project", "Delhi", "2020", "2022", "2016", "Certificate", "Issuer"):
            self.assertIn(value, output)

    @patch("cv_engine.requests.post")
    def test_two_distinct_provider_calls(self, post):
        def api_response(url, **kwargs):
            response = Mock(ok=True)
            if url.endswith("/embeddings"):
                inputs = kwargs["json"]["input"]
                response.json.return_value = {"data":[{"index": index, "embedding":[1.0, 0.0]} for index, _ in enumerate(inputs)]}
            elif "huggingface.co" in url:
                response.json.return_value = {"choices": [{"message": {"content": '{"resume":{"name":"Test","skills":["Python"]},"jd_analysis":{"target_job_title":"Backend Engineer","keywords":["SQL"]},"skill_categories":{},"capability_evidence":[{"capability":"backend development","supporting_resume_evidence":"Built APIs","demonstrated_technologies":["Python"]}]}'}}]}
            else:
                response.json.return_value = {"choices": [{"message": {"content": '{"name":"Test","skills":["Python","SQL"]}'}}]}
            return response
        post.side_effect = api_response
        _, tailored = run_pipeline("RAW-CV-UNIQUE-SENTINEL", "Python SQL", "openai-key", "hf-token", "gpt-4o", "Qwen/test")
        self.assertEqual(tailored["skills"], ["Python", "SQL"])
        self.assertIn("huggingface.co", post.call_args_list[0][0][0])
        self.assertIn("/embeddings", post.call_args_list[1][0][0])
        self.assertIn("/embeddings", post.call_args_list[2][0][0])
        self.assertIn("api.openai.com", post.call_args_list[3][0][0])
        preprocessing_prompt = post.call_args_list[0][1]["json"]["messages"][1]["content"]
        self.assertIn("skill_categories", preprocessing_prompt)
        self.assertIn("jd_analysis", preprocessing_prompt)
        self.assertIn("capability_evidence", preprocessing_prompt)
        tailoring_prompt = post.call_args_list[3][1]["json"]["messages"][1]["content"]
        self.assertIn("SELECT only", tailoring_prompt)
        self.assertIn("Do not preserve the original bullet count", tailoring_prompt)
        self.assertIn("only permitted source of candidate facts", tailoring_prompt)
        self.assertIn("Build the resume from the TARGET JD", tailoring_prompt)
        self.assertIn("Targeting Java Ecosystems", tailoring_prompt)
        self.assertIn("must not state or imply hands-on experience", tailoring_prompt)
        self.assertIn("RAG RETRIEVED CANDIDATE EVIDENCE", tailoring_prompt)
        self.assertIn("Built APIs", tailoring_prompt)
        self.assertNotIn("RAW-CV-UNIQUE-SENTINEL", tailoring_prompt)
        self.assertEqual(post.call_count, 4)

    @patch("cv_engine.create_embeddings")
    def test_semantic_rag_retrieves_different_wording(self, create_embeddings):
        preprocessed = {"resume": {"experience": [{"role": "Developer", "company": "Example", "bullets": ["Built Python APIs and automated data processing."]}], "projects": [], "skills": ["Python", "REST APIs"], "education": []}, "jd_analysis": {"required_skills": ["APIs"], "keywords": ["automation", "backend"]}, "capability_evidence": []}
        # Candidate says "automated data processing" while the JD says "workflow orchestration".
        # Embeddings intentionally match the shared capability without any exact phrase overlap.
        create_embeddings.side_effect = [
            [[0.0, 1.0], [1.0, 0.0]],  # experience chunk, skills chunk
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],  # API/workflow/backend requirements
        ]
        preprocessed["jd_analysis"] = {"required_skills": ["service endpoints"], "keywords": ["workflow orchestration", "server-side systems"]}
        context = build_rag_context(preprocessed, "test-key")
        evidence = " ".join(item["evidence"] for item in context["retrieved_candidate_evidence"])
        self.assertEqual(context["index_size"], 2)
        self.assertIn("automated data processing", evidence)
        self.assertEqual(create_embeddings.call_count, 2)
        self.assertNotIn("workflow orchestration", evidence)

    @patch("cv_engine.create_embeddings")
    def test_semantic_rag_excludes_unsupported_skill(self, create_embeddings):
        preprocessed = {"resume": {"experience": [{"role":"Developer", "company":"Example", "bullets":["Maintained REST services."]}], "projects":[], "skills":["Python"], "education":[]}, "jd_analysis":{"required_skills":["Kubernetes"], "keywords":[]}, "capability_evidence":[]}
        create_embeddings.side_effect = [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0]]]
        context = build_rag_context(preprocessed, "test-key")
        self.assertEqual(context["retrieved_candidate_evidence"], [])

    def test_ats_analysis(self):
        score, matched, missing = ats_analysis({"skills": ["Python"]}, "Python SQL")
        self.assertIn("python", matched)
        self.assertIn("sql", missing)
        self.assertGreater(score, 0)

    @patch("cv_engine.requests.post")
    def test_hugging_face_permission_error(self, post):
        response = Mock(ok=False, status_code=403, text="Forbidden")
        response.json.return_value = {"error": "insufficient permissions"}
        post.return_value = response
        with self.assertRaisesRegex(Exception, "Make calls to Inference Providers"):
            from cv_engine import post_chat
            post_chat("https://router.huggingface.co/v1/chat/completions", "token", "model", [])

    @patch("cv_engine.time.sleep")
    @patch("cv_engine.requests.post")
    def test_hugging_face_timeout_is_retried_and_sanitized(self, post, sleep):
        response = Mock(ok=False, status_code=504, text="<html>gateway timeout</html>")
        response.json.side_effect = ValueError()
        post.return_value = response
        with self.assertRaisesRegex(Exception, "timed out after 3 attempts"):
            from cv_engine import post_chat
            post_chat("https://router.huggingface.co/v1/chat/completions", "token", "model", [])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
