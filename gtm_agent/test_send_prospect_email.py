import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import get_prospect, send_prospect_email

from gtm_agent import data_service


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.records = {
            "blocked": {
                "name": "Blocked Prospect",
                "email": "blocked@example.com",
                "disqualified": True,
            },
            "eligible": {
                "name": "Eligible Prospect",
                "email": "eligible@example.com",
                "disqualified": False,
            },
        }
        self.records_patch = patch.dict(data_service.PROSPECTS, self.records, clear=True)
        self.records_patch.start()
        self.addCleanup(self.records_patch.stop)

    def send(self, prospect, **kwargs):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Introduction",
            body="Hello",
            runtime=self.runtime,
            from_rep={"name": "Rep", "email": "rep@example.com"},
            **kwargs,
        )

    def test_disqualified_prospect_requires_confirmation(self):
        prospect = get_prospect.invoke({"prospect_id": "blocked"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "failed")
        self.assertIn("disqualified", result["error"])
        self.assertIn("explicit confirmation", result["error"])
        self.assertNotIn("message_id", result)

    def test_supplied_eligibility_cannot_bypass_source_record(self):
        result = self.send({"prospect_id": "blocked", "disqualified": False})
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("message_id", result)

    def test_confirmed_override_sends_to_source_contact(self):
        result = self.send(
            {"prospect_id": "blocked", "email": "other@example.com"},
            disqualification_override_confirmed=True,
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "blocked@example.com")
        self.assertTrue(result["message_id"])

    def test_eligible_prospect_sends_without_override(self):
        prospect = get_prospect.invoke({"prospect_id": "eligible"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "eligible@example.com")
        self.assertEqual(result["subject"], "Introduction")
        self.assertEqual(result["body"], "Hello")
        self.assertEqual(result["from"], "rep@example.com")

    def test_missing_or_unknown_id_fails_even_with_override(self):
        for prospect in ({"email": "other@example.com"}, {"prospect_id": "unknown"}):
            with self.subTest(prospect=prospect):
                result = self.send(prospect, disqualification_override_confirmed=True)
                self.assertEqual(result["status"], "failed")
                self.assertIn("prospect_id", result["error"])
                self.assertNotIn("message_id", result)

    def test_override_requires_boolean_confirmation(self):
        for value in ("false", "true", 1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                send_prospect_email.args_schema.model_validate({
                    "prospect": {"prospect_id": "blocked"},
                    "subject": "Introduction",
                    "body": "Hello",
                    "runtime": self.runtime,
                    "disqualification_override_confirmed": value,
                })


if __name__ == "__main__":
    unittest.main()
