from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_tasks.py"

spec = importlib.util.spec_from_file_location("validate_tasks", SCRIPT_PATH)
validate_tasks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validate_tasks
spec.loader.exec_module(validate_tasks)


class ValidateTasksFixtureTests(unittest.TestCase):
    def test_fixture_required_fields_are_checked_for_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            task_dir = project_root / "tasks" / "T_fixture_schema"
            fixture_dir = project_root / "fixtures" / "gmail"
            task_dir.mkdir(parents=True)
            fixture_dir.mkdir(parents=True)

            (task_dir / "task.yaml").write_text(
                textwrap.dedent(
                    """
                    task_id: T_fixture_schema
                    task_name: Fixture schema validation
                    version: "1.0"
                    prompt:
                      text: Check fixture schema.
                      language: en
                    services:
                      - name: gmail
                        command: python mock_services/gmail/server.py
                        port: 9100
                        health_check: http://localhost:9100/gmail/messages
                        reset_endpoint: http://localhost:9100/gmail/reset
                        env:
                          GMAIL_FIXTURES: fixtures/gmail/messages.json
                    tools: []
                    tool_endpoints: []
                    scoring_components:
                      - name: done
                        weight: 1.0
                        check:
                          type: keywords_present
                          keywords: [done]
                    safety_checks: []
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (task_dir / "grader.py").write_text(
                textwrap.dedent(
                    """
                    from claw_eval.graders.base import AbstractGrader


                    class DemoGrader(AbstractGrader):
                        def grade(self, messages, dispatches, task, **kwargs):
                            return {}
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (fixture_dir / "messages.json").write_text(
                json.dumps(
                    [
                        {
                            "message_id": "m1",
                            "from": "a@example.com",
                            "subject": "Complete",
                            "date": "2026-05-10",
                            "body": "ok",
                        },
                        {
                            "message_id": "m2",
                            "from": "b@example.com",
                            "subject": "Missing body",
                            "date": "2026-05-10",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            original_project_root = validate_tasks.PROJECT_ROOT
            validate_tasks.PROJECT_ROOT = project_root
            try:
                validator = validate_tasks.TaskValidator(task_dir)
                self.assertFalse(validator.validate())
            finally:
                validate_tasks.PROJECT_ROOT = original_project_root

            self.assertIn(
                "fixture fixtures/gmail/messages.json[1]: missing required field "
                "'body' (expected for gmail)",
                validator.errors,
            )


if __name__ == "__main__":
    unittest.main()
