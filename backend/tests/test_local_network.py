from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.main import app


class LocalNetworkConfigurationTests(unittest.TestCase):
    def test_cors_origins_can_be_configured_for_the_lan_frontend_origin(self):
        with patch.dict(
            "os.environ",
            {"ATHLETELENS_CORS_ORIGINS": "http://localhost:5173,http://192.168.1.25:5173"},
            clear=False,
        ):
            settings = get_settings()
        self.assertEqual(
            settings.cors_origins,
            ("http://localhost:5173", "http://192.168.1.25:5173"),
        )

    def test_fastapi_cors_middleware_allows_local_desktop_origins_without_credentials(self):
        middleware = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
        self.assertIn("http://localhost:5173", middleware.kwargs["allow_origins"])
        self.assertIn("http://127.0.0.1:5173", middleware.kwargs["allow_origins"])
        self.assertFalse(middleware.kwargs["allow_credentials"])

    def test_cors_configuration_rejects_wildcard_and_non_origin_urls(self):
        for configured in ("*", "https://*.example.com", "https://example.com/path", "file:///tmp"):
            with self.subTest(origin=configured), patch.dict(
                "os.environ", {"ATHLETELENS_CORS_ORIGINS": configured}, clear=False
            ), self.assertRaises(ValueError):
                get_settings()


if __name__ == "__main__":
    unittest.main()
