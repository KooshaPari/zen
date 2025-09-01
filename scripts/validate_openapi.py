from __future__ import annotations

from pathlib import Path


def main() -> int:
    try:
        from openapi_spec_validator import validate_spec
        from openapi_spec_validator.readers import read_from_filename
    except Exception:
        print("openapi-spec-validator not installed; skipping validation")
        return 0

    # Validate archived legacy spec
    root = Path(__file__).resolve().parent.parent / "archive" / "openapi" / "openapi.json"
    if not root.exists():
        print(f"OpenAPI file not found at {root}")
        return 1
    try:
        spec_dict, base_uri = read_from_filename(str(root))
        validate_spec(spec_dict, base_uri=base_uri)
        print("OpenAPI spec validation: OK")
        return 0
    except Exception as e:
        print(f"OpenAPI spec validation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
