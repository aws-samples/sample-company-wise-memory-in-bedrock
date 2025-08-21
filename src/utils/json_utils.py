"""
JSON utilities for cleaning LLM responses.
"""


def clean_json_response(response: str) -> str:
    """Clean LLM response by removing code block markers.

    Args:
        response: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    response = response.strip()

    # Remove ```json and ``` markers
    if response.startswith('```json'):
        response = response[7:]
    elif response.startswith('```'):
        response = response[3:]

    if response.endswith('```'):
        response = response[:-3]

    return response.strip()
