"""MCP server that exposes travel-related tools."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

mcp = FastMCP("Travel Tools")

WMO_CONDITIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@mcp.tool()
def get_weather(city: str) -> str:
    """Get today's weather for a city.

    Call this tool whenever you need current or today's conditions to plan a trip,
    pack clothing, or decide between indoor and outdoor activities. Pass a city
    name such as "Paris" or "Tokyo". Returns a short summary of today's
    temperature in Celsius and the weather conditions. No API key is required.
    """
    query = city.strip()
    if not query:
        return "Please provide a city name."

    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urlencode({"name": query, "count": 1, "language": "en", "format": "json"})
        )
        results = geo.get("results") or []
        if not results:
            return f"Could not find a location named '{query}'."

        place = results[0]
        place_name = place.get("name", query)
        country = place.get("country")
        label = f"{place_name}, {country}" if country else place_name

        forecast = _get_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urlencode(
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code",
                    "timezone": "auto",
                }
            )
        )
        current = forecast.get("current") or {}
        temperature = current.get("temperature_2m")
        code = current.get("weather_code")
        conditions = WMO_CONDITIONS.get(int(code), "mixed conditions") if code is not None else "unknown conditions"

        if temperature is None:
            return f"Weather data for {label} is currently unavailable."
        return f"Today in {label}: {temperature}°C, {conditions}."
    except (URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return f"Could not fetch weather for '{query}': {exc}"


if __name__ == "__main__":
    mcp.run()
