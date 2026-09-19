import asyncio
import html
import json
import os
import sys
from pathlib import Path

import chromadb
import psycopg2
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

APP_DIR = Path(__file__).resolve().parent
TIPS_PATH = APP_DIR / "tips.txt"
CHROMA_DIR = APP_DIR / "chroma"
WEATHER_SERVER = APP_DIR / "weather_server.py"
COLLECTION_NAME = "travel_tips"

st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600&display=swap");

        html, body, [class*="css"] {
            font-family: "Source Sans 3", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(900px 420px at 12% -8%, #d7eef2 0%, transparent 55%),
                linear-gradient(180deg, #f7f4ee 0%, #eef6f7 100%);
            color: #1f2a2e;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f4c5c 0%, #14616e 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8f4ee !important;
        }

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #fff !important;
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stFormSubmitButton > button {
            background: #e07a5f;
            color: #fff !important;
            border: none;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] .stFormSubmitButton > button:hover {
            background: #c96a52;
            color: #fff !important;
            border: none;
        }

        .hero {
            background: linear-gradient(135deg, #0f4c5c 0%, #1a7a86 62%, #e07a5f 160%);
            border-radius: 22px;
            padding: 1.6rem 1.8rem;
            color: #fff;
            box-shadow: 0 18px 40px rgba(15, 76, 92, 0.22);
            margin: 0.4rem 0 1.2rem 0;
            box-sizing: border-box;
            overflow: hidden;
        }

        .hero h1 {
            font-family: "Fraunces", serif;
            font-size: clamp(1.6rem, 3vw, 2.2rem);
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
            line-height: 1.2;
            white-space: normal;
        }

        .hero p {
            margin: 0;
            font-size: 1.02rem;
            opacity: 0.92;
            line-height: 1.45;
        }

        .placeholder-card {
            background: #fff;
            border: 1px dashed #c8d8dc;
            border-radius: 18px;
            padding: 2.6rem 1.6rem;
            text-align: center;
            color: #5c6b70;
            box-shadow: 0 10px 28px rgba(15, 76, 92, 0.08);
            box-sizing: border-box;
        }

        .placeholder-card .icon {
            font-size: 1.8rem;
        }

        .placeholder-card h3 {
            font-family: "Fraunces", serif;
            color: #0f4c5c;
            margin: 0.45rem 0 0.35rem 0;
        }

        .placeholder-card p {
            margin: 0;
        }

        .tips-box {
            background: #fff;
            border-left: 5px solid #e07a5f;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-top: 1rem;
            box-shadow: 0 10px 28px rgba(15, 76, 92, 0.08);
        }

        .tips-box h3 {
            font-family: "Fraunces", serif;
            color: #0f4c5c;
            margin: 0 0 0.7rem 0;
        }

        .tips-box ul {
            margin: 0;
            padding-left: 1.15rem;
            color: #1f2a2e;
        }

        .tips-box li {
            margin-bottom: 0.45rem;
        }

        .weather-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #0f4c5c;
            color: #fff;
            border-radius: 999px;
            padding: 0.42rem 0.95rem;
            font-size: 0.92rem;
            font-weight: 500;
            margin: 0 0 0.95rem 0;
            box-shadow: 0 8px 18px rgba(15, 76, 92, 0.18);
        }

        [data-testid="stExpander"] {
            background: #fff;
            border: 1px solid #d5e3e6;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15, 76, 92, 0.08);
            margin-bottom: 0.85rem;
        }

        [data-testid="stExpander"] summary p {
            font-family: "Fraunces", serif;
            font-size: 1.05rem;
            color: #0f4c5c;
        }

        .slot {
            background: #f7fbfb;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }

        .slot-label {
            font-weight: 600;
            color: #0f4c5c;
            margin-bottom: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DAY_FIELDS = ("day_number", "theme", "morning", "afternoon", "evening", "food_tip")


def get_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def get_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def run_query(sql: str, params: tuple | None = None, fetch: bool = False):
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your environment, then restart the app."
        )
    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            return rows
        finally:
            cur.close()
    finally:
        conn.close()


def save_trip(destination: str, days: int, style: str, itinerary: list[dict]) -> None:
    run_query(
        "INSERT INTO trips (destination, days, style, itinerary) VALUES (%s, %s, %s, %s)",
        (destination, days, style, json.dumps(itinerary)),
    )


def list_saved_trips() -> list[tuple]:
    return (
        run_query(
            "SELECT id, destination, days, style, itinerary, created_at "
            "FROM trips ORDER BY created_at DESC",
            fetch=True,
        )
        or []
    )


def delete_saved_trip(trip_id: int) -> None:
    run_query("DELETE FROM trips WHERE id = %s", (trip_id,))


def parse_saved_itinerary(raw) -> list[dict]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    raw_days = data.get("days", data) if isinstance(data, dict) else data
    if not isinstance(raw_days, list):
        return []
    itinerary = []
    for index, day in enumerate(raw_days, start=1):
        if not isinstance(day, dict):
            continue
        item = {field: str(day.get(field, "")).strip() for field in DAY_FIELDS}
        item["day_number"] = int(day.get("day_number") or index)
        if not item["theme"]:
            item["theme"] = f"Day {item['day_number']}"
        itinerary.append(item)
    return itinerary


def load_tip_lines() -> list[str]:
    if not TIPS_PATH.exists():
        raise FileNotFoundError(f"tips.txt was not found at {TIPS_PATH}")
    return [
        line.strip()
        for line in TIPS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_resource(show_spinner=False)
def get_tips_collection():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment, then restart the app."
        )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=api_key,
            api_key_env_var="OPENAI_API_KEY",
            model_name="text-embedding-3-small",
        ),
    )
    if collection.count() == 0:
        lines = load_tip_lines()
        if lines:
            collection.add(
                ids=[f"tip-{index}" for index in range(len(lines))],
                documents=lines,
            )
    return collection


@st.cache_data(show_spinner=False, ttl=3600)
def retrieve_tips(destination: str) -> list[str]:
    collection = get_tips_collection()
    count = collection.count()
    if count == 0 or not destination.strip():
        return []
    result = collection.query(
        query_texts=[destination.strip()],
        n_results=min(3, count),
    )
    documents = (result.get("documents") or [[]])[0]
    return [doc for doc in documents if doc]


async def _call_get_weather(city: str) -> str:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(WEATHER_SERVER)],
        cwd=str(APP_DIR),
    )
    async with Client(params) as client:
        result = await client.call_tool("get_weather", {"city": city})
    texts = [
        block.text
        for block in (result.content or [])
        if getattr(block, "text", None)
    ]
    return " ".join(texts).strip()


def fetch_weather(destination: str) -> str:
    city = destination.split(",")[0].strip()
    if not city:
        return ""
    return asyncio.run(_call_get_weather(city))


def generate_itinerary(
    destination: str,
    days: int,
    travel_style: str,
    tips: list[str] | None = None,
    weather: str | None = None,
) -> list[dict]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment, then restart the app."
        )

    tips_block = ""
    if tips:
        joined = "\n".join(f"- {tip}" for tip in tips)
        tips_block = f"\nUse these local tips where relevant:\n{joined}\n"

    weather_block = ""
    if weather:
        weather_block = (
            f"\nCurrent weather: {weather}\n"
            "Adapt the itinerary to this forecast: prefer indoor plans if it is "
            "raining, snowing, stormy, or very cold; otherwise include outdoor time.\n"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert travel planner. Return valid JSON only, "
                    "with no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Create a {days}-day {travel_style.lower()} itinerary for "
                    f"{destination}. Return JSON with this shape:\n"
                    '{"days":[{"day_number":1,"theme":"short theme",'
                    '"morning":"plan","afternoon":"plan","evening":"plan",'
                    '"food_tip":"local food recommendation"}]}\n'
                    f"Include exactly {days} days. Keep each field to 1 sentence "
                    "and use specific place names."
                    f"{weather_block}"
                    f"{tips_block}"
                ),
            },
        ],
        temperature=0.4,
        max_tokens=min(280 + days * 160, 1400),
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    raw_days = payload.get("days", payload)
    if not isinstance(raw_days, list):
        raise ValueError("The model did not return a list of days.")

    itinerary = []
    for index, day in enumerate(raw_days, start=1):
        if not isinstance(day, dict):
            continue
        item = {field: str(day.get(field, "")).strip() for field in DAY_FIELDS}
        item["day_number"] = int(day.get("day_number") or index)
        if not item["theme"]:
            item["theme"] = f"Day {item['day_number']}"
        itinerary.append(item)

    if not itinerary:
        raise ValueError("The model returned an empty itinerary.")
    return itinerary


def render_day_card(day: dict, expanded: bool = False) -> None:
    title = f"🗓️ Day {day['day_number']}: {day['theme']}"
    with st.expander(title, expanded=expanded):
        st.markdown(
            f"""
            <div class="slot">
                <div class="slot-label">🌅 Morning</div>
                <div>{html.escape(day['morning'])}</div>
            </div>
            <div class="slot">
                <div class="slot-label">☀️ Afternoon</div>
                <div>{html.escape(day['afternoon'])}</div>
            </div>
            <div class="slot">
                <div class="slot-label">🌙 Evening</div>
                <div>{html.escape(day['evening'])}</div>
            </div>
            <div class="slot">
                <div class="slot-label">🍜 Food tip</div>
                <div>{html.escape(day['food_tip'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_weather_badge(weather: str) -> None:
    if not weather:
        return
    st.markdown(
        f'<div class="weather-badge">🌤️ {html.escape(weather)}</div>',
        unsafe_allow_html=True,
    )


def render_tips_box(tips: list[str]) -> None:
    if not tips:
        return
    items = "".join(f"<li>{html.escape(tip)}</li>" for tip in tips)
    st.markdown(
        f"""
        <div class="tips-box">
            <h3>📍 Local insider tips</h3>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>✈️ AI Trip Planner</h1>
        <p>Design a tailored itinerary in minutes — pick a destination, trip length, and travel style.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "trip" not in st.session_state:
    st.session_state.trip = None
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "plan_error" not in st.session_state:
    st.session_state.plan_error = None
if "generate" not in st.session_state:
    st.session_state.generate = False
if "retrieved_tips" not in st.session_state:
    st.session_state.retrieved_tips = []
if "weather" not in st.session_state:
    st.session_state.weather = ""

with st.sidebar:
    st.header("Trip details")
    with st.form("trip_form", border=False):
        destination = st.text_input("Destination", placeholder="e.g. Kyoto, Japan")
        days = st.number_input("Number of days", min_value=1, max_value=14, value=5, step=1)
        travel_style = st.selectbox("Travel style", ["Budget", "Balanced", "Luxury"])
        plan_clicked = st.form_submit_button("Plan my trip", use_container_width=True)

    if plan_clicked:
        if not destination.strip():
            st.warning("Please enter a destination.")
        else:
            st.session_state.trip = {
                "destination": destination.strip(),
                "days": int(days),
                "style": travel_style,
            }
            st.session_state.itinerary = None
            st.session_state.plan_error = None
            st.session_state.retrieved_tips = []
            st.session_state.weather = ""
            st.session_state.generate = True

    st.divider()
    st.subheader("My past trips")
    try:
        past_trips = list_saved_trips()
    except Exception as exc:
        past_trips = []
        st.caption(f"Could not load saved trips: {exc}")

    if not past_trips:
        st.caption("No saved trips yet.")
    else:
        for trip_id, dest, days, style, itinerary_json, created_at in past_trips:
            date_label = created_at.strftime("%d %b %Y") if created_at else "Unknown date"
            open_col, delete_col = st.columns([3.2, 1.1])
            with open_col:
                opened = st.button(
                    f"{dest} · {date_label}",
                    key=f"open_trip_{trip_id}",
                    use_container_width=True,
                )
            with delete_col:
                deleted = st.button(
                    "Delete",
                    key=f"delete_trip_{trip_id}",
                    use_container_width=True,
                )
            if opened:
                st.session_state.trip = {
                    "id": trip_id,
                    "destination": dest,
                    "days": days,
                    "style": style,
                    "saved": True,
                    "saved_on": date_label,
                }
                st.session_state.itinerary = parse_saved_itinerary(itinerary_json)
                st.session_state.retrieved_tips = []
                st.session_state.weather = ""
                st.session_state.plan_error = None
                st.session_state.generate = False
                st.rerun()
            if deleted:
                delete_saved_trip(trip_id)
                loaded = st.session_state.trip or {}
                if loaded.get("id") == trip_id:
                    st.session_state.trip = None
                    st.session_state.itinerary = None
                st.rerun()

st.subheader("Your itinerary")

if st.session_state.generate and st.session_state.trip:
    trip = st.session_state.trip
    with st.spinner("Crafting your itinerary..."):
        try:
            st.session_state.retrieved_tips = retrieve_tips(trip["destination"])
        except Exception:
            st.session_state.retrieved_tips = []
        try:
            st.session_state.weather = fetch_weather(trip["destination"])
        except Exception:
            st.session_state.weather = ""
        try:
            st.session_state.itinerary = generate_itinerary(
                trip["destination"],
                trip["days"],
                trip["style"],
                st.session_state.retrieved_tips,
                st.session_state.weather,
            )
            try:
                save_trip(
                    trip["destination"],
                    trip["days"],
                    trip["style"],
                    st.session_state.itinerary,
                )
            except Exception as exc:
                st.session_state.plan_error = f"Trip generated, but could not save it: {exc}"
        except AuthenticationError:
            st.session_state.plan_error = (
                "The OpenAI API key was rejected. Check OPENAI_API_KEY and restart the app."
            )
        except RateLimitError:
            st.session_state.plan_error = (
                "OpenAI has no remaining credits for this API key. "
                "Add credits, then try again."
            )
        except (APIError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            st.session_state.plan_error = f"Could not plan this trip: {exc}"
        except Exception as exc:
            st.session_state.plan_error = f"Could not plan this trip: {exc}"
    st.session_state.generate = False

if st.session_state.plan_error:
    st.error(st.session_state.plan_error)

if st.session_state.itinerary:
    trip = st.session_state.trip or {}
    destination_label = trip.get("destination", "your destination")
    render_weather_badge(st.session_state.weather)
    saved_note = " · saved" if trip.get("saved") else ""
    saved_on = f" · {trip['saved_on']}" if trip.get("saved_on") else ""
    st.caption(
        f"{trip.get('days', len(st.session_state.itinerary))}-day "
        f"{str(trip.get('style', '')).lower()} plan for {destination_label}"
        f"{saved_note}{saved_on}."
    )
    for index, day in enumerate(st.session_state.itinerary):
        render_day_card(day, expanded=(index == 0))
    render_tips_box(st.session_state.retrieved_tips)
else:
    st.markdown(
        """
        <div class="placeholder-card">
            <div class="icon">🗺️</div>
            <h3>Your itinerary will appear here</h3>
            <p>Fill in your trip details in the sidebar and click Plan my trip.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_weather_badge(st.session_state.weather)
    render_tips_box(st.session_state.retrieved_tips)
