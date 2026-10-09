from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import ollama
from gtts import gTTS
import os
import tempfile
import requests as http_requests
from rag_pipeline import build_rag_index, retrieve_context

import subprocess
import imageio_ffmpeg

# Resolve the bundled ffmpeg binary once at import time
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

app = Flask(__name__)

# ─── Conversation Memory ────────────────────────────────────────────
conversation_history = []
MAX_HISTORY = 10

# ─── Configuration ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

AUDIO_PATH = os.path.join(STATIC_DIR, "response.mp3")
OLLAMA_MODEL = "llama3"

# ─── Weather Configuration ──────────────────────────────────────────
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "0afa5cdf4b70b91f6a0aea590899cfff")
OPENWEATHER_BASE    = "https://api.openweathermap.org/data/2.5"

_weather_cache = {}
WEATHER_CACHE_TTL = 1800

recognizer = sr.Recognizer()

# ─── RAG Pipeline ───────────────────────────────────────────────────
# Build BM25 index at startup (runs once; fast enough for HTTP worker)
build_rag_index()

print("✅ KisanAI ready!")

# ─── Weather Helpers ────────────────────────────────────────────────

import time

def _weather_from_response(data):
    w    = data.get("weather", [{}])[0]
    m    = data.get("main", {})
    wind = data.get("wind", {})
    rain = data.get("rain", {})
    return {
        "city"         : data.get("name", ""),
        "country"      : data.get("sys", {}).get("country", ""),
        "condition"    : w.get("main", ""),
        "description"  : w.get("description", ""),
        "temp_c"       : round(m.get("temp", 0) - 273.15, 1),
        "feels_like"   : round(m.get("feels_like", 0) - 273.15, 1),
        "humidity"     : m.get("humidity", 0),
        "wind_kmh"     : round(wind.get("speed", 0) * 3.6, 1),
        "rain_1h_mm"   : rain.get("1h", 0),
        "visibility_km": round(data.get("visibility", 0) / 1000, 1),
    }


def fetch_weather(lat=None, lon=None, city=None):
    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        return None

    cache_key = f"{lat},{lon}" if lat is not None else (city or "")
    cached = _weather_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < WEATHER_CACHE_TTL:
        print("  ☁️ Weather cache hit")
        return cached["data"]

    try:
        if lat is not None and lon is not None:
            url    = f"{OPENWEATHER_BASE}/weather"
            params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
        elif city:
            url    = f"{OPENWEATHER_BASE}/weather"
            params = {"q": city + ",IN", "appid": OPENWEATHER_API_KEY}
        else:
            return None

        resp = http_requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        normalized = _weather_from_response(resp.json())
        _weather_cache[cache_key] = {"data": normalized, "ts": time.time()}
        print(f"  ☁️ Weather: {normalized['city']} {normalized['temp_c']}°C {normalized['condition']}")
        return normalized
    except Exception as e:
        print(f"  ⚠️ Weather fetch failed: {e}")
        return None


def weather_context_str(weather):
    if not weather:
        return ""
    rain_note = f", rain {weather['rain_1h_mm']}mm/h" if weather['rain_1h_mm'] else ""
    return (
        f"[CURRENT WEATHER — {weather['city']}, {weather['country']}]: "
        f"{weather['temp_c']}°C, feels {weather['feels_like']}°C, "
        f"{weather['description']}, humidity {weather['humidity']}%, "
        f"wind {weather['wind_kmh']} km/h{rain_note}"
    )

# ─── Language Detection ─────────────────────────────────────────────

def detect_language(text):
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total      = len([c for c in text if c.strip()])

    if total == 0:
        return "en"

    if devanagari / total > 0.2:
        marathi_words = [
            "आहे", "नाही", "आणि", "करा", "माझ्या",
            "तुमच्या", "आमच्या", "पीक", "शेती", "मला",
            "कसे", "काय", "केले", "होते", "असते"
        ]
        if any(w in text for w in marathi_words):
            return "mr"
        return "hi"
    return "en"

# ─── Audio Conversion ───────────────────────────────────────────────

def convert_to_wav(input_path):
    """
    Convert any browser audio format (webm, ogg, mp4, m4a) to 16kHz mono WAV
    by calling the bundled imageio_ffmpeg binary directly via subprocess.

    WHY SUBPROCESS INSTEAD OF PYDUB:
    pydub internally calls `ffprobe` to detect audio format metadata. When
    ffprobe is not on the Windows PATH (even if ffmpeg.exe is available),
    pydub raises WinError 2. By calling ffmpeg directly we skip ffprobe
    entirely — ffmpeg can decode webm/ogg/mp4 without needing ffprobe.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.wav':
        return input_path, False  # already WAV

    print(f"  🔄 Converting {ext} → WAV via ffmpeg subprocess...")
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()  # close so ffmpeg can write to it on Windows

    cmd = [
        _FFMPEG_EXE,
        "-y",               # overwrite output without asking
        "-i", input_path,   # input file
        "-ar", "16000",     # 16 kHz sample rate (best for STT)
        "-ac", "1",         # mono channel
        "-f", "wav",        # force WAV output format
        tmp_wav.name
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace')[-500:]
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}: {err}")

        size_kb = os.path.getsize(tmp_wav.name) / 1024
        print(f"  ✅ Converted → {tmp_wav.name} ({size_kb:.1f} KB)")
        return tmp_wav.name, True  # caller must delete

    except Exception as e:
        # Clean up the temp file if something went wrong
        try:
            os.unlink(tmp_wav.name)
        except Exception:
            pass
        raise RuntimeError(f"Audio conversion failed ({ext} → WAV): {e}")

# ─── Core Functions ─────────────────────────────────────────────────

def speech_to_text(audio_file, hint_lang=None):
    """
    Transcribe browser audio using Google Speech Recognition.

    Flow:
    1. Convert audio to WAV (raises RuntimeError on failure — never silent)
    2. Check duration — reject clips under 0.5s (empty recording guard)
    3. If hint_lang set → use only that locale (most reliable)
    4. Auto-detect: en-IN → hi-IN → mr-IN with script validation
    """
    print(f"👂 Transcribing: {audio_file}")

    # Step 1: Convert — raises RuntimeError if ffmpeg/pydub unavailable or file corrupt
    wav_path, should_delete = convert_to_wav(audio_file)

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
            duration_s = source.DURATION
            print(f"  🎵 Audio duration: {duration_s:.1f}s")
            # Step 2: Duration guard — catches empty MediaRecorder blobs
            if duration_s < 0.5:
                raise RuntimeError(
                    f"Audio too short ({duration_s:.1f}s) — no speech captured. "
                    "Please check microphone permissions and speak again."
                )
    finally:
        if should_delete:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    lang_map = {"hi": "hi-IN", "mr": "mr-IN", "en": "en-IN"}

    # ── Case 1: Farmer manually selected a language ──────────────────
    if hint_lang and hint_lang in lang_map:
        google_code = lang_map[hint_lang]
        try:
            text = recognizer.recognize_google(audio_data, language=google_code)
            print(f"  ✅ Manual lang {google_code} → {text}")
            return text.strip(), hint_lang
        except sr.UnknownValueError:
            print(f"  ❌ Could not understand audio in {google_code}")
            return "", hint_lang
        except sr.RequestError as e:
            print(f"  ❌ API error: {e}")
            return "", hint_lang

    # ── Case 2: Auto-detect ───────────────────────────────────────────
    en_text = ""
    try:
        en_text = recognizer.recognize_google(audio_data, language="en-IN")
        print(f"  🔍 en-IN result → {en_text}")
    except sr.UnknownValueError:
        print("  ❌ en-IN → could not understand")
    except sr.RequestError as e:
        print(f"  ❌ en-IN API error: {e}")

    if en_text:
        char_lang = detect_language(en_text)
        if char_lang == "en":
            print("  ✅ Confirmed English (Latin script dominant)")
            return en_text.strip(), "en"

    hi_text = ""
    try:
        hi_text = recognizer.recognize_google(audio_data, language="hi-IN")
        print(f"  🔍 hi-IN result → {hi_text}")
    except sr.UnknownValueError:
        print("  ❌ hi-IN → could not understand")
    except sr.RequestError as e:
        print(f"  ❌ hi-IN API error: {e}")

    if hi_text:
        char_lang = detect_language(hi_text)
        if char_lang == "hi":
            print("  ✅ Confirmed Hindi (Devanagari dominant, no Marathi words)")
            return hi_text.strip(), "hi"

    mr_text = ""
    try:
        mr_text = recognizer.recognize_google(audio_data, language="mr-IN")
        print(f"  🔍 mr-IN result → {mr_text}")
    except sr.UnknownValueError:
        print("  ❌ mr-IN → could not understand")
    except sr.RequestError as e:
        print(f"  ❌ mr-IN API error: {e}")

    if mr_text:
        char_lang = detect_language(mr_text)
        if char_lang == "mr":
            print("  ✅ Confirmed Marathi")
            return mr_text.strip(), "mr"

    for text, lang in [(en_text, "en"), (hi_text, "hi"), (mr_text, "mr")]:
        if text:
            print(f"  ⚠️ Fallback to {lang}")
            return text.strip(), lang

    print("  ❌ No transcription result at all")
    return "", "en"


def get_llm_response(query, lang="en", weather=None):
    global conversation_history
    print(f"🧠 Asking Ollama ({OLLAMA_MODEL}) in lang={lang}: {query}")

    lang_name = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "mr": "Marathi (मराठी)"
    }.get(lang, "English")

    # ── RAG Context ───────────────────────────────────────────────────
    rag_context = retrieve_context(query, top_k=5)
    if rag_context:
        print(f"  📚 RAG retrieved {rag_context.count(chr(10))} context lines")
        rag_block = f"""
{rag_context}
INSTRUCTIONS for using the above knowledge base:
- Treat retrieved facts as ground truth for your answer.
- If a retrieved fact directly answers the query, use it.
- Do NOT contradict the retrieved facts.
- Do NOT mention "knowledge base" or "retrieved data" to the farmer — just answer naturally.
"""
    else:
        print("  📚 RAG: No relevant context found — answering from LLM knowledge")
        rag_block = ""

    weather_block = ""
    if weather:
        weather_block = f"""
WEATHER DATA (already fetched — do NOT ask the farmer for location):
{weather_context_str(weather)}
RULES for using weather:
- You already have the farmer's weather. NEVER ask them to share location or city.
- Proactively reference these conditions in your answer when relevant.
- E.g. high humidity → warn about fungal disease; rain today → skip irrigation; hot → irrigate early morning.
"""
    else:
        weather_block = """
WEATHER: Not available for this session.
- Do NOT ask the farmer for their location or city name.
- Give general advice without weather references.
"""

    system_prompt = f"""You are KisanAI, an agriculture expert helping farmers in India.
You remember everything said in this conversation and refer back to it when relevant.
{rag_block}
{weather_block}
STRICT LANGUAGE RULE:
- The farmer is speaking in {lang_name}
- You MUST reply ONLY in {lang_name}
- Do NOT use any other language at all
- Do NOT mix languages
- Your ENTIRE response must be in {lang_name} only

FORMAT RULE:
- Use "- " (hyphen space) for bullet points, NOT the bullet character
- Use **bold** for important words
- Keep response concise and practical
- Maximum 200 words"""

    conversation_history.append({"role": "user", "content": query})

    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history = conversation_history[-(MAX_HISTORY * 2):]
        print(f"  ✂️ Trimmed history to {len(conversation_history)} messages")

    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    print(f"  📚 Sending {len(conversation_history)} history messages to Ollama")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "num_gpu"    : 0,
            "temperature": 0.3,
            "top_p"      : 0.9,
        }
    )
    answer = response["message"]["content"]

    conversation_history.append({"role": "assistant", "content": answer})
    print(f"  💾 History now has {len(conversation_history)} messages")
    print(f"✅ Ollama answer: {answer[:100]}...")
    return answer


def text_to_speech(text, lang="en"):
    print(f"🔊 Converting to speech in lang: {lang}")

    import re
    clean_text = re.sub(r'\*+', '', text)
    clean_text = re.sub(r'#+\s', '', clean_text)
    clean_text = re.sub(r'\n+', '. ', clean_text)

    lang_map = {"en": "en", "hi": "hi", "mr": "mr"}
    tts_lang = lang_map.get(lang, "en")
    tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
    tts.save(AUDIO_PATH)
    print(f"✅ MP3 saved: {AUDIO_PATH}")


# ─── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    """Mobile-optimized UI for the Android WebView app."""
    return render_template("mobile.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts a multipart audio blob from the browser's MediaRecorder API.
    hint_lang comes from the FormData field (set by the language selector in the UI).

    Three guards before STT:
      1. No audio file in request      → 400 immediately
      2. File too small (< 1 KB)       → 400 with clear message
      3. convert_to_wav / duration     → 400 via RuntimeError from speech_to_text
    """
    try:
        # Extract hint_lang from FormData or JSON
        if request.content_type and 'application/json' in request.content_type:
            hint_lang = request.json.get("hint_lang") if request.is_json else None
        else:
            hint_lang = request.form.get("hint_lang")

        if not hint_lang:
            try:
                hint_lang = request.json.get("hint_lang") if request.is_json else None
            except Exception:
                hint_lang = None

        print(f"📨 /transcribe called | hint_lang={hint_lang} | files={list(request.files.keys())}")

        if "audio" not in request.files:
            print("  ⚠️ No audio file in request")
            return jsonify({"status": "error", "message": "No audio file received. Please send audio as a file upload."}), 400

        audio_file = request.files["audio"]
        suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".webm"
        if not suffix:
            suffix = ".webm"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        audio_file.save(tmp.name)
        tmp.close()

        file_size = os.path.getsize(tmp.name)
        print(f"  💾 Saved upload to {tmp.name} ({suffix}, {file_size} bytes)")

        # Guard: reject empty/corrupt uploads before attempting decode
        if file_size < 1000:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            return jsonify({
                "status" : "error",
                "message": (
                    f"Audio file too small ({file_size} bytes) — recording may be empty. "
                    "Please check microphone permissions and try again."
                )
            }), 400

        try:
            text, lang = speech_to_text(tmp.name, hint_lang)
        except RuntimeError as e:
            # Conversion failure or audio-too-short — surface clearly to the browser
            print(f"  ❌ speech_to_text raised: {e}")
            return jsonify({"status": "error", "message": str(e)}), 400
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        return jsonify({"status": "ok", "text": text, "language": lang})

    except Exception as e:
        print(f"❌ Transcribe error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/geolocate", methods=["GET"])
def geolocate():
    """
    Server-side IP geolocation — avoids browser mixed-content/CORS issues.
    Uses the requester's IP to call ip-api.com from the server.
    Falls back to ipinfo.io if ip-api fails.
    """
    try:
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or request.remote_addr
        )
        print(f"🌍 Geolocating IP: {client_ip}")

        try:
            r = http_requests.get(
                f"http://ip-api.com/json/{client_ip}",
                params={"fields": "status,lat,lon,city,regionName,country"},
                timeout=4
            )
            d = r.json()
            if d.get("status") == "success" and d.get("lat"):
                print(f"  ✅ ip-api: {d['city']}, {d['regionName']} ({d['lat']}, {d['lon']})")
                return jsonify({
                    "status": "ok",
                    "lat": d["lat"], "lon": d["lon"],
                    "city": d["city"], "region": d["regionName"], "country": d["country"]
                })
        except Exception as e:
            print(f"  ⚠️ ip-api failed: {e}")

        try:
            r = http_requests.get(f"https://ipinfo.io/{client_ip}/json", timeout=4)
            d = r.json()
            if "loc" in d:
                lat_s, lon_s = d["loc"].split(",")
                print(f"  ✅ ipinfo: {d.get('city')} ({lat_s}, {lon_s})")
                return jsonify({
                    "status": "ok",
                    "lat": float(lat_s), "lon": float(lon_s),
                    "city": d.get("city", ""), "region": d.get("region", ""), "country": d.get("country", "")
                })
        except Exception as e:
            print(f"  ⚠️ ipinfo failed: {e}")

        return jsonify({"status": "error", "message": "Could not determine location"}), 503

    except Exception as e:
        print(f"❌ Geolocate error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data  = request.json
        query = data.get("query", "")
        lang  = data.get("language", "en")
        lat   = data.get("lat")
        lon   = data.get("lon")
        city  = data.get("city")

        if not query:
            return jsonify({"status": "error", "message": "Empty query"}), 400

        weather = None
        if lat is not None and lon is not None:
            weather = fetch_weather(lat=lat, lon=lon)
        elif city:
            weather = fetch_weather(city=city)

        answer = get_llm_response(query, lang=lang, weather=weather)
        text_to_speech(answer, lang=lang)

        return jsonify({
            "status"   : "ok",
            "answer"   : answer,
            "audio_url": "/static/response.mp3",
            "language" : lang,
            "weather"  : weather
        })
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/weather", methods=["GET", "POST"])
def weather_route():
    try:
        if request.method == "POST":
            body = request.json or {}
            lat  = body.get("lat")
            lon  = body.get("lon")
            city = body.get("city")
        else:
            lat  = request.args.get("lat", type=float)
            lon  = request.args.get("lon", type=float)
            city = request.args.get("city")

        data = fetch_weather(lat=lat, lon=lon, city=city)
        if data is None:
            return jsonify({"status": "error", "message": "Weather unavailable. Check API key or location."}), 503
        return jsonify({"status": "ok", "weather": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/clear_history", methods=["POST"])
def clear_history():
    global conversation_history
    conversation_history = []
    print("🗑️ Conversation history cleared")
    return jsonify({"status": "ok", "message": "History cleared"})


# ─── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')