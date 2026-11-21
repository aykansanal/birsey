"""
Hafta 8: AI-Driven Development & Vibe Coding'e Giriş
Çalıştırma:
    pip install streamlit google-genai
    python -m streamlit run hafta8_app.py
"""

import base64
import os
from pathlib import Path
import streamlit as st

try:
    from google import genai
    from google.genai import types
    GENAI_OK = True
except ImportError:
    GENAI_OK = False



def render_html(html: str, height: int = 500):
    """Yapay zekanın ürettiği HTML'i izole bir çerçevede gösterir (her Streamlit sürümünde çalışır)."""
    try:
        if hasattr(st, "iframe"):
            src = "data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii")
            st.iframe(src, height=height)
            return
    except Exception:
        pass
    import streamlit.components.v1 as components
    components.html(html, height=height, scrolling=True)


KEY_FILE = Path(__file__).with_name(".gemini_key")


def saved_key() -> str:
    """Anahtar sırası: ortam değişkeni → .gemini_key dosyası → st.secrets."""
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:
        if KEY_FILE.exists():
            return KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


st.set_page_config(page_title="Hafta 8 - AI-Driven Development", page_icon="🚀", layout="wide")

# ----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ----------------------------------------------------------------------------
import re
import time

TIMEOUT_MS = 25_000  # bir model 25 sn içinde cevap vermezse sıradakine geç
SKIP_WORDS = ("image", "tts", "embed", "live", "audio", "robotics", "computer-use",
              "aqa", "imagen", "veo", "learnlm", "gemma", "native")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str, timeout_ms: int = TIMEOUT_MS):
    # cache_resource istemciyi canlı tutar; geçici bırakılırsa Python onu silip
    # bağlantıyı kapatır ("client has been closed" hatası).
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))


def _rank(name: str):
    """Küçük sayı = önce denenir: kararlı > preview, flash > lite > pro, yeni sürüm > eski."""
    n = name.lower()
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    version = float(m.group(1)) if m else 0.0
    unstable = any(w in n for w in ("preview", "exp"))
    tier = 2 if "pro" in n else (1 if "lite" in n else 0)
    return (unstable, tier, -version, n)


@st.cache_data(ttl=600, show_spinner=False)
def list_models(api_key: str):
    """Bu API Key ile gerçekten kullanılabilen metin modellerini Google'dan çeker."""
    names = []
    client = get_client(api_key, 15_000)
    for m in client.models.list():
        name = (m.name or "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if ("generateContent" in actions and name.startswith("gemini")
                and not any(w in name.lower() for w in SKIP_WORDS)):
            names.append(name)
    return sorted(set(names), key=_rank)


def _candidates(api_key: str, preferred: str):
    chain = [preferred]
    chain += [m for m in st.session_state.get("working_models", []) if m not in chain]
    try:
        chain += [n for n in list_models(api_key) if n not in chain]
    except Exception:
        pass
    return chain[:6]


def _short_err(msg: str) -> str:
    m = re.search(r"\b(\d{3}) ([A-Z_]{4,})", msg)
    return f"{m.group(1)} {m.group(2)}" if m else msg[:60]


def scan_models(api_key: str, models):
    """Her modele küçük bir istek atıp çalışanları bulur (paralel). Hızlı olan önce gelir."""
    from concurrent.futures import ThreadPoolExecutor
    client = get_client(api_key, 20_000)

    def probe(model):
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=model, contents="Sadece 'tamam' yaz.",
                config=types.GenerateContentConfig(temperature=0.0),
            )
            ok = bool(r.text)
            return {"model": model, "ok": ok, "sec": time.time() - t0, "err": "" if ok else "boş cevap"}
        except Exception as e:
            return {"model": model, "ok": False, "sec": time.time() - t0, "err": _short_err(str(e))}

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(probe, models))
    return sorted(res, key=lambda r: (not r["ok"], r["sec"]))


def generate(api_key, preferred, system_prompt, contents, temperature):
    """Seçili modeli dener; 404 / 503 / zaman aşımında sıradaki modele geçer.
    (metin, kullanılan_model) döndürür."""
    errors = []
    client = get_client(api_key)
    for model in _candidates(api_key, preferred):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt or None,
                    temperature=temperature,
                ),
            )
            if resp.text:
                return resp.text, model
            errors.append(f"{model}: boş cevap")
        except Exception as e:
            msg = str(e)
            errors.append(f"{model}: {msg[:180]}")
            if any(c in msg for c in ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                break  # anahtar sorunu, başka model denemenin anlamı yok
    raise RuntimeError("Hiçbir model cevap vermedi:\n" + "\n".join(errors))


def _notify(requested, used):
    if used != requested:
        st.toast(f"{requested} cevap vermedi, {used} kullanıldı", icon="🔁")


def ask_gemini(api_key, model, system_prompt, user_text, temperature=0.4):
    """Tek seferlik metin üretimi."""
    text, used = generate(api_key, model, system_prompt, user_text, temperature)
    _notify(model, used)
    return text


def chat_gemini(api_key, model, system_prompt, history, temperature=0.7):
    """Geçmişi de gönderen chatbot çağrısı."""
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in history
    ]
    text, used = generate(api_key, model, system_prompt, contents, temperature)
    _notify(model, used)
    return text


import ast


def python_check(code: str):
    """ARAÇ: Python'un kendi ayrıştırıcısıyla sözdizimini kontrol eder (kesin sonuç, tahmin değil)."""
    try:
        ast.parse(code)
        return True, "Sözdizimi hatası yok."
    except SyntaxError as e:
        line = (e.text or "").strip()
        return False, f"SyntaxError — satır {e.lineno}: {e.msg}" + (f"  →  `{line}`" if line else "")


def first_code_block(text: str):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def run_agent(api_key, model, mode, input_type, lang, user_input, system, temperature, review=True):
    """Basit agent: araçla kontrol et → model cevaplasın → cevaptaki kodu araçla doğrula,
    hatalıysa modele geri bildir (en fazla 2 kez) → ikinci bir Gemini çağrısıyla eleştir.
    (cevap, adımlar) döndürür."""
    steps = []
    is_py = input_type == "Kod" and lang == "Python"
    prefix = f"[Girdi türü: {input_type}" + (f", dil: {lang}" if lang else "") + "]\n\n"
    temp = min(temperature, 0.2)  # daha tutarlı cevap için düşük yaratıcılık

    tool_note = ""
    if is_py:
        ok, msg = python_check(user_input)
        steps.append(f"🔧 **Araç (Python ayrıştırıcı)** girdiyi kontrol etti → {'✅' if ok else '❌'} {msg}")
        tool_note = f"\n\n[ARAÇ SONUCU — kesin doğrudur, listene mutlaka ekle]: {msg}"

    steps.append("🧠 **Gemini** ilk cevabı üretiyor...")
    answer = ask_gemini(api_key, model, system, prefix + user_input + tool_note, temp)

    if is_py and mode.endswith("Hataları düzelt"):
        for attempt in range(1, 3):
            code = first_code_block(answer)
            if not code:
                steps.append("⚠️ Cevapta kod bloğu bulunamadı, doğrulama atlandı.")
                break
            ok, msg = python_check(code)
            steps.append(f"🔧 **Araç** düzeltilmiş kodu doğruladı → {'✅' if ok else '❌'} {msg}")
            if ok:
                break
            steps.append(f"🔁 Hata modele geri bildirildi (deneme {attempt}/2)")
            answer = ask_gemini(
                api_key, model, system,
                prefix + user_input + tool_note +
                f"\n\n[ÖNCEKİ CEVABIN]:\n{answer}\n\n[ARAÇ SONUCU]: Düzelttiğin kodda hâlâ hata var: {msg}\n"
                "Cevabı aynı formatta, bu hatayı da düzelterek baştan yaz.",
                temp,
            )

    if review:
        steps.append("🕵️ **İkinci Gemini (eleştirmen)** cevabı orijinal girdiyle karşılaştırıyor...")
        reviewer_sys = (
            "Sen titiz bir kod/metin denetçisisin. Sana orijinal girdi, (varsa) araç sonucu ve bir taslak cevap verilecek. "
            "Taslakta: (1) kaçırılan gerçek hata/madde, (2) yanlış alarm veya çelişki, (3) bozuk Türkçe var mı kontrol et. "
            "Sorun varsa AYNI formatta, düzeltilmiş nihai cevabın TAMAMINI yaz. Sorun yoksa taslağı aynen döndür. "
            "Cevabın başına denetim notu ekleme, sadece nihai cevabı yaz."
        )
        try:
            reviewed = ask_gemini(
                api_key, model, reviewer_sys,
                f"ORİJİNAL İSTEK TALİMATI:\n{system}\n\nORİJİNAL GİRDİ:\n{prefix}{user_input}{tool_note}\n\nTASLAK CEVAP:\n{answer}",
                0.1,
            )
            if is_py and mode.endswith("Hataları düzelt"):
                code = first_code_block(reviewed)
                ok, msg = python_check(code) if code else (False, "kod bloğu yok")
                steps.append(f"🔧 **Araç** eleştirmenin kodunu doğruladı → {'✅' if ok else '❌'} {msg}")
                if ok:
                    answer = reviewed
                else:
                    steps.append("↩️ Eleştirmenin kodu hatalı çıktı, önceki cevap korundu.")
            else:
                answer = reviewed
        except Exception as e:
            steps.append(f"⚠️ Eleştirmen çalışamadı ({str(e)[:80]}), ilk cevap korundu.")

    steps.append("✅ Tamamlandı.")
    return answer, steps


def need_key():
    if not GENAI_OK:
        st.error("`google-genai` kurulu değil. Terminalde: `pip install google-genai`")
        return True
    if not st.session_state.get("api_key"):
        st.warning("Sol menüden Gemini API Key'inizi girin. (Nasıl alınır? → '🛠️ Kurulum' sekmesi)")
        return True
    return False


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Ayarlar")
    key_in = st.text_input(
        "Gemini API Key",
        type="password",
        value=saved_key(),
        help="https://aistudio.google.com/apikey adresinden ücretsiz alabilirsiniz.",
    )
    st.session_state["api_key"] = key_in
    # Kaydetme sadece kendi bilgisayarında çalışır; yayında (Streamlit Cloud) ziyaretçiler
    # birbirinin anahtarını görmesin diye kapalıdır.
    on_cloud = Path("/mount/src").exists()
    if key_in and key_in != saved_key() and not on_cloud:
        if st.button("💾 Anahtarı bu bilgisayara kaydet"):
            KEY_FILE.write_text(key_in.strip(), encoding="utf-8")
            st.success("Kaydedildi. Bir sonraki açılışta otomatik gelir.")
            st.rerun()
    elif not key_in:
        st.info("Önce API Key'i gir. Modeller anahtardan sonra otomatik taranır.")
    available = []
    if GENAI_OK and key_in:
        try:
            available = list_models(key_in)
        except Exception as e:
            st.warning(f"Model listesi alınamadı: {str(e)[:150]}")
    if GENAI_OK and key_in and available and st.session_state.get("scanned_key") != key_in:
        with st.spinner("Çalışan modeller otomatik taranıyor (~25 sn)..."):
            res = scan_models(key_in, available)
        st.session_state["scan"] = res
        st.session_state["working_models"] = [r["model"] for r in res if r["ok"]]
        st.session_state["scanned_key"] = key_in
    working = st.session_state.get("working_models", [])
    options = (working + [m for m in available if m not in working]) \
        or ["gemini-flash-latest", "gemini-flash-lite-latest"]
    model_name = st.selectbox(
        "Model", options, index=0,
        help="Liste, API Key'inle gerçekten kullanabildiğin modellerden otomatik çekilir. "
             "Seçili model cevap vermezse sıradaki çalışan modele otomatik geçilir.",
    )
    if available:
        st.caption(f"✅ {len(available)} kullanılabilir model bulundu")

    if st.button("🔌 Bağlantıyı test et"):
        if not need_key():
            t0 = time.time()
            try:
                txt, used = generate(key_in, model_name, "", "Sadece 'tamam' yaz.", 0.0)
                st.success(f"{used} · {time.time() - t0:.1f} sn · cevap: {txt.strip()[:40]}")
            except Exception as e:
                st.error(str(e))
    if st.button("🔍 Tüm modelleri tara (çalışanı bul)"):
        if not need_key():
            with st.spinner(f"{len(available) or 1} model test ediliyor (en fazla ~25 sn)..."):
                res = scan_models(key_in, available or [model_name])
            st.session_state["scan"] = res
            st.session_state["working_models"] = [r["model"] for r in res if r["ok"]]
            st.session_state["scanned_key"] = key_in
            st.rerun()
    if st.session_state.get("scan"):
        scan = st.session_state["scan"]
        n_ok = sum(r["ok"] for r in scan)
        (st.success if n_ok else st.error)(
            f"{n_ok}/{len(scan)} model çalışıyor" + (" — en hızlısı otomatik seçildi" if n_ok else "")
        )
        with st.expander("Model tarama sonuçları"):
            st.dataframe(
                [{"Model": r["model"], "Durum": "✅" if r["ok"] else "❌ " + r["err"],
                  "Süre (sn)": round(r["sec"], 1)} for r in scan],
                hide_index=True, use_container_width=True,
            )

    temperature = 0.3  # sabit: tutarlı ve dengeli cevaplar

# ----------------------------------------------------------------------------
# Başlık
# ----------------------------------------------------------------------------
st.title("🚀 Hafta 8: AI-Driven Development & Vibe Coding'e Giriş")

tabs = st.tabs([
    "🚀 Proje 1: Akıllı Kod Asistanı",
    "💬 Chatbot & System Prompt",
    "🧠 LLM & Prompt Engineering",
    "🛠️ Kurulum",
    "🎧 Vibe Coding",
    "💻 Uygulama: Doğal Dille Kod",
])

# ----------------------------------------------------------------------------
# 2) Proje 1: Akıllı Kod Asistanı / Konu Özetleyici
# ----------------------------------------------------------------------------
with tabs[0]:
    st.header("🚀 PROJE 1: Akıllı Kod Asistanı / Konu Özetleyici")
    st.caption("Metin veya kod gir → özetle, açıkla ya da hatalarını düzelt.")

    mode = st.radio(
        "Ne yapmak istiyorsun?",
        ["📝 Özetle", "📖 Açıklama yaz", "🐞 Hataları düzelt"],
        horizontal=True,
    )
    input_type = st.radio("Girdi türü", ["Metin", "Kod"], horizontal=True)
    lang = st.selectbox("Kod dili (kod ise)", ["Python", "C#", "JavaScript", "Java", "C++", "SQL", "Diğer"]) \
        if input_type == "Kod" else None

    sample = {
        "Metin": "Büyük dil modelleri (LLM), çok büyük metin veri kümeleri üzerinde eğitilen ve bir "
                 "sonraki kelimeyi tahmin ederek metin üreten yapay zeka modelleridir. Transformer "
                 "mimarisi sayesinde bağlamı uzun mesafelerde takip edebilirler.",
        "Kod": "def ortalama(liste):\n    toplam = 0\n    for i in range(1, len(liste)):\n"
               "        toplam += liste[i]\n    return toplam / len(liste\n",
    }[input_type]
    user_input = st.text_area("Girdi", value=sample, height=220)

    p1_system = {
        "📝 Özetle": "Sen bir özetleme asistanısın. Verilen metni/kodu Türkçe, açık ve kısa özetle. "
                     "Önce 1-2 cümlelik genel özet, sonra maddeler halinde ana noktaları ver.",
        "📖 Açıklama yaz": "Sen bir yazılım eğitmenisin. Verilen metni/kodu bir öğrenciye anlatır gibi Türkçe, "
                           "adım adım açıkla. Kod ise satır satır ne yaptığını ve neden yaptığını anlat.",
        "🐞 Hataları düzelt": "Sen deneyimli bir kod inceleyicisin. Verilen koddaki sözdizimi ve mantık hatalarını bul. "
                              "Çıktı formatı: 1) Bulunan hatalar (madde madde), 2) Düzeltilmiş kodun tamamı "
                              "(kod bloğu içinde), 3) Ne değiştirdiğinin kısa açıklaması. Türkçe yaz.",
    }[mode]

    with st.expander("🔧 Bu modun System Prompt'u (düzenleyebilirsin)"):
        p1_system = st.text_area("System Prompt", value=p1_system, height=140, key=f"p1_sys_{mode}")

    agent_on = st.checkbox(
        "🤖 Agent modu (araçla doğrula + ikinci Gemini ile kontrol et)", value=True,
        help="Tek seferlik cevap yerine: Python ayrıştırıcısıyla kodu kontrol eder, cevaptaki kodu doğrular, "
             "hatalıysa modele geri gönderir ve ikinci bir Gemini çağrısıyla cevabı denetler. "
             "Daha güvenilir ama 2-3 kat daha yavaştır.",
    )

    if st.button("Çalıştır", type="primary", key="p1_run"):
        if not user_input.strip():
            st.warning("Lütfen bir girdi yaz.")
        elif not need_key():
            prefix = f"[Girdi türü: {input_type}" + (f", dil: {lang}" if lang else "") + "]\n\n"
            with st.spinner("Agent çalışıyor..." if agent_on else "Gemini düşünüyor..."):
                try:
                    if agent_on:
                        out, steps = run_agent(st.session_state["api_key"], model_name, mode, input_type,
                                               lang, user_input, p1_system, temperature)
                        with st.expander("🤖 Agent adımları (ne yaptı?)", expanded=True):
                            for step in steps:
                                st.markdown("- " + step)
                    else:
                        out = ask_gemini(st.session_state["api_key"], model_name, p1_system,
                                         prefix + user_input, temperature)
                    st.markdown("### ✅ Sonuç")
                    st.markdown(out)
                except Exception as e:
                    st.error(f"Hata: {e}")

# ----------------------------------------------------------------------------
# 3) Chatbot & System Prompt
# ----------------------------------------------------------------------------
with tabs[1]:
    st.header("💬 Metin Üretimi, Chatbot Mantığı ve System Prompt")

    with st.expander("📘 Kavramlar", expanded=True):
        st.markdown(
            "- **Metin üretimi:** Model, verilen girdiye (prompt) uygun devam metnini üretir.\n"
            "- **Chatbot mantığı:** Model *hafızasızdır*. Her istekte önceki mesajların tamamı "
            "(`user` / `model` rolleriyle) tekrar gönderilir. Sohbet geçmişi bizim uygulamamızda tutulur.\n"
            "- **System Prompt:** Modelin rolünü, tonunu ve kurallarını belirleyen, kullanıcıya görünmeyen "
            "üst düzey talimattır. Aynı soruya farklı System Prompt ile çok farklı cevaplar alınır."
        )
        st.code(
            "history = [\n"
            "  {'role': 'user',  'text': 'Merhaba'},\n"
            "  {'role': 'model', 'text': 'Merhaba! Nasıl yardımcı olabilirim?'},\n"
            "  {'role': 'user',  'text': 'Python nedir?'},   # yeni mesaj\n"
            "]\n"
            "# Her çağrıda: system_instruction + TÜM history gönderilir",
            language="python",
        )

    personas = {
        "Yardımsever asistan": "Sen yardımsever bir Türkçe asistansın. Kısa ve net cevap ver.",
        "Python hocası": "Sen sabırlı bir Python hocasısın. Her cevapta küçük bir kod örneği ver, "
                         "kavramları basit anlat. Sadece programlama sorularını yanıtla.",
        "Korsan 🏴‍☠️": "Sen bir korsansın. Her cevabı korsan ağzıyla, eğlenceli ver ama bilgi doğru olsun.",
        "Özel (kendin yaz)": "",
    }
    persona = st.selectbox("Hazır System Prompt", list(personas.keys()))
    chat_system = st.text_area(
        "System Prompt",
        value=personas[persona] or "Buraya kendi system prompt'unu yaz...",
        height=100,
        key=f"chat_sys_{persona}",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑️ Sohbeti temizle"):
        st.session_state.chat_history = []
        st.rerun()

    for m in st.session_state.chat_history:
        with st.chat_message("assistant" if m["role"] == "model" else "user"):
            st.markdown(m["text"])

    if msg := st.chat_input("Mesajını yaz..."):
        st.session_state.chat_history.append({"role": "user", "text": msg})
        with st.chat_message("user"):
            st.markdown(msg)
        if not need_key():
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    try:
                        reply = chat_gemini(st.session_state["api_key"], model_name, chat_system,
                                            st.session_state.chat_history, max(temperature, 0.5))
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "model", "text": reply})
                    except Exception as e:
                        st.error(f"Hata: {e}")

    with st.expander("🔍 Modele giden geçmişi göster"):
        st.json(st.session_state.chat_history)

# ----------------------------------------------------------------------------
# 4) LLM & Prompt Engineering
# ----------------------------------------------------------------------------
with tabs[2]:
    st.header("🧠 LLM Mantığı ve Prompt Engineering İlkeleri")

    st.subheader("LLM nasıl çalışır?")
    st.markdown(
        "1. **Tokenization:** Metin küçük parçalara (token) bölünür.\n"
        "2. **Transformer:** Dikkat (attention) mekanizmasıyla tokenlar arası ilişki öğrenilir.\n"
        "3. **Sonraki token tahmini:** Model her adımda olası sonraki token'ın olasılığını hesaplar ve birini seçer.\n"
        "4. **Temperature:** Seçimdeki rastgeleliği belirler (düşük = tutarlı, yüksek = yaratıcı).\n"
        "5. **Context window:** Modelin bir seferde 'görebildiği' maksimum token sayısıdır.\n\n"
        "⚠️ LLM'ler bilgi *bilmez*, olası metni *üretir*. Bu yüzden **halüsinasyon** olabilir; çıktıyı test edin."
    )

    st.subheader("Prompt Engineering İlkeleri")
    principles = {
        "1. Net ol": "❌ `Bir kod yaz.`  →  ✅ `Python ile bir sayının asal olup olmadığını kontrol eden fonksiyon yaz.`",
        "2. Rol ver": "`Sen kıdemli bir Python geliştiricisisin...`",
        "3. Bağlam ver": "Dil, kütüphane, sürüm, hedef kitle, kısıtlar: `Python 3.12, sadece standart kütüphane.`",
        "4. Çıktı formatını belirt": "`Cevabı JSON olarak ver` / `Madde madde, en fazla 5 madde` / `Sadece kod bloğu`",
        "5. Örnek ver (few-shot)": "`Girdi: 'merhaba' → Çıktı: 'MERHABA'. Girdi: 'dünya' → Çıktı: ?`",
        "6. Adım adım düşündür": "`Önce planı yaz, sonra kodu yaz.`",
        "7. Yinele ve iyileştir": "İlk cevap kusurluysa hatayı göster ve düzeltmesini iste.",
    }
    for k, v in principles.items():
        with st.expander(k):
            st.markdown(v)

    st.subheader("🧪 Kötü Prompt vs İyi Prompt (canlı karşılaştırma)")
    col1, col2 = st.columns(2)
    bad = col1.text_area("❌ Kötü prompt", "Bana bir hesap makinesi yap", height=100)
    good = col2.text_area(
        "✅ İyi prompt",
        "Sen deneyimli bir Python eğitmenisin. Streamlit ile toplama, çıkarma, çarpma, bölme yapan "
        "basit bir hesap makinesi yaz. Sıfıra bölmeyi kontrol et. Tek dosya olsun, yorum satırları Türkçe olsun. "
        "Sadece kod bloğu döndür.",
        height=100,
    )
    if st.button("İkisini de çalıştır", key="cmp"):
        if not need_key():
            c1, c2 = st.columns(2)
            with st.spinner("Çalışıyor..."):
                try:
                    c1.markdown(ask_gemini(st.session_state["api_key"], model_name, "", bad, temperature))
                    c2.markdown(ask_gemini(st.session_state["api_key"], model_name, "", good, temperature))
                except Exception as e:
                    st.error(f"Hata: {e}")

# ----------------------------------------------------------------------------
# 5) Kurulum
# ----------------------------------------------------------------------------
with tabs[3]:
    st.header("🛠️ Kurulum Rehberi")
    k1, k2 = st.tabs(["🔑 Gemini API Key", "🧰 Cursor / VS Code Copilot"])

    with k1:
        st.markdown(
            "### Gemini API Key alma\n"
            "1. [aistudio.google.com/apikey](https://aistudio.google.com/apikey) adresine git.\n"
            "2. Google hesabınla giriş yap.\n"
            "3. **Create API key** butonuna tıkla.\n"
            "4. Oluşan anahtarı kopyala ve **kimseyle paylaşma / GitHub'a yükleme**.\n"
            "5. Bu uygulamada sol menüdeki **Gemini API Key** alanına yapıştır."
        )
        st.markdown("**Anahtarı ortam değişkeni olarak ayarlama (opsiyonel):**")
        st.code("# Windows (PowerShell)\n$env:GEMINI_API_KEY = \"anahtarin\"\n\n"
                "# macOS / Linux\nexport GEMINI_API_KEY=\"anahtarin\"", language="bash")
        st.markdown("**Minimum Python kodu:**")
        st.code(
            "from google import genai\n\n"
            "client = genai.Client(api_key='ANAHTARIN')\n"
            "resp = client.models.generate_content(\n"
            "    model='gemini-flash-latest',\n"
            "    contents='Merhaba, kendini tanıt.'\n"
            ")\n"
            "print(resp.text)",
            language="python",
        )
        st.code("pip install streamlit google-genai", language="bash")

    with k2:
        st.markdown(
            "### Cursor IDE\n"
            "1. [cursor.com](https://cursor.com) → **Download** ile indir ve kur.\n"
            "2. Açılışta VS Code ayarlarını içe aktarabilirsin.\n"
            "3. `Ctrl+L` → Chat, `Ctrl+K` → satır içi düzenleme, `Ctrl+I` → Agent/Composer.\n\n"
            "### Windsurf\n"
            "1. [windsurf.com](https://windsurf.com) adresinden indir ve kur.\n"
            "2. Cascade paneli ile doğal dilde kod yazdır.\n\n"
            "### VS Code + GitHub Copilot\n"
            "1. [code.visualstudio.com](https://code.visualstudio.com) → VS Code kur.\n"
            "2. Extensions (`Ctrl+Shift+X`) → **GitHub Copilot** ve **GitHub Copilot Chat** kur.\n"
            "3. GitHub hesabınla giriş yap (öğrenci iseniz GitHub Student Pack ile ücretsiz olabilir).\n"
            "4. `Ctrl+Alt+I` ile Chat'i aç; Agent modunu seç.\n\n"
            "### Roo Code / Cline eklentileri\n"
            "1. Extensions'ta **Roo Code** veya **Cline** ara ve kur.\n"
            "2. Ayarlardan sağlayıcı olarak **Google Gemini** seç, API Key'ini yapıştır.\n"
            "3. Doğal dilde görev ver, eklenti dosyaları oluşturup düzenlesin."
        )

# ----------------------------------------------------------------------------
# 6) Vibe Coding
# ----------------------------------------------------------------------------
with tabs[4]:
    st.header("🎧 Vibe Coding Yaklaşımı ve Kullanılacak Araçlar")
    st.info(
        "**Vibe coding;** öğrencilerin algoritma ve mantığı doğal dille ifade ederek AI Kod Asistanlarına "
        "kod yazdırdığı ve oluşan kodları test edip yönlendirdiği yöntemdir."
    )

    st.subheader("Döngü")
    st.markdown("`Fikir / Mantık` → `Doğal dilde prompt` → `AI kod üretir` → `Çalıştır & test et` → "
                "`Hatayı / eksiği AI'ya anlat` → `Tekrar`")

    st.subheader("Kullanılacak Geliştirme Ortamları")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 🖱️ Cursor IDE veya Windsurf")
        st.write("Doğrudan yapay zeka odaklı geliştirilen kod editörleri.")
    with b:
        st.markdown("#### 🧩 VS Code + eklentiler")
        st.write("GitHub Copilot / Roo Code / Cline eklentileri.")

    st.subheader("Ders İçi Metodoloji (PÇ 2, 14)")
    st.markdown(
        "Öğrenciye sıfırdan sentaks ezberletmek yerine şu becerileri kazandırır:\n"
        "- *\"Nasıl bir prompt yazarsam AI bana **Supabase** bağlantısını doğru kurar?\"*\n"
        "- *\"Üretilen koddaki hatayı AI'ya nasıl ayıklatırım (**debugging**)?\"*"
    )

    st.subheader("Örnek Promptlar")
    with st.expander("🗄️ Supabase bağlantısı için prompt"):
        st.code(
            "Sen kıdemli bir Python geliştiricisisin.\n"
            "Streamlit uygulamamda Supabase'e bağlanmak istiyorum.\n"
            "- `supabase` Python kütüphanesini kullan.\n"
            "- URL ve anon key'i st.secrets içinden oku (koda gömme).\n"
            "- 'notlar' tablosundan tüm satırları çekip st.dataframe ile göster.\n"
            "- Yeni not ekleyen bir form yaz.\n"
            "- Hata olursa st.error ile kullanıcıya anlaşılır mesaj göster.\n"
            "Önce kısa bir plan yaz, sonra kodu ver.",
            language="text",
        )
    with st.expander("🐞 Debugging için prompt"):
        st.code(
            "Aşağıdaki kodu çalıştırdığımda şu hatayı alıyorum.\n\n"
            "HATA:\n<terminaldeki hatanın tamamını yapıştır>\n\n"
            "KOD:\n<ilgili kod>\n\n"
            "Beklediğim davranış: <ne olmasını istiyorsun>\n"
            "Önce hatanın nedenini açıkla, sonra düzeltilmiş kodun tamamını ver.",
            language="text",
        )

    st.subheader("🎯 Mini Debugging Alıştırması")
    st.markdown("Aşağıdaki kodda hata var. AI'ya nasıl ayıklatırdın? Önce kendi prompt'unu yaz, sonra dene.")
    buggy = "for i in range(5)\n    print(i * 2)\nprint('Bitti'"
    st.code(buggy, language="python")
    dbg_prompt = st.text_area(
        "Debugging prompt'un",
        f"Bu Python kodunda hata var, bulup düzelt ve ne yaptığını açıkla:\n\n{buggy}",
        height=140,
    )
    if st.button("AI'ya ayıklat", key="dbg"):
        if not need_key():
            with st.spinner("..."):
                try:
                    st.markdown(ask_gemini(st.session_state["api_key"], model_name,
                                           "Sen bir debugging uzmanısın. Türkçe cevap ver.",
                                           dbg_prompt, 0.2))
                except Exception as e:
                    st.error(f"Hata: {e}")

# ----------------------------------------------------------------------------
# 7) Uygulama: Doğal dille kod yazdırma + basit arayüz
# ----------------------------------------------------------------------------
with tabs[5]:
    st.header("💻 Uygulama: Doğal Dille Kod Yazdırma ve Basit Arayüz Tasarımı")

    mode2 = st.radio("Ne üretelim?", ["🐍 Python kodu", "🌐 Basit web arayüzü (HTML/CSS/JS)"], horizontal=True)

    examples = {
        "🐍 Python kodu": "Kullanıcıdan bir sayı alıp 1'den o sayıya kadar olan asal sayıları listeleyen bir fonksiyon yaz.",
        "🌐 Basit web arayüzü (HTML/CSS/JS)": "Yapılacaklar listesi (todo) uygulaması: görev ekleme, tamamlandı işaretleme ve silme olsun. "
                                              "Modern, mavi tonlarda, mobil uyumlu olsun.",
    }
    nl_prompt = st.text_area("Ne istediğini doğal dille anlat", value=examples[mode2], height=130)

    if mode2.startswith("🐍"):
        sys2 = ("Sen bir Python eğitmenisin. İstenen kodu yaz. Cevap formatı: kısa plan (maddeler), "
                "tek bir ```python kod bloğu, ardından kodun nasıl çalıştırılacağı. Yorum satırları Türkçe olsun.")
    else:
        sys2 = ("Sen bir ön yüz geliştiricisisin. İstenen arayüzü TEK bir HTML dosyası olarak (CSS ve JS içinde) yaz. "
                "Cevap olarak SADECE ```html kod bloğu döndür, başka açıklama ekleme.")

    if st.button("Kodu üret", type="primary", key="gen"):
        if not nl_prompt.strip():
            st.warning("Bir istek yaz.")
        elif not need_key():
            with st.spinner("Kod yazılıyor..."):
                try:
                    st.session_state["gen_out"] = ask_gemini(
                        st.session_state["api_key"], model_name, sys2, nl_prompt, temperature
                    )
                    st.session_state["gen_mode"] = mode2
                except Exception as e:
                    st.error(f"Hata: {e}")

    out = st.session_state.get("gen_out")
    if out:
        st.markdown("### Üretilen cevap")
        st.markdown(out)

        # Koddan blok ayıkla
        import re
        m = re.search(r"```(?:html|python)?\n(.*?)```", out, re.S)
        code = m.group(1) if m else out

        if st.session_state.get("gen_mode", "").startswith("🌐"):
            st.markdown("### 👀 Canlı önizleme")
            render_html(code, height=500)
            st.download_button("⬇️ index.html indir", code, file_name="index.html", mime="text/html")
        else:
            st.download_button("⬇️ kod.py indir", code, file_name="kod.py", mime="text/x-python")
            st.caption("İndirdiğin dosyayı VS Code'da açıp çalıştır ve test et — vibe coding döngüsünün parçası! "
                       "Hata çıkarsa 🎧 Vibe Coding sekmesindeki debugging promptunu kullan.")
