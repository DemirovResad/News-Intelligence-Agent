import ast
import html
import json
from datetime import datetime, timezone

import streamlit as st

# Tavily-nin "country" parametrinin dəstəklədiyi tam siyahı (OpenAPI enum-dan) —
# UI-da sərbəst mətn əvəzinə birbaşa bu siyahıdan seçim edilir.
TAVILY_COUNTRIES = [
    "afghanistan", "albania", "algeria", "andorra", "angola", "argentina", "armenia",
    "australia", "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados",
    "belarus", "belgium", "belize", "benin", "bhutan", "bolivia", "bosnia and herzegovina",
    "botswana", "brazil", "brunei", "bulgaria", "burkina faso", "burundi", "cambodia",
    "cameroon", "canada", "cape verde", "central african republic", "chad", "chile",
    "china", "colombia", "comoros", "congo", "costa rica", "croatia", "cuba", "cyprus",
    "czech republic", "denmark", "djibouti", "dominican republic", "ecuador", "egypt",
    "el salvador", "equatorial guinea", "eritrea", "estonia", "ethiopia", "fiji",
    "finland", "france", "gabon", "gambia", "georgia", "germany", "ghana", "greece",
    "guatemala", "guinea", "haiti", "honduras", "hungary", "iceland", "india", "indonesia",
    "iran", "iraq", "ireland", "israel", "italy", "jamaica", "japan", "jordan",
    "kazakhstan", "kenya", "kuwait", "kyrgyzstan", "latvia", "lebanon", "lesotho",
    "liberia", "libya", "liechtenstein", "lithuania", "luxembourg", "madagascar",
    "malawi", "malaysia", "maldives", "mali", "malta", "mauritania", "mauritius",
    "mexico", "moldova", "monaco", "mongolia", "montenegro", "morocco", "mozambique",
    "myanmar", "namibia", "nepal", "netherlands", "new zealand", "nicaragua", "niger",
    "nigeria", "north korea", "north macedonia", "norway", "oman", "pakistan", "panama",
    "papua new guinea", "paraguay", "peru", "philippines", "poland", "portugal", "qatar",
    "romania", "russia", "rwanda", "saudi arabia", "senegal", "serbia", "singapore",
    "slovakia", "slovenia", "somalia", "south africa", "south korea", "south sudan",
    "spain", "sri lanka", "sudan", "sweden", "switzerland", "syria", "taiwan",
    "tajikistan", "tanzania", "thailand", "togo", "trinidad and tobago", "tunisia",
    "turkey", "turkmenistan", "uganda", "ukraine", "united arab emirates",
    "united kingdom", "united states", "uruguay", "uzbekistan", "venezuela", "vietnam",
    "yemen", "zambia", "zimbabwe",
]

# ============================================================
# TODO: özün bağla — kompilə olunmuş LangGraph graph obyektini
# import et. Adətən graph.py-da belə bir şey olur:
#
#     from langgraph.graph import StateGraph
#     graph = StateGraph(NewsIntelState)
#     ...
#     compiled_graph = graph.compile()
#
# Aşağıdakı importu öz modulunun adına uyğunlaşdır:
# ============================================================
try:
    from pipline import run_pipeline 
except ImportError:
    run_pipeline = None

# Web Search söndürüləndə (SerpAPI limitini qorumaq üçün) birbaşa extract
# node-larını çağırmaq üçün — plan/search-ü bypass edir.
try:
    from extract import extract_node, extract_profile_node
except ImportError:
    extract_node = None
    extract_profile_node = None

try:
    from call_model import get_langfuse_client, get_langfuse_handler
except ImportError:
    get_langfuse_client = None
    get_langfuse_handler = None


st.set_page_config(page_title="News Intelligence", layout="wide")

SENTIMENT_COLORS = {
    "positive": "#1a7f37",
    "negative": "#cf222e",
    "neutral": "#57606a",
}
SENTIMENT_LABELS_AZ = {
    "positive": "Müsbət",
    "negative": "Mənfi",
    "neutral": "Neytral",
}

st.markdown(
    """
    <style>
    .news-card {
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        background-color: #0d1117;
    }
    .news-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
    .news-meta { font-size: 0.82rem; color: #8b949e; margin-bottom: 8px; }
    .sentiment-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
    }
    .category-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        background-color: #21262d;
        color: #c9d1d9;
        margin-right: 6px;
    }
    .news-summary { margin-top: 10px; }
    .news-implication {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #21262d;
        font-size: 0.9rem;
        color: #d29922;
    }

    .profile-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }
    .profile-card {
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        background-color: #0d1117;
        overflow: hidden;
    }
    .profile-subfield { margin-bottom: 10px; }
    .profile-subfield-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8b949e;
        margin-bottom: 2px;
    }
    .profile-subfield-value {
        font-size: 0.92rem;
        color: #e6edf3;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FIELD_LABELS_AZ = {
    "identity": "Kimlik",
    "founding_date": "Yaranma tarixi",
    "employee_count": "İşçi sayı",
    "activity": "Fəaliyyət",
    "sector": "Sektor",
    "headquarters": "Baş ofis",
    "website": "Veb-sayt",
    "customers": "Müştərilər",
    "partners": "Tərəfdaşlar",
    "leadership": "Rəhbərlik",
    "financials": "Maliyyə",
    "risk": "Risk",
    "legal": "Hüquqi",
}


def _split_value_xulase(field_data):
    """extract.py-dakı PROFILE_PROMPT hər sahə üçün {"xulase": ..., "value": ...,
    "mənbə": ...} formatında obyekt qaytarır. model_dump()-dan sonra bu, dict kimi
    gəlir — "value" (qısa fakt), "xulase" (ətraflı təsvir) və "mənbə" (mənbə URL-i)
    ayrılır. Dict deyilsə (defensiv fallback — məs. sahə sadə string kimi
    saxlanılıbsa və ya səhvən stringify olunub dict-ə bənzəyən mətn yaranıbsa)
    sadə mətn kimi rəftar edilir, mənbə boş qaytarılır."""
    if isinstance(field_data, dict):
        return (
            str(field_data.get("value") or "").strip(),
            str(field_data.get("xulase") or "").strip(),
            str(field_data.get("mənbə") or "").strip(),
        )

    text = str(field_data or "").strip()
    if text.startswith("{") and "'value'" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return (
                    str(parsed.get("value") or "").strip(),
                    str(parsed.get("xulase") or "").strip(),
                    str(parsed.get("mənbə") or "").strip(),
                )
        except (ValueError, SyntaxError):
            pass
    return text, "", ""

# ------------------------------------------------------------
# Sidebar — axtarış forması
# ------------------------------------------------------------
with st.sidebar:
    st.header("Axtarış parametrləri")

    use_web_search = st.toggle(
        "Web Search istifadə et",
        value=False,
        help="Sönük olanda API limiti xərclənmir — saxlanılmış raw_results JSON faylından oxuyub birbaşa extract (LLM) mərhələsini test edir.",
    )

    company_name = st.text_input("Şirkət adı", placeholder="məs. ABB")

    if use_web_search:
        country = st.selectbox(
            "Ölkə",
            options=[""] + TAVILY_COUNTRIES,
            index=TAVILY_COUNTRIES.index("azerbaijan") + 1,
            format_func=lambda c: "— seçilməyib —" if c == "" else c.title(),
            help="Tavily-nin dəstəklədiyi ölkələr — yazaraq axtara bilərsən.",
        )
        sector = st.text_input(
            "Sektor (opsional)",
            placeholder="məs. Bank",
            help="Ad qeyri-müəyyənliyini aradan qaldırmaq üçün — eyni adlı fərqli şirkətlər ola bilər.",
        )
        months_back = st.number_input("Neçə aylıq xəbər", min_value=1, max_value=60, value=6, step=1)
        top_n_news = st.number_input("Maksimum xəbər sayı", min_value=1, max_value=100, value=20, step=1)
        uploaded_json_file = None
        run_search = st.button("Axtar", type="primary", use_container_width=True)
    else:
        country = None
        sector = None
        months_back = None
        top_n_news = None
        uploaded_json_file = st.file_uploader(
            "raw_results JSON faylı (search_node-un əvvəlcə saxladığı)",
            type=["json"],
        )
        run_search = st.button(
            "Extract-i işə sal (JSON-dan)", type="primary", use_container_width=True
        )

st.title("📊 News Intelligence")

if "result_state" not in st.session_state:
    st.session_state.result_state = None

# ------------------------------------------------------------
# Pipeline-ı işə sal
# ------------------------------------------------------------
if run_search:
    if not company_name.strip():
        st.error("Şirkət adını daxil et.")

    elif use_web_search:
        if run_pipeline is None:
            st.error(
                "`run_pipeline` import olunmayıb — faylın yuxarısındakı TODO-nu "
                "öz `graph.py` modulunun adına uyğunlaşdır."
            )
        else:
            # TODO: bu dict-in açarlarını öz NewsIntelState sxeminə uyğunlaşdır
            initial_state = {
                "company_name": company_name.strip(),
                "company_country": country.strip() if country else None,
                "company_sector": sector.strip() if sector else None,
                # NewsIntelState "days_back" (gün) gözləyir — UI aylıq seçim aldığı üçün çeviririk
                "days_back": int(months_back) * 30 if months_back else None,
                "top_n_news": int(top_n_news) if top_n_news else None,
            }
            with st.spinner(f"'{company_name}' üçün pipeline işləyir (planner → search → extract)..."):
                try:
                    # run_pipeline bütün addımları tək Langfuse trace-i altında işə salır
                    st.session_state.result_state = run_pipeline(initial_state)
                except Exception as e:
                    st.error(f"Pipeline xətası: {type(e).__name__}: {e}")
                    st.session_state.result_state = None

    else:
        # Web Search sönük — JSON-dan raw_results oxu, plan/search-ü bypass et,
        # yalnız extract (LLM) mərhələsini işə sal. Bu da Langfuse-da ayrıca
        # trace kimi görünür (mövcuddursa).
        if extract_node is None or extract_profile_node is None:
            st.error("`extract_node` / `extract_profile_node` import olunmayıb — `extract.py` yoxla.")
        elif uploaded_json_file is None:
            st.error("raw_results JSON faylını yüklə.")
        else:
            try:
                raw_results = json.load(uploaded_json_file)
            except Exception as e:
                st.error(f"JSON oxuna bilmədi: {type(e).__name__}: {e}")
                raw_results = None

            if raw_results is not None:
                state = {
                    "company_name": company_name.strip(),
                    "raw_results": raw_results,
                }
                with st.spinner(f"'{company_name}' üçün extract (LLM) işləyir — search bypass edilib..."):
                    try:
                        if get_langfuse_client is not None and get_langfuse_handler is not None:
                            langfuse = get_langfuse_client()
                            handler = get_langfuse_handler()
                            with langfuse.start_as_current_observation(
                                as_type="span",
                                name="news_intel_extract_only",
                            ) as span:
                                span.update(input=state)
                                config = {"callbacks": [handler]}
                                news_result = extract_node(state, config=config)
                                profile_result = extract_profile_node(state, config=config)
                                span.update(output={
                                    "analyzed_news_count": len(news_result.get("analyzed_news", [])),
                                    "has_profile": profile_result.get("company_profile") is not None,
                                })
                        else:
                            news_result = extract_node(state)
                            profile_result = extract_profile_node(state)

                        st.session_state.result_state = {**news_result, **profile_result}
                    except Exception as e:
                        st.error(f"Extract xətası: {type(e).__name__}: {e}")
                        st.session_state.result_state = None

result_state = st.session_state.result_state

if result_state is None:
    st.info("Axtarışa başlamaq üçün sol paneldə məlumatları doldur və 'Axtar' düyməsinə bas.")
    st.stop()


# ------------------------------------------------------------
# Profil kartları — üzərinə gələndə (hover) açılan ətraflı məlumat
# pəncərəsi, tam CSS-only (position: absolute overlay, body-yə təsir etmir)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .profile-card-wrap {
        position: relative;
    }
    .profile-tooltip {
        position: absolute;
        top: 100%;
        left: 0;
        margin-top: 8px;
        width: 320px;
        max-height: 320px;
        overflow-y: auto;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.55);
        z-index: 1000;
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        transition: opacity 0.15s ease;
    }
    .profile-card-wrap:hover .profile-tooltip {
        opacity: 1;
        visibility: visible;
    }
    .profile-tooltip-title {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #e6edf3;
    }
    .profile-tooltip-subfield { margin-bottom: 8px; }
    .profile-tooltip-subfield-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8b949e;
    }
    .profile-tooltip-subfield-value {
        font-size: 0.88rem;
        color: #d9dee3;
        line-height: 1.4;
    }
    .profile-tooltip-summary {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #21262d;
        font-size: 0.85rem;
        color: #8b949e;
        line-height: 1.4;
    }
    .profile-card-source-link {
        position: absolute;
        top: 10px;
        right: 12px;
        z-index: 2;
        text-decoration: none;
        font-size: 0.95rem;
        opacity: 0.75;
        line-height: 1;
    }
    .profile-card-source-link:hover {
        opacity: 1;
    }
    .identity-source-link {
        text-decoration: none;
        font-size: 0.95rem;
        margin-left: 8px;
        opacity: 0.75;
    }
    .identity-source-link:hover {
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Profil kartları (yan-yana)
# ------------------------------------------------------------
company_profile = result_state.get("company_profile")

if company_profile is not None:
    st.subheader("Şirkət profili")

    profile_dict = (
        company_profile.model_dump() if hasattr(company_profile, "model_dump") else dict(company_profile)
    )

    # identity — kart deyil, birbaşa şirkət adı (value) başlıq kimi, altında
    # xülasə göstərilir. "Kimlik" label sözü istifadə olunmur.
    identity_raw = profile_dict.pop("identity", None)
    if identity_raw:
        identity_value, identity_xulase, identity_source = _split_value_xulase(identity_raw)
        if identity_value:
            source_link_html = (
                f'<a class="identity-source-link" href="{html.escape(identity_source)}" target="_blank" title="Mənbəyə keç">🔗</a>'
                if identity_source else ""
            )
            st.markdown(f"### {html.escape(identity_value)}{source_link_html}", unsafe_allow_html=True)
        if identity_xulase:
            st.caption(identity_xulase)
        st.write("")

    profile_items = [(k, v) for k, v in profile_dict.items() if v]

    if profile_items:
        # 4 kart bir sırada, qalanı növbəti sıraya — hamısı eyni sabit hündürlükdə.
        # Hər sahə indi müstəqil, flat bir string-dir (nested obyekt deyil) — kartda
        # qısa (clamp olunmuş) göstərilir, tam 3-4 cümlə hover tooltip-də açılır.
        CARD_HEIGHT = 130
        cols_per_row = 4
        for i in range(0, len(profile_items), cols_per_row):
            row_items = profile_items[i:i + cols_per_row]
            cols = st.columns(len(row_items))
            for col, (field_name, field_data) in zip(cols, row_items):
                with col:
                    title_label = FIELD_LABELS_AZ.get(field_name, field_name.replace('_', ' ').title())
                    value_text, xulase_text, source_url = _split_value_xulase(field_data)

                    # Kartda YALNIZ value görünür (key sözlər yazılmır) — xülasə
                    # yalnız hover tooltip-də açılır. Mənbə varsa, kartın sağ
                    # yuxarı küncündə klikə açılan link ikonu göstərilir.
                    card_body_html = f'<div class="profile-subfield-value">{html.escape(value_text)}</div>'
                    tooltip_summary_html = (
                        f'<div class="profile-tooltip-summary">{html.escape(xulase_text)}</div>'
                        if xulase_text else ""
                    )
                    source_link_html = (
                        f'<a class="profile-card-source-link" href="{html.escape(source_url)}" '
                        f'target="_blank" title="Mənbəyə keç">🔗</a>'
                        if source_url else ""
                    )

                    st.markdown(
                        f"""
                        <div class="profile-card-wrap">
                            <div class="profile-card" style="height:{CARD_HEIGHT}px;">
                                {source_link_html}
                                <div class="profile-card-title">{html.escape(title_label)}</div>
                                {card_body_html}
                            </div>
                            <div class="profile-tooltip">
                                <div class="profile-tooltip-title">{html.escape(title_label)}</div>
                                {tooltip_summary_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("Profil sahələri üçün məlumat tapılmadı.")
else:
    st.caption("Profil məlumatı yoxdur.")

st.divider()

# ------------------------------------------------------------
# News kartları — tarixə görə (yenidən köhnəyə) sıralanmış
# ------------------------------------------------------------
def _parse_news_date(date_str):
    """Sıralama üçün tarixi parse edir. Alınmasa None qaytarır — belə itemlər siyahının sonuna düşür."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    except ValueError:
        return None


analyzed_news = result_state.get("analyzed_news", [])
news_dicts = [
    item.model_dump() if hasattr(item, "model_dump") else dict(item)
    for item in analyzed_news
]
news_dicts.sort(
    key=lambda d: _parse_news_date(d.get("date")) or datetime.min.replace(tzinfo=timezone.utc),
    reverse=True,
)

st.subheader(f"Xəbərlər ({len(news_dicts)})")

if not news_dicts:
    st.caption("Uyğun xəbər tapılmadı.")
else:
    for item_dict in news_dicts:
        title = item_dict.get("title", "")
        date = item_dict.get("date", "")
        source = item_dict.get("source", "")
        url = item_dict.get("url", "")
        sentiment = (item_dict.get("sentiment") or "neutral").lower()
        category = item_dict.get("category", "")
        summary = item_dict.get("summary", "")
        implication = item_dict.get("implication")

        sentiment_color = SENTIMENT_COLORS.get(sentiment, "#57606a")
        sentiment_label = SENTIMENT_LABELS_AZ.get(sentiment, sentiment)

        title_html = f'<a href="{url}" target="_blank" style="color:#58a6ff; text-decoration:none;">{title}</a>' if url else title

        implication_html = (
            f'<div class="news-implication">⚠️ {implication}</div>' if implication else ""
        )

        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-title">{title_html}</div>
                <div class="news-meta">{source} · {date}</div>
                <span class="sentiment-badge" style="background-color:{sentiment_color};">{sentiment_label}</span>
                <span class="category-badge">{category}</span>
                <div class="news-summary">{summary}</div>
                {implication_html}
            </div>
            """,
            unsafe_allow_html=True,
        )