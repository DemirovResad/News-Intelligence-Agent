from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from call_model import get_model, invoke_with_retry
from state import NewsIntelState
from langfuse import observe

# Hər profil kateqoriyası üçün TƏK kanonik (EN) mövzu ifadəsi. search_plan-da
# hər kateqoriya üçün YALNIZ 2 sorğu olacaq: bu EN ifadə + ölkənin öz dilinə
# tərcüməsi (aşağıya bax). Kateqoriya adları search.py-dakı PROFILE_CATEGORIES
# ilə EYNİ olmalıdır (search_plan.get(category, []) ilə oxunur).
CATEGORY_TOPICS: Dict[str, str] = {
    "identity": "official name history",
    "founding_date": "founded date",
    "employee_count": "number of employees",
    "activity": "business activity products services",
    "sector": "industry sector",
    "headquarters": "headquarters location",
    "website": "official website",
    "customers": "corporate customers clients",
    "partners": "business partners",
    "leadership": "CEO executive leadership",
    "financials": "revenue financials funding",
    "risk": "risk controversy criticism",
    "legal": "lawsuit legal case",
}

_CANONICAL_TOPICS: List[str] = list(CATEGORY_TOPICS.values())
_CATEGORY_ORDER: List[str] = list(CATEGORY_TOPICS.keys())

# company_country verilməyəndə tərcümə üçün istifadə olunan defolt ölkə —
# search.py-dakı DEFAULT_COUNTRY ilə uyğun (layihənin əsas bazarı).
DEFAULT_LANGUAGE_COUNTRY = "azerbaijan"

# Ölkə üzrə tərcümə nəticəsini process daxilində keşləyirik ki, eyni ölkədən
# olan hər company üçün təkrar LLM çağırışı olmasın.
_country_topics_cache: Dict[str, List[str]] = {}


class _TranslatedTopics(BaseModel):
    """LLM-in YEGANƏ vəzifəsi — mövzu ifadələrini ölkənin əsas dilinə tərcümə etmək,
    sorğu MƏZMUNUNU yaratmaq YOX. Struktur/kateqoriyalar sabit template-dən gəlir."""
    topics: List[str] = Field(
        description="Hər ifadənin ölkənin əsas dilinə tərcüməsi, GİRİŞLƏ EYNİ SIRADA"
    )


def _translate_topics_for_country(country: str, config: Optional[dict] = None) -> List[str]:
    """13 kanonik (EN) mövzu ifadəsini ölkənin əsas dilinə tərcümə edir, ölkə üzrə
    keşlənir. Uğursuz olsa (ya da say uyğun gəlməsə) boş siyahı qaytarır — çağıran
    bu halda sadəcə EN sorğu ilə davam edir, pipeline dayanmır."""
    if country in _country_topics_cache:
        return _country_topics_cache[country]

    model = get_model()
    structured_model = model.with_structured_output(_TranslatedTopics)
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(_CANONICAL_TOPICS))
    prompt = (
        f"Aşağıdakı {len(_CANONICAL_TOPICS)} ingiliscə ifadəni \"{country}\" ölkəsinin "
        f"ƏSAS DANIŞIQ DİLİNƏ tərcümə et. Heç nə əlavə etmə, izah yazma — YALNIZ tərcümə.\n"
        f"GİRİŞLƏ EYNİ SIRADA, EYNİ SAYDA ({len(_CANONICAL_TOPICS)}) ifadə qaytar, heç birini buraxma:\n\n"
        f"{numbered}"
    )

    translated: List[str] = []
    try:
        result = invoke_with_retry(structured_model, prompt, config=config)
        if len(result.topics) == len(_CANONICAL_TOPICS):
            translated = result.topics
        else:
            print(f"[plan_node] '{country}' tərcüməsi say uyğunsuzluğu: "
                  f"{len(result.topics)}/{len(_CANONICAL_TOPICS)} — nəzərə alınmır")
    except Exception as e:
        print(f"[plan_node] '{country}' üçün dil tərcüməsi alınmadı: {type(e).__name__}: {e}")

    _country_topics_cache[country] = translated
    return translated


def _build_profile_search_plan(company_name: str, country: Optional[str], sector: Optional[str],
                                config: Optional[dict] = None) -> Dict[str, List[str]]:
    """13 kateqoriyanın hər biri üçün YALNIZ 2 sorğu qurur: EN (kanonik, LLM-siz,
    zəmanətli) + ölkənin öz dilində (LLM tərcüməsi, keşlənərək). Ölkə verilməyibsə,
    DEFAULT_LANGUAGE_COUNTRY istifadə olunur. Tərcümə alınmasa, sadəcə EN sorğu qalır —
    pipeline dayanmır."""
    context = " ".join(part for part in [country, sector] if part)
    translate_country = country or DEFAULT_LANGUAGE_COUNTRY
    localized_topics = _translate_topics_for_country(translate_country, config=config)

    search_plan: Dict[str, List[str]] = {}
    for i, category in enumerate(_CATEGORY_ORDER):
        queries = [
            " ".join(part for part in [company_name, context, CATEGORY_TOPICS[category]] if part)
        ]
        if i < len(localized_topics):
            queries.append(
                " ".join(part for part in [company_name, context, localized_topics[i]] if part)
            )
        search_plan[category] = queries

    return search_plan


@observe(name="plan_node")
def plan_node(state: NewsIntelState, config: Optional[dict] = None) -> dict:
    """13 profil kateqoriyası üçün axtarış sorğularını qurur — hər kateqoriya üçün
    YALNIZ 2 sorğu: EN (kanonik) + ölkənin öz dili. LLM YALNIZ tərcümə üçün (bir
    dəfə, ölkə üzrə keşlənərək) çağırılır, sorğu strukturunu/kateqoriyalarını
    yaratmır."""
    company_name = state["company_name"]
    country = state.get("company_country")
    sector = state.get("company_sector")

    search_plan = _build_profile_search_plan(company_name, country, sector, config=config)
    return {"search_plan": search_plan}