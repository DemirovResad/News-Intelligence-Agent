import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional
from urllib.parse import urlparse

import requests
import trafilatura


from langfuse import observe

from state import NewsIntelState

TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
RAW_RESULTS_DIR = os.environ.get("RAW_RESULTS_DIR", "raw_results")


DEFAULT_COUNTRY = "azerbaijan"

def _resolve_country(country: Optional[str]) -> str:
    """UI artıq Tavily-nin dəstəklədiyi tam ölkə siyahısından (dropdown) seçim
    etdiyi üçün alias mapping-ə ehtiyac yoxdur — sadəcə normallaşdırır (boşdursa
    default-a düşür)."""
    if not country:
        return DEFAULT_COUNTRY
    return country.strip().lower()


_NEWS_DOMAIN_OVERRIDES = {
    "azerbaijan": ["report.az", "apa.az", "trend.az", "oxu.az", "azertag.az", "musavat.com"],
}


_discovered_domains_cache: dict = {}


MAX_DISCOVERED_DOMAINS = 6

MIN_NEWS_RESULTS_THRESHOLD = 5


DEFAULT_NEWS_LOOKBACK_DAYS = 180
DEFAULT_TOP_N_NEWS = 20


PROFILE_CATEGORIES = [
    "identity",
    "founding_date",
    "employee_count",
    "activity",
    "sector",
    "headquarters",
    "website",
    "customers",
    "partners",
    "leadership",
    "financials",
    "risk",
    "legal",
]


PROFILE_RESULTS_PER_QUERY = 3   


def _domain_from_url(url: Optional[str]) -> Optional[str]:
    """Tavily nəticəsi ayrıca mənbə adı vermir — domenini URL-dən çıxarırıq."""
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except ValueError:
        return None


def _discover_country_news_domains(resolved_country: str) -> List[str]:
    """_NEWS_DOMAIN_OVERRIDES-də olmayan ölkələr üçün include_domains-i əl ilə
    yazmaq əvəzinə Tavily-nin özündən tapır: "{ölkə} news" sorğusunun
    (topic=general) nəticələrindəki domenləri çıxarır. Nəticə boşdursa (API
    xətası və s.) boş siyahı qaytarır — çağıran defolt topic="news"-ə keçir."""
    if resolved_country in _discovered_domains_cache:
        return _discovered_domains_cache[resolved_country]

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"{resolved_country} news",
        "topic": "general",
        "search_depth": "basic",
        "max_results": 10,
        "include_raw_content": False,
    }

    domains: List[str] = []
    try:
        response = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        for r in response.json().get("results", []):
            domain = _domain_from_url(r.get("url"))
            if domain and domain not in domains:
                domains.append(domain)
            if len(domains) >= MAX_DISCOVERED_DOMAINS:
                break
    except requests.RequestException as e:
        print(f"[search_node] '{resolved_country}' üçün domen kəşfi alınmadı: {type(e).__name__}: {e}")

    _discovered_domains_cache[resolved_country] = domains
    return domains


def _fetch_tavily_news(query: str, days: int, country: str, top_n: int) -> List[dict]:
    """Tək bir sorğu üçün Tavily-nin topic="news" nəticələrini gətirir."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": "news",
        "days": days,
        "search_depth": "basic",
        "max_results": top_n,
        "include_raw_content": False,
    }

    response = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def _fetch_tavily_news_domain_restricted(query: str, include_domains: List[str], top_n: int) -> List[dict]:
    """topic="news" əhatəsi zəif olan ölkələr üçün — topic="general" + öz təyin
    etdiyimiz/kəşf olunmuş mənbə siyahısı ilə axtarır. "days" parametri
    general-də dəstəklənmir, tarix filtri _within_lookback post-filter-i ilə
    aparılır (_search_recent_news-də artıq var)."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": "general",
        "include_domains": include_domains,
        "search_depth": "basic",
        "max_results": top_n,
        "include_raw_content": False,
    }

    response = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def _fetch_tavily_general(query: str, country: str) -> List[dict]:
    """Tək bir sorğu üçün Tavily-nin topic="general" nəticələrini gətirir."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "topic": "general",
        "country": country,
        "search_depth": "basic",
        "max_results": PROFILE_RESULTS_PER_QUERY,
        "include_raw_content": False,
    }

    response = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """ISO 8601 (2025-01-01T00:00:00Z) və RFC 2822 (Wed, 01 Jan 2025 00:00:00 GMT)
    formatlarının hər ikisini dəstəkləyir — Tavily bəzən ikincini qaytarır və
    sadəcə fromisoformat istifadə etsək, belə itemlər səssizcə None-a düşüb
    sort-u pozur."""
    if not date_str:
        return None
    try:
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _within_lookback(date_str: Optional[str], days: int) -> bool:
    """date sahəsinə əsasən son N gün filtri (Tavily-nin "days" parametri soft filter
    olduğu üçün post-filter kimi saxlanılır)."""
    parsed = _parse_date(date_str)
    if parsed is None:
        # Tarix parse olunmadısa kənarda qoymuruq — extract mərhələsi qərar versin
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return parsed >= cutoff


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _dedup(items: List[dict]) -> List[dict]:
    """Link və normallaşdırılmış başlığa görə dublikatları çıxarır."""
    seen_links, seen_titles = set(), set()
    deduped = []
    for item in items:
        link = item.get("link")
        title_key = _normalize_title(item.get("title"))
        if link and link in seen_links:
            continue
        if title_key and title_key in seen_titles:
            continue
        if link:
            seen_links.add(link)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(item)
    return deduped


def _fetch_article_text(url: Optional[str]) -> Optional[str]:
    """Linkdən məqalənin tam mətnini çıxarır (menyu/reklam kimi zibili atır).
    LLM-ə vermək üçün snippet əvəzinə istifadə olunur. Alınmasa None qaytarır —
    boru xəttini dayandırmır."""
    if not url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded)
    except Exception:
        return None


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip()).strip("_") or "unknown"


def _save_raw_results(company_name: str, raw_results: List[dict]) -> Optional[str]:
    """raw_results-u JSON faylına yazır. Alınmasa None qaytarır — boru xəttini dayandırmır."""
    try:
        os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{_sanitize_filename(company_name)}_{timestamp}.json"
        path = os.path.join(RAW_RESULTS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        print(f"[search_node] raw_results saxlanıldı: {os.path.abspath(path)}")
        return path
    except OSError as e:
        print(f"[search_node] raw_results saxlanılmadı: {type(e).__name__}: {e}")
        return None


def _build_query(company_name: str, country: Optional[str], sector: Optional[str]) -> str:
    """Ad qeyri-müəyyənliyini azaltmaq üçün ölkə/sektor konteksti ilə sorğu qurur.
    Planlama mərhələsinə ehtiyac yoxdur — API sorğunu birbaşa özü qurur."""
    parts = [company_name]
    if country:
        parts.append(country)
    if sector:
        parts.append(sector)
    return " ".join(parts)
                                        

def _search_recent_news(company_name: str, country: Optional[str], sector: Optional[str],
                         resolved_country: str, days: int, top_n: int) -> List[dict]:
    """Son xəbərləri axtarır. Manual seed-də olan ölkələr (bax: _NEWS_DOMAIN_OVERRIDES)
    birbaşa topic="general"+include_domains istifadə edir (məlum ki, topic="news"
    əhatəsi zəifdir). Qalan bütün ölkələr üçün ƏVVƏLCƏ defolt topic="news" sınanılır;
    real nəticə sayı MIN_NEWS_RESULTS_THRESHOLD-dan azdırsa (əhatə zəif çıxdı),
    YALNIZ O ZAMAN dinamik domen kəşfinə keçilir. Qərar ölkə adına görə fərziyyə
    ilə deyil, faktiki API nəticəsinə əsasən verilir."""
    query = _build_query(company_name, country, sector)
    # DİQQƏT: override yalnız istifadəçi ÖLKƏNİ AÇIQ ŞƏKİLDƏ seçdikdə işə düşməlidir.
    # `country` boşdursa, `resolved_country` default olaraq "azerbaijan"-a düşür —
    # bu, ölkəsi Azərbaycanla əlaqəsi olmayan şirkətləri (məs. CD Projekt Red)
    # səhvən Azərbaycan xəbər saytlarına məhdudlaşdırardı. Ona görə override-i
    # yalnız `country` faktiki verilibsə tətbiq edirik.
    override_domains = _NEWS_DOMAIN_OVERRIDES.get(resolved_country) if country else None

    if override_domains:
        raw_results = _fetch_tavily_news_domain_restricted(query, override_domains, top_n)
    else:
        raw_results = _fetch_tavily_news(query, days=days, country=resolved_country, top_n=top_n)
        # Domen-kəşf fallback-ı da yalnız ölkə açıq seçilibsə mənalıdır — əks
        # halda "{default ölkə} news" sorğusu əlaqəsiz domenlər tapıb axtarışı
        # yenidən səhv istiqamətə yönləndirə bilər.
        if country and len(raw_results) < MIN_NEWS_RESULTS_THRESHOLD:
            print(f"[search_node] topic=\"news\" '{resolved_country}' üçün {len(raw_results)} nəticə "
                  f"verdi (əhatə zəif) — domen kəşfinə keçilir")
            discovered_domains = _discover_country_news_domains(resolved_country)
            if discovered_domains:
                fallback_results = _fetch_tavily_news_domain_restricted(query, discovered_domains, top_n)
                if fallback_results:
                    raw_results = fallback_results

    raw_items: List[dict] = []
    for r in raw_results:
        raw_items.append({
            "title": r.get("title"),
            "snippet": r.get("content"),
            "link": r.get("url"),
            "date": r.get("published_date"),
            "source_field": "recent_news",
            "news_source": _domain_from_url(r.get("url")),
        })

    filtered = [item for item in raw_items if _within_lookback(item["date"], days)]
    deduped = _dedup(filtered)
    deduped.sort(key=lambda item: _parse_date(item["date"]) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    top_results = deduped[:top_n]

    for item in top_results:
        item["content"] = _fetch_article_text(item["link"])

    return top_results


def _search_profile_categories(search_plan: dict, resolved_country: str) -> List[dict]:
    """plan.py-da hazırlanan 13 profil kateqoriyasının sorğularını Tavily-nin topic="general" ilə icra edir."""
    raw_items: List[dict] = []
    for category in PROFILE_CATEGORIES:
        queries = search_plan.get(category, [])
        for query in queries:
            results = _fetch_tavily_general(query, country=resolved_country)
            for r in results:
                raw_items.append({
                    "title": r.get("title"),
                    "snippet": r.get("content"),
                    "link": r.get("url"),
                    "date": r.get("published_date"),
                    "source_field": category,
                })
    deduped = _dedup(raw_items)
    for item in deduped:
        item["content"] = _fetch_article_text(item["link"])
    return deduped


@observe(name="search_node")
def search_node(state: NewsIntelState) -> dict:
    """İki hissəni icra edir:
    1) Son xəbərlər — API sorğunu özü qurub topic="news" ilə axtarır (plana ehtiyac yoxdur)
    2) 13 profil sahəsi — plan.py-nın hazırladığı sorğularla topic="general" ilə axtarır
    Hər ikisi eyni raw_results siyahısına (source_field ilə fərqləndirilərək) yığılır."""
    company_name = state["company_name"]
    country = state.get("company_country")
    sector = state.get("company_sector")
    search_plan = state.get("search_plan", {})
    days = state.get("days_back") or DEFAULT_NEWS_LOOKBACK_DAYS
    top_n = state.get("top_n_news") or DEFAULT_TOP_N_NEWS

    resolved_country = _resolve_country(country)

    news_results = _search_recent_news(company_name, country, sector, resolved_country, days, top_n)
    profile_results = _search_profile_categories(search_plan, resolved_country)
    raw_results = state.get("raw_results", []) + news_results + profile_results

    _save_raw_results(company_name, raw_results)

    return {"raw_results": raw_results}