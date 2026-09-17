import contextvars
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from call_model import get_model, invoke_with_retry
from search import PROFILE_CATEGORIES
from state import NewsIntelState, NewsAnalysis, NewsClassification
from langfuse import observe

# Google AI Studio-dakı dəqiq Gemma model id-si — .env-də MODEL_NAME ilə override oluna bilər.
GEMMA_MODEL_NAME = os.getenv("MODEL_NAME")
GEMMA_PROVIDER = "google_genai"

# Mətni bu qədər simvoldan sonra kəsirik ki, Gemma-nın kontekst limitini aşmayaq.
MAX_TEXT_CHARS = 8000

# Profil mənbələri birləşdirilərkən hər bir nəticədən götürüləcək maksimum simvol.
PROFILE_ITEM_TEXT_CHARS = 800

# Xəbər təhlili batch formatında aparılır — hər LLM çağırışı bu qədər xəbəri birgə təhlil edir.
NEWS_BATCH_SIZE = 5

# Batch-lər bu qədər thread ilə paralel işlədilir (network-bound çağırışlar üçün
# ThreadPoolExecutor kifayətdir — CPU-bound olmadığı üçün GIL maneə olmur).
MAX_PARALLEL_BATCHES = 4

BATCH_CLASSIFY_PROMPT = """Sən "{company_name}" şirkəti haqqında xəbərləri təhlil edən analitik agentsən.
{context_lines}

Xəbərlər ingilis, rus və ya başqa dildə olsa belə, cavabını mütləq Azərbaycan dilində yaz.

Aşağıda "INDEX N" ilə nişanlanmış {n} xəbər var. HƏR BİRİ ÜÇÜN AYRI-AYRI bunları müəyyən et:
- sentiment: "positive", "neutral" və ya "negative"
- category: maliyyə, hüquqi, məhsul, rəhbərlik dəyişikliyi, M&A, kiber-hadisə, reputasiya və ya digər uyğun kateqoriya (Azərbaycan dilində)
- summary: 1-2 cümləlik qısa xülasə (Azərbaycan dilində)
- implication: bu xəbər "{company_name}" üçün nə anlama gəlir, hər hansı risk siqnalı varmı (Azərbaycan dilində, yoxdursa boş burax)
- is_relevant: xəbər həqiqətən YUXARIDA TƏSVİR OLUNAN "{company_name}" şirkətinə aiddirsə true;
  əgər eyni və ya oxşar adlı, amma FƏRQLİ ölkədə/sektorda fəaliyyət göstərən başqa bir şirkətə
  aiddirsə (ad üst-üstə düşməsi təsadüfidirsə) və ya ümumiyyətlə əlaqəsizdirsə false

VACIB: nəticədə hər xəbərin "index" dəyərini AŞAĞIDAKI İLƏ EYNİ saxla ki, hansı təhlilin
hansı xəbərə aid olduğu itməsin. {n} xəbərin hamısı üçün nəticə qaytar, heç birini buraxma.

{items_block}
"""

PROFILE_PROMPT = """Sən "{company_name}" şirkəti haqqında qısa profil kartları hazırlayan analitik agentsən.
{context_lines}

Aşağıda kateqoriyalara görə qruplaşdırılmış axtarış nəticələri var. Bu mənbələrdən \
istifadə edərək AŞAĞIDAKI SAHƏLƏRİN HƏR BİRİNİ AYRI-AYRI, BİR-BİRİNDƏN ASILI OLMADAN \
doldur. MƏTNİN HAMISINI AZƏRBAYCAN DİLİNDƏ YAZ — mənbələr ingilis (və ya başqa) \
dildə olsa belə, tərcümə edib Azərbaycanca yaz.

Hər sahə üçün əvvəlcə ətraflı təsviri, sonra value-nu, sonra mənbəni yaz:

- xulase: mənbələrdəki faydalı məlumatdan qurulmuş təsvir — 3-4 cümlə, \
Azərbaycan dilində. İdentity sahəsi üçün 7-8 cümlə.

- value: yuxarıda yazdığın xülasənin İÇİNDƏN həmin sahə üçün ƏN VACIB, KONKRET \
FAKTI çıxarıb qeyd et — ümumi şablon cümlə yox, birbaşa dəyər. Məsələn:
  - leadership → rəhbərin/CEO-nun konkret adı (məs. "CEO: Elşən Məmmədov")
  - employee_count → konkret rəqəm və ya aralıq (məs. "250-500 işçi")
  - headquarters → şəhər/ölkə adı (məs. "Bakı, Azərbaycan")
  - sector → sektorun konkret adı (məs. "Bank və maliyyə xidmətləri")
  - founding_date → konkret il/tarix (məs. "2005-ci il")
  - website → domən (məs. "example.az")
  - digər sahələr üçün də eyni məntiqlə: təsvirdəki ən konkret, faydalı faktı seç.
  value MAKSIMUM 1 cümlə/ifadə olmalıdır — izah yox, birbaşa fakt.

- mənbə: value-nu QURDUĞUN ƏSAS FAKTIN aşağıdakı mənbələr arasından HANSI birindən
  götürüldüyünü göstər — mənbənin AŞAĞIDA VERİLMİŞ tam URL-ni olduğu kimi qaytar
  (dəyişdirmə, qısaltma, uydurma YOX). Yalnız aşağıdakı sources bölməsində FAKTİKİ
  olaraq verilmiş URL-lərdən birini yaz. Bir neçə mənbədən istifadə etmisənsə, ən
  vacib faktı verən TƏK mənbəni seç. Uyğun mənbə tapa bilmirsənsə, boş burax.

Sahələr:
- identity: Rəsmi ad və alternativ adlar (brend, keçmiş ad)
- founding_date: Yaranma tarixi (nə vaxt, hansı formada təsis edilib)
- employee_count: İşçi sayı
- activity: Əsas fəaliyyət sahəsi, məhsul/xidmətlər
- sector: Sektor
- headquarters: Baş ofis, yerləşmə
- website: Rəsmi veb-sayt
- customers: Əsas korporativ müştərilər
- partners: Əsas tərəfdaşlar
- leadership: Rəhbərlik (CEO, founders)
- financials: Maliyyə göstəriciləri (aktivlər, gəlir, funding)
- risk: Risk siqnalları
- legal: Hüquqi məsələlər/proseslər

QAYDALAR:
- Yalnız mənbələrdə açıq şəkildə olan məlumatdan istifadə et. Heç nəyi uydurma.
- Bir sahə üçün mənbələrdə HEÇ bir faydalı məlumat yoxdursa, o sahəni (value, xulase, mənbə) null burax.
- Yuxarıdakı xülasədə konkret fakt yoxdursa (yalnız ümumi/qeyri-müəyyən məlumat varsa), value-ya ilk cümləni yaz — uydurma fakt yaratma.
- Hər sahəni müstəqil yaz — bir-birini təkrarlamasın, bir-birinin mətnini kopyalamasın.
- Bütün mətnlər Azərbaycan dilində olmalıdır, mənbə dilindən asılı olmayaraq (mənbə sahəsindəki URL istisnadır — onu olduğu kimi saxla).

{sources_block}
"""


def _build_structured_model(schema_cls):
    model = get_model(model_name=GEMMA_MODEL_NAME, provider=GEMMA_PROVIDER)
    return model.with_structured_output(schema_cls)


class _ProfileFieldWithSource(BaseModel):
    """CompanyProfileSummary-nin (state.py) hər sahəsi kimi {xulase, value}, üstəlik
    "mənbə" — value-nun hansı konkret URL-dən götürüldüyü. state.py-a toxunmuruq,
    structured output üçün YALNIZ bu lokal sxemadan istifadə edirik."""
    xulase: Optional[str] = None
    value: Optional[str] = None
    mənbə: Optional[str] = Field(
        default=None,
        description="value-nun götürüldüyü mənbənin tam URL-i (sources bölməsindəki URL-lərdən biri, olduğu kimi)",
    )


class _CompanyProfileWithSources(BaseModel):
    """CompanyProfileSummary-nin 13 sahəsi ilə EYNİ struktur, sadəcə hər sahə
    _ProfileFieldWithSource (mənbə əlavə olunmuş)."""
    identity: Optional[_ProfileFieldWithSource] = None
    founding_date: Optional[_ProfileFieldWithSource] = None
    employee_count: Optional[_ProfileFieldWithSource] = None
    activity: Optional[_ProfileFieldWithSource] = None
    sector: Optional[_ProfileFieldWithSource] = None
    headquarters: Optional[_ProfileFieldWithSource] = None
    website: Optional[_ProfileFieldWithSource] = None
    customers: Optional[_ProfileFieldWithSource] = None
    partners: Optional[_ProfileFieldWithSource] = None
    leadership: Optional[_ProfileFieldWithSource] = None
    financials: Optional[_ProfileFieldWithSource] = None
    risk: Optional[_ProfileFieldWithSource] = None
    legal: Optional[_ProfileFieldWithSource] = None


def _first_sentence(text: str, max_chars: int = 140) -> str:
    """xülasədən ilk cümləni çıxarır (value boş qalanda fallback üçün)."""
    text = text.strip()
    match = re.search(r"[.!?]", text)
    sentence = text[:match.end()] if match else text
    return sentence[:max_chars].strip()


def _fill_missing_values(profile: _CompanyProfileWithSources) -> _CompanyProfileWithSources:
    """PROFILE_PROMPT LLM-ə deyir ki, xülasədə konkret fakt yoxdursa value-ya ilk
    cümləni yazsın — amma bu, PROMPT-a əməl olunacağına zəmanət vermir (real halda
    xülasə dolu, value boş qala bilir, elə "Risk" kartında olduğu kimi). Kod
    səviyyəsində fallback: value boşdursa və xülasə varsa, xülasənin ilk cümləsini
    value kimi doldururuq ki, kart heç vaxt boş görünməsin."""
    for field_name in profile.model_fields:
        field_obj = getattr(profile, field_name)
        if field_obj is not None and not (field_obj.value or "").strip() and (field_obj.xulase or "").strip():
            field_obj.value = _first_sentence(field_obj.xulase)
    return profile


class _IndexedClassification(BaseModel):
    """Batch cavabında hər nəticəni orijinal xəbərlə (index üzrə) uyğunlaşdırmaq üçün —
    NewsClassification-ın özünü dəyişmirik, sadəcə əhatə edirik."""
    index: int = Field(description="Bu təhlilin aid olduğu xəbərin INDEX-i (promptdakı ilə eyni)")
    classification: NewsClassification


class _BatchClassification(BaseModel):
    """Bir batch-dəki (≤5) bütün xəbərlərin təhlili — tək LLM çağırışında."""
    results: List[_IndexedClassification]


def _chunk(items: List[dict], size: int) -> List[List[dict]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _build_batch_items_block(items: List[dict]) -> str:
    blocks = []
    for i, item in enumerate(items):
        text = (item.get("content") or item.get("snippet") or item.get("title") or "").strip()
        blocks.append(
            f"### INDEX {i}\n"
            f"Başlıq: {item.get('title') or ''}\n"
            f"Mətn:\n{text[:MAX_TEXT_CHARS]}"
        )
    return "\n\n".join(blocks)


def _build_context_lines(state: NewsIntelState) -> str:
    """Ölkə/sektor konteksti — ad oxşarlığı olan, amma fərqli ölkədə/sektorda
    fəaliyyət göstərən başqa şirkətləri ayırd etməyə kömək edir (həm xəbər
    classify-də is_relevant üçün, həm də profil çıxarılmasında)."""
    context_parts = []
    if state.get("company_country"):
        context_parts.append(f"Ölkə: {state['company_country']}")
    if state.get("company_sector"):
        context_parts.append(f"Sektor: {state['company_sector']}")
    return ("Şirkət konteksti — " + ", ".join(context_parts) + ".") if context_parts else ""


def _classify_news_batch(structured_batch_model, company_name: str, context_lines: str,
                          batch: List[dict], config: Optional[dict] = None) -> Dict[int, NewsClassification]:
    """Bir batch-i (≤5 xəbər) TƏK LLM çağırışında təhlil edir, {local_index: classification}
    lüğəti qaytarır. Çağırış uğursuz olsa boş dict qaytarır — digər batch-ləri dayandırmır."""
    prompt = BATCH_CLASSIFY_PROMPT.format(
        company_name=company_name,
        context_lines=context_lines,
        n=len(batch),
        items_block=_build_batch_items_block(batch),
    )
    try:
        batch_result = invoke_with_retry(structured_batch_model, prompt, config=config)
        return {r.index: r.classification for r in batch_result.results}
    except Exception as e:
        titles = ", ".join((item.get("title") or "")[:30] for item in batch)
        print(f"[extract_node XƏTA - batch] [{titles}]: {type(e).__name__}: {e}")
        return {}


@observe(name="extract_node")
def extract_node(state: NewsIntelState, config: Optional[dict] = None) -> dict:
    """raw_results-dakı recent_news elementlərini Gemma ilə təhlil edir: sentiment,
    kateqoriya, xülasə, implication çıxarır; əlaqəsiz (is_relevant=false) xəbərləri
    filtr edir. Dedup artıq search_node-da (link/başlıq üzrə) edilib.

    Xəbərlər NEWS_BATCH_SIZE (5) ölçüsündə batch-lərə bölünür — hər batch TƏK LLM
    çağırışında birgə təhlil olunur (index ilə uyğunlaşdırılaraq), batch-lər isə
    ThreadPoolExecutor ilə paralel işlədilir."""
    company_name = state["company_name"]
    all_items = state.get("raw_results", [])
    news_items = [
        item for item in all_items
        if item.get("source_field") == "recent_news"
    ]
    print(f"[extract_node] raw_results: {len(all_items)} ümumi, 'recent_news': {len(news_items)}")

    context_lines = _build_context_lines(state)
    structured_batch_model = _build_structured_model(_BatchClassification)

    batches = _chunk(news_items, NEWS_BATCH_SIZE)
    print(f"[extract_node] {len(news_items)} xəbər, {len(batches)} batch (batch ölçüsü={NEWS_BATCH_SIZE}), "
          f"paralel worker={min(MAX_PARALLEL_BATCHES, len(batches)) or 1}")

    analyzed: List[NewsAnalysis] = []
    # Hər batch üçün AYRICA context kopyası — eyni Context obyekti iki thread-də
    # eyni anda "run" oluna bilməz (contextvars reentrant deyil, məhz "already
    # entered" xətasının səbəbi budur). Hər kopya eyni valideyn context-dən
    # götürüldüyü üçün trace nesting-i qorunur, sadəcə hər biri müstəqil obyektdir.
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, max(len(batches), 1))) as executor:
        future_to_batch = {
            executor.submit(contextvars.copy_context().run, _classify_news_batch, structured_batch_model, company_name, context_lines, batch, config): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            index_to_classification = future.result()
            for local_index, item in enumerate(batch):
                classification = index_to_classification.get(local_index)
                if classification is None or not classification.is_relevant:
                    continue
                analyzed.append(NewsAnalysis(
                    title=item.get("title") or "",
                    date=item.get("date"),
                    source=item.get("news_source"),
                    url=item.get("link") or "",
                    sentiment=classification.sentiment,
                    category=classification.category,
                    summary=classification.summary,
                    implication=classification.implication,
                ))

    print(f"[extract_node] nəticə: {len(analyzed)}/{len(news_items)} xəbər analiz olundu")
    return {"analyzed_news": analyzed}


def _build_profile_sources_block(raw_results: List[dict]) -> str:
    """Profil kateqoriyalarının (identity, founding_date və s.) nəticələrini
    kateqoriya üzrə qruplaşdırılmış mətn blokuna çevirir."""
    grouped: dict = {}
    for item in raw_results:
        field = item.get("source_field")
        if field in PROFILE_CATEGORIES:
            grouped.setdefault(field, []).append(item)

    blocks = []
    for category in PROFILE_CATEGORIES:
        items = grouped.get(category, [])
        if not items:
            continue
        lines = [f"## {category}"]
        for item in items:
            text = (item.get("content") or item.get("snippet") or "").strip()[:PROFILE_ITEM_TEXT_CHARS]
            lines.append(f"- {item.get('title') or ''} ({item.get('link') or ''}): {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@observe(name="extract_profile_node")
def extract_profile_node(state: NewsIntelState, config: Optional[dict] = None) -> dict:
    """raw_results-dakı 13 axtarış kateqoriyasının nəticələrini Gemma ilə ümumiləşdirib
    hər profil sahəsini (identity, founding_date, employee_count, activity, sector,
    headquarters, website, customers, partners, leadership, financials, risk, legal)
    xülasə (xulase), konkret fakt (value) VƏ mənbə URL-i (mənbə) kimi çıxarır
    (_CompanyProfileWithSources — CompanyProfileSummary-nin state.py-a toxunmadan
    "mənbə" əlavə olunmuş lokal versiyası)."""
    company_name = state["company_name"]
    raw_results = state.get("raw_results", [])

    sources_block = _build_profile_sources_block(raw_results)
    if not sources_block:
        print(f"[extract_profile_node] sources_block boşdur — raw_results-da profil kateqoriyalarına ({PROFILE_CATEGORIES}) uyğun 'source_field' tapılmadı")
        return {"company_profile": None}

    context_lines = _build_context_lines(state)
    structured_model = _build_structured_model(_CompanyProfileWithSources)
    prompt = PROFILE_PROMPT.format(company_name=company_name, context_lines=context_lines, sources_block=sources_block)

    try:
        company_profile = invoke_with_retry(structured_model, prompt, config=config)
        if company_profile is not None:
            company_profile = _fill_missing_values(company_profile)
    except Exception as e:
        print(f"[extract_profile_node XƏTA] {type(e).__name__}: {e}")
        company_profile = None

    return {"company_profile": company_profile}