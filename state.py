from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field


class NewsAnalysis(BaseModel):
    """Tək bir xəbər üçün LLM analiz nəticəsi."""
    title: str = Field(description="Xəbərin başlığı")
    date: Optional[str] = Field(default=None, description="Xəbərin dərc tarixi")
    source: Optional[str] = Field(default=None, description="Xəbərin mənbəyi (nəşr/sayt adı)")
    url: str = Field(description="Xəbərin linki")
    sentiment: str = Field(
        description="Xəbərin tonu: positive, neutral və ya negative"
    )
    category: str = Field(
        description=(
            "Xəbərin əsas kateqoriyası, məsələn: maliyyə, hüquqi, məhsul, "
            "rəhbərlik dəyişikliyi, M&A, kiber-hadisə, reputasiya və s."
        )
    )
    summary: str = Field(description="Xəbərin qısa xülasəsi (1-2 cümlə)")
    implication: Optional[str] = Field(
        default=None,
        description="Bu xəbər şirkət üçün nə anlama gəlir — agentin yozumu",
    )


class ProfileFieldSummary(BaseModel):
    """Bir profil sahəsi üçün həm kart, həm ətraflı məzmun.
    Hər ikisi Azərbaycan dilində olmalıdır, mənbə dilindən asılı olmayaraq."""
    xulase: Optional[str] = Field(
        default=None,
        description="Ətraflı xülasə — 3-4 cümlə, Azərbaycan dilində",
    )
    value: Optional[str] = Field(
        default=None,
        description=(
            "Kartda göstəriləcək QISA məzmun — xülasənin içindən "
            "həmin sahə üçün ən vacib, konkret faktı (məs. rəhbərin adı, işçi "
            "sayı, şəhər) çıxarır, Azərbaycan dilində, MAKSIMUM 1 ifadə"
        ),
    )


class CompanyProfileSummary(BaseModel):
    """Şirkət profilinin hər sahəsi 'value' (qısa, kart üçün, maks 1 cümlə) və
    'tooltip' (ətraflı, 3-4 cümlə) olaraq ayrılıb. Hər ikisi Azərbaycan dilində
    olmalıdır. Mənbələrdə məlumat yoxdursa sahə (həm value, həm tooltip) null
    olmalıdır — uydurma qadağandır."""
    identity: Optional[ProfileFieldSummary] = Field(
        default=None, description="Rəsmi ad və alternativ adlar (brend, keçmiş ad)"
    )
    founding_date: Optional[ProfileFieldSummary] = Field(
        default=None, description="Yaranma tarixi"
    )
    employee_count: Optional[ProfileFieldSummary] = Field(
        default=None, description="İşçi sayı"
    )
    activity: Optional[ProfileFieldSummary] = Field(
        default=None, description="Əsas fəaliyyət sahəsi, məhsul/xidmətlər"
    )
    sector: Optional[ProfileFieldSummary] = Field(
        default=None, description="Sektor"
    )
    headquarters: Optional[ProfileFieldSummary] = Field(
        default=None, description="Baş ofis, yerləşmə"
    )
    website: Optional[ProfileFieldSummary] = Field(
        default=None, description="Rəsmi veb-sayt"
    )
    customers: Optional[ProfileFieldSummary] = Field(
        default=None, description="Əsas korporativ müştərilər"
    )
    partners: Optional[ProfileFieldSummary] = Field(
        default=None, description="Əsas tərəfdaşlar"
    )
    leadership: Optional[ProfileFieldSummary] = Field(
        default=None, description="Rəhbərlik (CEO, founders)"
    )
    financials: Optional[ProfileFieldSummary] = Field(
        default=None, description="Maliyyə göstəriciləri (aktivlər, gəlir, funding)"
    )
    risk: Optional[ProfileFieldSummary] = Field(
        default=None, description="Risk siqnalları"
    )
    legal: Optional[ProfileFieldSummary] = Field(
        default=None, description="Hüquqi məsələlər/proseslər"
    )


class RiskAssessment(BaseModel):
    """Ümumi risk qiymətləndirməsi."""
    level: str = Field(description="Risk səviyyəsi: low, medium və ya high")
    reasoning: str = Field(description="Bu səviyyənin seçilmə əsaslandırması")


class FinalReport(BaseModel):
    """Synthesize mərhələsinin yekun çıxışı."""
    company_profile: Optional[CompanyProfileSummary] = None
    analyzed_news: List[NewsAnalysis] = Field(default_factory=list)
    executive_summary: str = Field(description="Yekun ümumi xülasə")
    risk_assessment: RiskAssessment
    overall_trend: Optional[str] = Field(
        default=None, description="Ümumi sentiment/trend qiymətləndirməsi"
    )


class NewsIntelState(TypedDict):
    # Giriş
    company_name: str
    company_country: Optional[str]  # name ambiguity üçün əlavə kontekst
    company_sector: Optional[str]   # name ambiguity üçün əlavə kontekst
    days_back: Optional[int]        # xəbər axtarışı üçün lookback pəncərəsi (gün); yoxdursa search.py-dakı default istifadə olunur
    top_n_news: Optional[int]       # saxlanılacaq maksimum xəbər sayı; yoxdursa search.py-dakı default istifadə olunur

    # Plan mərhələsi — profil sahələrinə uyğun struktur JSON (SerpAPI-ready sorğular)
    search_plan: dict

    # Search mərhələsi
    raw_results: List[dict]  # {title, snippet, link, date, source_field}

    # Extract mərhələsi
    company_profile: Optional[CompanyProfileSummary]
    analyzed_news: List[NewsAnalysis]  # dublikat/aidiyyətsiz xəbərlər filtr olunmuş halda

    # Synthesize mərhələsi
    final_report: Optional[FinalReport]



class NewsClassification(BaseModel):
    """Gemma-nın structured output-la doldurduğu sahələr."""
    sentiment: str
    category: str
    summary: str
    implication: Optional[str] = None
    is_relevant: bool = True