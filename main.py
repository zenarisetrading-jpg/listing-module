# main.py
"""
S2C — Super App: Integrated Listing Auditor & Creator
Version: V22 (Safety Hardening + Scraper Fix)
Logic: V13 base + V20 Refinement + V21 Safety Filters + V22 Input Sanitization.
Fixes: 
1. Scraper URL connection error resolved.
2. Banned words are now stripped from the 'target keywords' list before AI generation.
"""

import streamlit as st
import pandas as pd
import json
import re
import io
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import random
from PIL import Image, ImageFilter, ImageOps
import os
import plotly.express as px
import numpy as np
import base64 

# Import OpenAI immediately to fail fast if missing
try:
    from openai import OpenAI
except ImportError:
    st.error("OpenAI library not installed. Please run `pip install openai`.")
    st.stop()

# --- GLOBAL SETUP & BRANDING ---
st.set_page_config(layout="wide", page_title="Zenarise Listing Module")

# --- BRAND STYLING CSS ---
def apply_brand_styling():
    st.markdown("""
        <style>
            /* Stronger Apple Font Stack */
            html, body, [class*="css"], div, button, input, select, textarea, .stAlert {
                font-family: "SF Pro Text", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
                color: #1E1E1E !important; /* Force dark grey text generally */
            }
            /* Force inputs to have light backgrounds and dark text */
            .stTextInput input, .stTextArea textarea, .stNumberInput input {
                background-color: #FFFFFF !important;
                color: #1E1E1E !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #1E1E1E !important;
                font-weight: 700 !important;
                letter-spacing: -0.5px;
            }
            /* Primary Color Accent */
            :root {
                --primary-color: #2EA398 !important;
            }
            /* Custom Button Styling - Minimal & Posh */
            div.stButton > button {
                background-color: #2EA398 !important; /* Brand Teal */
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.6rem 1.2rem !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                transition: all 0.2s ease-in-out !important;
            }
            div.stButton > button:hover {
                background-color: #F58220 !important; /* Brand Orange on Hover */
                box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
                transform: translateY(-1px) !important;
            }
            div.stButton > button:active {
                transform: translateY(1px) !important;
                box-shadow: none !important;
            }
            /* Remove default Streamlit clutter */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Adjust padding */
            .block-container {
                padding-top: 1rem; 
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

# Apply styling globally
apply_brand_styling()

# Helper function for logo centering
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

# Initialize Session State used across apps
if "openai_client" not in st.session_state: st.session_state["openai_client"] = None
# New state for landing page navigation
if "on_landing_page" not in st.session_state: st.session_state["on_landing_page"] = True

# --- SECRETS AUTH (V13 Logic) ---
# Try to load key from secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if api_key and not st.session_state["openai_client"]:
    try:
        # Initialize client silently if key exists
        st.session_state["openai_client"] = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client with provided secret: {e}")

client = st.session_state["openai_client"]

# --- BRANDED SIDEBAR NAVIGATION ---
# Only show sidebar if NOT on the landing page
if not st.session_state["on_landing_page"]:
    with st.sidebar:
        # Home button to return to landing page
        if st.button("🏠 Home", use_container_width=True):
            st.session_state["on_landing_page"] = True
            st.rerun()
            
        st.header("⚙️ Settings")
        # V13 Auth Check Logic
        if client:
             st.success("🟢 AI System Online")
        else:
             st.error("🔴 AI System Offline")
             st.warning("⚠️ Crucial Configuration Missing.")
             st.info("Please configure the `OPENAI_API_KEY` in your `.streamlit/secrets.toml` file.")
             st.stop()

# Optional Libs Checks
try: import pdfplumber
except ImportError: pdfplumber = None
try: import pytesseract
except ImportError: pytesseract = None

# ============================================================
# 1. HELPER FUNCTIONS (EXACT V13)
# ============================================================
FACTUAL_PATTERNS = [
    r'(\d+(?:\.\d+)?\s*(?:ml|l|oz|gallon|quart|pint|cup))',
    r'(\d+(?:\.\d+)?\s*(?:g|kg|lb|ounce|pound))',
    r'(\d+(?:\.\d+)?\s*(?:cm|mm|m|in|ft|inch|foot|meter))',
    r'(\d+(?:\.\d+)?\s*(?:hr|hour|min|minute|sec|second))',
    r'(\d+(?:\.\d+)?\s*(?:°c|°f|k|deg))',
    r'(\d+\s*(?:pcs|pack|set|count))',
    r'(18/8\s*ss|304\s*ss|stainless\s*steel)',
    r'(bpa\s*free|non-toxic)',
    r'(ipx\d|waterproof|leakproof)',
]

def extract_factual_claims(text: str) -> list:
    claims = set()
    text = text.lower()
    for pattern in FACTUAL_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            claims.add(match.strip())
    return sorted(list(claims))

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        return img
    except Exception: return img

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    if pytesseract is None: return ""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = preprocess_image_for_ocr(img)
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception: return ""

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    out = []
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages: out.append(page.extract_text() or "")
        except: pass
    return "\n".join(out)

def extract_text_for_file(uploaded_file) -> str:
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"): return extract_text_from_pdf_bytes(raw)
    if name.endswith((".png",".jpg",".jpeg")): return extract_text_from_image_bytes(raw)
    return ""

def fetch_html_text(url, timeout=10):
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        soup = BeautifulSoup(response.content, "html.parser")
        targets = soup.find_all(['table', 'ul', 'div'], class_=re.compile('spec|desc|detail|attr'))
        text = "\n".join([t.get_text(separator="\n") for t in targets])
        if not text: text = soup.get_text(separator="\n")
        return text[:10000]
    except Exception: return ""

def load_csv_keywords(uploaded_file):
    if uploaded_file is None: return []
    try:
        df = pd.read_csv(uploaded_file)
        cands = [c for c in df.columns if 'keyword' in c.lower() or 'search' in c.lower() or 'phrase' in c.lower()]
        if cands: return list(df[cands[0]].dropna().astype(str).tolist()[:100])
    except: pass
    return []

# ============================================================
# 2. LLM PROMPTS & WRAPPERS (V21: SAFETY PATCH)
# ============================================================
SYSTEM_BH_PROMPT = """
You are a senior product strategist. Input: canonical_specs (JSON list), keywords (list).
Return strictly a JSON ARRAY of 8-12 Benefit Hierarchy entries based on the specs and keywords.
Each entry must be: { "feature":string, "functional_benefit":string, "emotional_benefit":string, "priority":int(1-5, 5 is highest) }
Do NOT invent numeric measurements. Use only provided specs/evidence.
Do not include markdown formatting like ```json at the start or end.
"""

# V21 Update: STRICT SAFETY CHECK added to Rule 2
BASE_LISTING_PROMPT = """
You are an elite Amazon Listing Optimizer for brand 'S2C'.
Your task is to transform an underperforming listing into a category leader.

CRITICAL INSTRUCTIONS:
1.  **RESOLVE AUDIT FAILINGS:** You must aggressively fix every point listed in `audit_report_to_address.opportunities_to_fix`.
2.  **STRICT SAFETY CHECK:** You are FORBIDDEN from using any words listed in `constraints.forbidden_words`. 
    - If the input implies "prevent" or "treat", you MUST substitute them with safe terms like "help reduce", "support", "maintain", "promote", or "minimize".
    - Do NOT make disease claims (e.g., "cures insomnia"). Use structure/function claims (e.g., "promotes restful sleep").
3.  **INTEGRATE FACTS:** You MUST include all `factual_claims_to_include` within the bullet points. Weave them naturally into benefit-driven statements. Do not just list them.
4.  **MAXIMIZE KEYWORDS:** Seamlessly integrate the provided `keywords` into the Title, Bullets, and Description for maximum relevance without keyword stuffing. Priority goes to high-volume keywords not present in the original listing.
5.  **FOLLOW BLUEPRINT:** Use the `benefit_hierarchy` as the structural backbone for your content.

Inputs provided: `benefit_hierarchy`, `audit_report_to_address`, `factual_claims_to_include`, `keywords`, `constraints`.

Outputs required (strictly JSON):
- title (string): <200 chars, Must include top 3-5 keywords naturally.
- bullets (array of 5 strings): Narrative, persuading paragraphs. Each must contain facts and benefits.
- description (string): HTML formatted, highly persuasive sales copy.
- backend_search_terms (array of 5 strings): Don't repeat title/bullet keywords.
"""

APLUS_INSTRUCTIONS = """
- aplus_modules (array of exactly 5 objects): Suggest a sequence of 5 distinct A+ modules that tell a product story (e.g., Hero -> Main Benefit -> Secondary Feature -> Lifestyle/Social Proof -> Specs/Comparison). 
Every single module object MUST have this exact structure to act as a design brief:
{ 
    "type": string (e.g., "standard_hero", "standard_text_image", "three_feature_col"),
    "headline": string,
    "body": string (or localized text for multi-block modules),
    "image_prompt_idea": string (A detailed, photorealistic visual description for a graphic designer to create the image for this module. MUST NOT BE N/A.)
}
"""

SYSTEM_REFINE_PROMPT = """
You are an expert Amazon copywriter editor.
Input: current_listing (JSON), user_instruction (string).
Task: Revise the content strictly following the user_instruction.
Return ONLY the updated JSON structure for the listing: title, bullets, description, backend_search_terms, and **aplus_modules**. Preserve the aplus_modules structure perfectly.
Do not include markdown formatting like ```json at the start or end.
"""

def call_llm_json(ai_client, system_prompt, user_content, model="gpt-4o-mini", temp=0.1):
    try:
        resp = ai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            temperature=temp, 
            max_tokens=4000, 
            timeout=110 
        )
        raw_content = resp.choices[0].message.content
        if raw_content:
            cleaned_content = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned_content)
        return None
    except json.JSONDecodeError as e:
        st.error(f"AI response was not valid JSON. Refinement failed. Error: {e}. Raw (truncated): {raw_content[:200]}...")
        return None
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

# ============================================================
# 3. APP STATE & NAVIGATION (V13 Logic)
# ============================================================
if "app_mode" not in st.session_state: st.session_state["app_mode"] = "Audit"
if "audit_data" not in st.session_state: st.session_state["audit_data"] = {} 
if "creator_data" not in st.session_state: st.session_state["creator_data"] = {} 

def nav_to(mode):
    st.session_state["app_mode"] = mode
    st.session_state["on_landing_page"] = False
    st.rerun()

# ============================================================
# 4. MODULE: AUDITOR (V21: UPDATED ROBUST SCRAPER)
# ============================================================

class RobustScraper:
    def __init__(self):
        # Your ScraperAPI Key
        self.api_key = "25a6a24456463da81ea1ff0d5838924c" 
        # FIX APPLIED: Removed markdown brackets to fix connection error
        self.base_url = "http://api.scraperapi.com"

    def scrape_listing(self, url):
        if "amazon" not in url: return None, "Invalid Amazon URL."
        
        # --- 1. Dynamic Region & Forced English Logic ---
        # We default to US
        country_code = 'us'
        
        # If UAE, use UAE Proxy + Force English
        if ".ae" in url:
            country_code = 'ae'
            if "language=" not in url:
                separator = "&" if "?" in url else "?"
                url += f"{separator}language=en_AE"
                
        # If KSA, use KSA Proxy + Force English (en_AE works for KSA too)
        elif ".sa" in url:
            country_code = 'sa'
            if "language=" not in url:
                separator = "&" if "?" in url else "?"
                url += f"{separator}language=en_AE"

        # --- 2. Construct Payload ---
        payload = {
            'api_key': self.api_key,
            'url': url,
            'country_code': country_code, # 'ae', 'sa', or 'us'
            'device_type': 'desktop',
            # 'render': 'true' # Uncomment if you see empty results (costs 5 credits)
        }

        try:
            # 3. Send Request (60s timeout for slower geo-proxies)
            response = requests.get(self.base_url, params=payload, timeout=60)
            
            if response.status_code != 200:
                return None, f"ScraperAPI Failed (Status: {response.status_code}). Msg: {response.text}"
            
            # 4. Parse Content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- Parsing Logic ---
            title_node = soup.find(id="productTitle")
            title = title_node.get_text().strip() if title_node else ""
            
            # Robust Bullet Extraction
            bullets = []
            # ScraperAPI returns clean HTML, so standard selectors usually work
            for li in soup.select("#feature-bullets li"):
                txt = li.get_text().strip()
                if txt and not txt.lower().startswith("make sure"): 
                    bullets.append(txt)
            
            # Fallback for different layouts
            if not bullets:
                bullets = [li.get_text().strip() for li in soup.select(".a-unordered-list.a-vertical.a-spacing-mini li") if li.get_text().strip()]
            
            # Description Extraction
            desc_container = soup.find(id="productDescription")
            desc = desc_container.get_text(separator="\n").strip() if desc_container else ""
            
            # Image Counting
            img_count = len(soup.select("#altImages li.item")) + len(soup.select(".regularAltImageViewLayout li"))
            
            # SAFETY NET: Default to 7 images if title exists but images are hidden/lazy-loaded
            if img_count == 0 and title: 
                img_count = 7 
            
            video_count = len(soup.select(".video-container, #video-block, .a-video-wrapper"))
            video_present = 1 if video_count > 0 else 0

            img_counts = (max(1, img_count-3), 1, 1, video_present) if img_count > 3 else (img_count, 0, 0, video_present)
            
            if not title:
                return None, f"Connected to Amazon {country_code.upper()} (English), but got empty data."
                
            return {"title": title, "bullets": bullets, "description": desc, "img_counts": img_counts}, None

        except Exception as e:
            return None, f"Connection Error: {str(e)}"

# Re-initialize the scraper instance
scraper = RobustScraper()

class HybridAuditor:
    def __init__(self, ai_client): self.client = ai_client
    
    def _score_structure(self, bullets, description, img_counts):
        n_white, n_info, n_life, n_video = img_counts
        GOLD_WHITE, GOLD_INFO, GOLD_LIFE, GOLD_VIDEO = 1, 4, 3, 1
        
        s_white = min(1.0, n_white / GOLD_WHITE) if GOLD_WHITE > 0 else 1.0
        s_info = min(1.0, n_info / GOLD_INFO) if GOLD_INFO > 0 else 1.0
        s_life = min(1.0, n_life / GOLD_LIFE) if GOLD_LIFE > 0 else 1.0
        s_video = min(1.0, n_video / GOLD_VIDEO) if GOLD_VIDEO > 0 else 1.0
        
        media_score = (0.15 * s_white) + (0.35 * s_info) + (0.25 * s_life) + (0.25 * s_video)
        
        b_count = len(bullets)
        avg_b_len = sum(len(b) for b in bullets) / max(1, b_count)
        s_bullets = 1.0 if 5 <= b_count <= 7 and 150 <= avg_b_len <= 250 else 0.5
        
        score = (0.4 * s_bullets) + (0.6 * media_score)
        return min(1.0, round(score, 2)), media_score

    def _analyze_with_llm(self, title, bullets, description, keywords, brand_voice):
        listing_text = f"Title: {title}\nBullets: {bullets}\nDescription: {description}"
        system_prompt = """
        You are an Amazon Audit Algorithm (S2C Scoring). Grade (0.00-1.00) critically.
        Outputs needed in JSON: 
        - relevance_score (how well keywords are integrated naturally)
        - persuasion_score (how benefit-driven and compelling the copy is)
        - strengths (array of 3 specific strong points)
        - opportunities (array of 3 specific fixes needed, focus on missing facts or poor phrasing)
        Do not include markdown formatting.
        """
        user_prompt = json.dumps({"listing": listing_text, "keywords": keywords[:30], "voice": brand_voice})
        return call_llm_json(self.client, system_prompt, user_prompt, model="gpt-4o-mini", temp=0.1) or {}

    def audit_listing(self, title, bullets, description, target_keywords, img_counts):
        full_text = f"{title} {' '.join(bullets)} {description}".lower()
        factual_claims = extract_factual_claims(full_text)
        
        # Expanded list of compliance triggers
        banned_words = ["cure", "treat", "prevent", "diagnose", "best seller", "guarantee", "covid", "virus", "bacteria"]
        found_banned = [w for w in banned_words if f" {w} " in full_text]
        compliance_score = 0.2 if found_banned else 1.0
        
        structure_score, media_score_detail = self._score_structure(bullets, description, img_counts)
        
        ai_result = self._analyze_with_llm(title, bullets, description, target_keywords, "Premium")
        relevance_score = ai_result.get("relevance_score", 0.5)
        persuasion_score = ai_result.get("persuasion_score", 0.5)
        
        raw_avg = (0.30*persuasion_score + 0.25*relevance_score + 0.25*structure_score + 0.20*compliance_score)
        lqi = round(raw_avg * 10, 1)
        
        return {
            "LQI": lqi,
            "scores": {
                "Persuasion": round(persuasion_score*10, 1),
                "Relevance": round(relevance_score*10, 1),
                "Structure & Media": round(structure_score*10, 1),
                "Compliance": round(compliance_score*10, 1)
            },
            "media_detail_score": round(media_score_detail*10, 1),
            "details": {
                "strengths": ai_result.get("strengths",[]),
                "opportunities": ai_result.get("opportunities",[]),
                "banned": found_banned,
                "factual_claims": factual_claims
            }
        }

auditor = HybridAuditor(client)

# ============================================================
# NEW: LANDING PAGE FUNCTION (Branding Only)
# ============================================================
def render_landing_page():
    col_spacer1, col_content, col_spacer2 = st.columns([1, 2, 1])
    with col_content:
        # V20 UX Fix: Perfectly centered logo using HTML
        logo_base64 = image_to_base64("logo.png")
        if logo_base64:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                    <img src="data:image/png;base64,{logo_base64}" width="250">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback if logo.png is missing
            st.markdown("<h1 style='text-align: center; color: #2EA398;'>Zenarise</h1>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; margin-top: 0.5rem; font-size: 2.5rem;'>Listing Module</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem; font-size: 1.1rem;'>Select an action to begin.</p>", unsafe_allow_html=True)

        cta1, cta2 = st.columns(2, gap="medium")
        with cta1:
            if st.button("📊 Grade my existing listing", use_container_width=True):
                st.session_state["app_mode"] = "Audit"
                st.session_state["on_landing_page"] = False
                st.rerun()
        with cta2:
            if st.button("✨ Create new listing", use_container_width=True):
                st.session_state["app_mode"] = "Create"
                st.session_state["on_landing_page"] = False
                st.rerun()

# ============================================================
# MODULE RENDERERS (EXACT V13 LOGIC + V20 FIXES)
# ============================================================
def render_audit_module():
    st.title("📊 Grade Existing Listing")
    st.caption("Analyze content specifically to find gaps and extract facts for optimization.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Target")
        amazon_url = st.text_input("Amazon Product URL (DP Link)")
        if st.button("🚀 Fetch Listing Data", type="primary"):
            if not amazon_url: st.warning("Please enter a URL.")
            else:
                with st.spinner("Scraping Amazon..."):
                    data, err = scraper.scrape_listing(amazon_url)
                    if err: st.error(err)
                    else: st.session_state["scraped_data"] = data; st.success("Fetched!")
        
        data = st.session_state.get("scraped_data", {})
        if data:
            with st.expander("View/Edit Scraped Content", expanded=True):
                curr_title = st.text_area("Title", value=data.get("title",""), height=70, key="aud_title")
                curr_bullets = st.text_area("Bullets (one per line)", value="\n".join(data.get("bullets",[])), height=150, key="aud_bullets")
                curr_desc = st.text_area("Description", value=data.get("description",""), height=100, key="aud_desc")
                imgs = data.get("img_counts", (1,1,1,0))
                ic1,ic2,ic3,ic4 = st.columns(4)
                img_counts = (
                    ic1.number_input("White BG", value=imgs[0], min_value=0, key="aud_iw"),
                    ic2.number_input("Infographic", value=imgs[1], min_value=0, key="aud_ii"),
                    ic3.number_input("Lifestyle", value=imgs[2], min_value=0, key="aud_il"),
                    ic4.number_input("Video", value=imgs[3], min_value=0, max_value=1, key="aud_iv", help="1 if present, 0 if not")
                )
            
            st.markdown("---")
            has_aplus = st.checkbox("✅ Listing already has A+ Content?", value=False, help="If checked, the Creator module will skip generating new A+ suggestions.")

    with col2:
        st.subheader("Context (Keywords)")
        seo_file = st.file_uploader("Upload Cerebro/Xray CSV", type=["csv"], key="aud_seo")
        manual_kws = st.text_area("Additional Keywords (comma separated)", key="aud_manual_kws")

    st.markdown("---")
    
    can_run = st.session_state.get("scraped_data") is not None
    if st.button("📊 Run Hybrid Audit Scoring", type="primary", disabled=not can_run, use_container_width=True):
        bullets_list = [b.strip() for b in curr_bullets.split('\n') if b.strip()]
        target_kws = load_csv_keywords(seo_file)
        if manual_kws: target_kws += [k.strip() for k in manual_kws.split(',') if k.strip()]
        
        auditor.client = client
        with st.spinner("🤖 AI analyzing persuasion, structure, media, and facts..."):
            result = auditor.audit_listing(curr_title, bullets_list, curr_desc, target_kws, img_counts)
            st.session_state["audit_result"] = result
            
            # V21 Update: Persist banned words to session state
            st.session_state["audit_data"] = {
                "raw_title": curr_title,
                "raw_bullets": bullets_list,
                "raw_desc": curr_desc,
                "keywords": target_kws,
                "img_counts": img_counts,
                "initial_lqi": result["LQI"],
                "audit_opportunities": result["details"]["opportunities"],
                "audit_strengths": result["details"]["strengths"],
                "factual_claims": result["details"]["factual_claims"],
                "banned_words_found": result["details"]["banned"], # <--- SAVED HERE
                "has_aplus": has_aplus
            }

    if st.session_state.get("audit_result"):
        res = st.session_state["audit_result"]; lqi = res["LQI"]
        color = "#2EA398" if lqi >= 8.0 else "#F58220" if lqi >= 5.0 else "#E74C3C" # Branded colors
        
        c_score, c_chart = st.columns([2, 3])
        with c_score:
            st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 4rem;'>LQI: {lqi} / 10</h1>", unsafe_allow_html=True)
            st.metric("Media Score (vs Gold Standard)", f"{res['media_detail_score']} / 10")
        
        with c_chart:
            scores = res["scores"]
            df_radar = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())))
            fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,10], title="LQI Pillar Analysis")
            fig.update_traces(fill='toself', line_color='#2EA398')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Parsed Factual Claims")
        facts = res["details"]["factual_claims"]
        if facts: st.info(", ".join(facts))
        else: st.warning("No specific numeric or spec-based claims detected.")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("AI Opportunities")
            for o in res["details"]["opportunities"]: st.error(f"🔸 {o}")
            if res["details"]["banned"]: st.error(f"⛔ Banned Words Found: {', '.join(res['details']['banned'])}")
        with c2:
            st.caption("AI Strengths")
            for s in res["details"]["strengths"]: st.success(f"✅ {s}")
            
        st.markdown("---")
        if st.button("✨ Proceed to Optimizer (Create) ->", type="primary", use_container_width=True):
            nav_to("Create")

def render_creator_module():
    st.title("✨ Create & Optimize Listing")
    if st.button("← Back to Audit Start"): nav_to("Audit")
    
    audit_data = st.session_state.get("audit_data", {})
    if not audit_data:
        st.warning("No audit data found. Please run an audit first.")
        st.stop()
    
    st.markdown("---")
    # ================= STEP 1: CANONICAL SPECS (V13 Logic) =================
    st.header("Step 1: Canonical Specifications Base")
    
    with st.expander("View/Add Supplementary Data Sources", expanded=False):
        c_files, c_url = st.columns(2)
        with c_files: spec_files = st.file_uploader("Add Supplier Specs (PDF/Img)", accept_multiple_files=True, key="cr_specs")
        with c_url: alibaba_url = st.text_input("Add Supplier URL (Alibaba/1688)", key="cr_ali_url")

    # V13 Logic: Extract button -> then show editor
    if st.button("🔄 Extract & Merge Canonical Specs", type="primary"):
        with st.spinner("Analyzing sources for specs..."):
            combined_text = f"Existing Title: {audit_data.get('raw_title','')}\n"
            combined_text += f"Existing Bullets: {' '.join(audit_data.get('raw_bullets',[]))}\n"
            combined_text += f"Existing Description: {audit_data.get('raw_desc','')}\n"
            combined_text += f"\nExtracted Facts: {', '.join(audit_data.get('factual_claims', []))}\n"
            if alibaba_url: combined_text += f"\n---Supplier URL---\n{fetch_html_text(alibaba_url)}\n"
            if spec_files:
                for f in spec_files: combined_text += f"\n---Supplier File ({f.name})---\n{extract_text_for_file(f)}\n"

            system_prompt = "You are a specification engineer. Extract technical specs. Return a strictly valid JSON dictionary {feature:value}. No markdown."
            extracted_dict = call_llm_json(client, system_prompt, combined_text, model="gpt-4o-mini", temp=0.1) 
            
            if extracted_dict:
                rows = [{"feature": k, "value": v, "source": "AI_Extraction", "accept": True} for k,v in extracted_dict.items()]
                st.session_state["creator_data"]["canonical_df"] = pd.DataFrame(rows)
                st.success(f"Extracted {len(rows)} specs.")
            else:
                st.error("Could not extract specs. Try manual entry.")
                st.session_state["creator_data"]["canonical_df"] = pd.DataFrame(columns=["feature", "value", "source", "accept"])

    if "canonical_df" in st.session_state["creator_data"]:
        st.subheader("Review & Confirm Specs")
        # V20 Fix: Ensure the data editor initializes correctly with an 'include' column for filtering
        if 'include' not in st.session_state["creator_data"]["canonical_df"].columns:
             st.session_state["creator_data"]["canonical_df"]['include'] = True
             
        edited_df = st.data_editor(st.session_state["creator_data"]["canonical_df"], 
                                   column_config={"include": st.column_config.CheckboxColumn("Include", default=True)},
                                   num_rows="dynamic", key="cr_spec_editor", hide_index=True)
        
        if st.button("✅ Confirm Specs Base"):
            # V20 Fix: Filter based on the 'include' column from the data editor
            accepted_specs = edited_df[edited_df["include"] == True][["feature", "value"]].to_dict(orient="records")
            st.session_state["creator_data"]["accepted_specs"] = accepted_specs
            st.success(f"Confirmed base of {len(accepted_specs)} features.")

    # ================= STEP 2: BENEFIT HIERARCHY (V13 Logic) =================
    if st.session_state["creator_data"].get("accepted_specs"):
        st.markdown("---")
        st.header("Step 2: Benefit Hierarchy (BH)")
        brand_voice = st.text_input("Brand Voice / Tone", value="Premium, reliable, confident, yet accessible.", key="cr_voice")
        keywords = audit_data.get("keywords", [])
        
        # V13 Logic: Generate BH Button -> then show editor
        if st.button("🧠 Generate Benefit Hierarchy"):
            with st.spinner("AI is mapping features to benefits..."):
                user_content = json.dumps({"canonical_specs": st.session_state["creator_data"]["accepted_specs"], "keywords": keywords[:50]})
                bh_data = call_llm_json(client, SYSTEM_BH_PROMPT, user_content, temp=0.1, model="gpt-4o-mini")
                
                if bh_data and isinstance(bh_data, list):
                    st.session_state["creator_data"]["bh_df"] = pd.DataFrame(bh_data)
                    st.success("BH Blueprint generated.")
                else: st.error("AI generation failed.")

        if "bh_df" in st.session_state["creator_data"]:
            st.subheader("Review BH Blueprint")
            edited_bh_df = st.data_editor(st.session_state["creator_data"]["bh_df"], num_rows="dynamic", key="cr_bh_editor", hide_index=True)
            # V13 Logic: Confirm BH Button
            if st.button("✅ Confirm Blueprint"):
                st.session_state["creator_data"]["final_bh"] = edited_bh_df.to_dict(orient="records")
                st.success("Blueprint confirmed.")

    # ================= STEP 3: LISTING & A+ BRIEFS (V20 Logic) =================
    if st.session_state["creator_data"].get("final_bh"):
        st.markdown("---")
        st.header("Step 3: Final Listing & A+ Design Briefs")
        
        facts_to_include = audit_data.get("factual_claims", [])
        banned_list = audit_data.get("banned_words_found", [])
        has_aplus = audit_data.get("has_aplus", False)

        if facts_to_include:
            st.info(f"ℹ️ AI will include mandatory facts: {', '.join(facts_to_include)}")
        
        # V21 Update: Notify user about strict filters
        if banned_list:
            st.warning(f"⛔ STRICT SAFETY FILTER ACTIVE: The following words will be forcibly removed/replaced: {', '.join(banned_list)}")
        
        task_desc = "Generate optimized Amazon listing text."
        if not has_aplus:
             task_desc += " AND generate 5 A+ module design briefs."
             st.info("ℹ️ A+ Content suggestions will be generated since none were detected.")
        else:
             st.warning("ℹ️ Skipping A+ Content generation as existing A+ was noted.")

        if st.button("✍️ Generate Optimized Content", type="primary"):
             with st.spinner(f"AI is optimizing content ({'including A+ briefs' if not has_aplus else 'text only'})... Wait ~60-90s..."):
                 keywords = audit_data.get("keywords", [])
                 
                 # FIX APPLIED: HARDENING KEYWORD LIST
                 # Remove banned words from the target keyword list so the AI isn't tempted to integrate them.
                 if banned_list:
                     original_kw_count = len(keywords)
                     keywords = [
                         k for k in keywords 
                         if not any(b.lower() in k.lower() for b in banned_list)
                     ]
                     if len(keywords) < original_kw_count:
                         st.toast(f"⚠️ Filtered {original_kw_count - len(keywords)} unsafe keywords before sending to AI.", icon="🛡️")

                 # V20 Logic Fix: Using the new AGGRESSIVE prompt
                 final_system_prompt = BASE_LISTING_PROMPT
                 if not has_aplus:
                     final_system_prompt += APLUS_INSTRUCTIONS
                 final_system_prompt += "\nDo not include markdown formatting like ```json at the start or end."

                 user_content = json.dumps({
                     "task": task_desc,
                     "benefit_hierarchy": st.session_state["creator_data"]["final_bh"],
                     "keywords": keywords[:75],
                     "brand_voice": brand_voice,
                     "audit_report_to_address": {
                         "opportunities_to_fix": audit_data.get("audit_opportunities", []),
                         "strengths_to_maintain": audit_data.get("audit_strengths", [])
                     },
                     "factual_claims_to_include": facts_to_include,
                     "constraints": {
                         "title_length_range": "180-200 characters",
                         "title_kw_count": "Top 3-5 essential keywords",
                         "bullet_style": "Narrative, benefit-driven paragraphs with facts.",
                         "star_feature_requirement": "Ensure one bullet highlights a 'star feature'.",
                         "forbidden_words": banned_list # V21: Passed as explicit constraint
                     }
                 })
                 listing = call_llm_json(client, final_system_prompt, user_content, temp=0.3, model="gpt-4o")
                 if listing and "title" in listing:
                     st.session_state["creator_data"]["final_listing"] = listing
                     st.success("Optimized draft generated!")
                 else: st.error("AI generation failed.")

        if "final_listing" in st.session_state["creator_data"]:
            listing = st.session_state["creator_data"]["final_listing"]

            st.subheader("1. Text Listing Preview")
            with st.container(border=True):
                st.markdown(f"**Title:** {listing.get('title','')}")
                st.markdown("**Bullets:**")
                for b in listing.get("bullets", []): st.markdown(f"- {b}")
                with st.expander("View Description (HTML)"):
                    st.code(listing.get("description",""), language="html")
                    st.markdown(listing.get("description",""), unsafe_allow_html=True)
                st.markdown(f"**Backend Keywords:** {', '.join(listing.get('backend_search_terms',[]))}")

            st.subheader("2. Refine Text Listing")
            refine_instr = st.text_input("Enter instruction for AI editor")
            
            # V20 Fix: Refine button logic to preserve A+ modules
            if st.button("🤖 Apply Refinement"):
                if not refine_instr: st.warning("Need instructions.")
                else:
                    with st.spinner("AI editor is revising..."):
                        # V20 Fix: Send the ENTIRE current listing (including A+ modules) to the AI
                        current_listing_json = json.dumps(listing)
                        user_content = json.dumps({
                            "current_listing": current_listing_json,
                            "user_instruction": refine_instr
                        })
                        
                        # V20 Fix: Temp=0.1 for precise changes
                        refined_listing = call_llm_json(client, SYSTEM_REFINE_PROMPT, user_content, temp=0.1, model="gpt-4o")
                        
                        if refined_listing and "title" in refined_listing:
                            # V20 Fix: The refined JSON should now contain aplus_modules if they were in the input
                            st.session_state["creator_data"]["final_listing"] = refined_listing
                            st.rerun()
                        else: st.error("Refinement failed. The AI couldn't generate valid JSON or failed to follow the preservation instruction.")

            aplus_modules = listing.get("aplus_modules", [])
            if aplus_modules and not has_aplus:
                st.markdown("---")
                st.subheader("3. A+ Content Design Briefs")
                st.caption("Hand these module concepts and visual prompts to your graphic designer.")
                
                for i, module in enumerate(aplus_modules):
                    with st.container(border=True):
                        md_type = module.get('type','standard').replace('_',' ').title()
                        st.markdown(f"### Module {i+1}: {md_type} - {module.get('headline','')}")
                        st.write(module.get('body',''))
                        st.info(f"🎨 **Designer Prompt:** {module.get('image_prompt_idea', 'N/A')}")
            elif has_aplus:
                 st.markdown("---")
                 st.info("A+ Content briefs were skipped as existing A+ was noted in the audit.")

            # ================= RESCORING SECTION (V13 Logic) =================
            st.markdown("---")
            st.header("🏆 Final Validation: Rescore")
            if st.button("📊 Rescore New Listing Now", type="primary", use_container_width=True):
                new_title = listing.get("title","")
                new_bullets = listing.get("bullets", [])
                new_desc_clean = BeautifulSoup(listing.get("description",""), "html.parser").get_text(" ")
                original_kws = audit_data.get("keywords", [])
                original_img_counts = audit_data.get("img_counts", (1,1,1,0))

                auditor.client = client
                with st.spinner("Calculating new LQI..."):
                    new_result = auditor.audit_listing(new_title, new_bullets, new_desc_clean, original_kws, original_img_counts)
                
                st.subheader("Improvement Report")
                col1, col2 = st.columns(2)
                initial_lqi = audit_data.get("initial_lqi", 0)
                new_lqi = new_result["LQI"]
                col1.metric("Original LQI", initial_lqi)
                col2.metric("New LQI", new_lqi, delta=round(new_lqi - initial_lqi, 1))

                scores = new_result["scores"]
                df_radar = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,10], title="New LQI Pillar Analysis")
                fig.update_traces(fill='toself', line_color='#2EA398')
                st.plotly_chart(fig, use_container_width=True)

                if new_lqi > 8.5: st.balloons(); st.success("Optimization complete! High-performing asset ready.")
                elif new_lqi > initial_lqi: st.success("Solid improvement.")
                else: st.warning("Score did not improve significantly.")

# ============================================================
# 6. MAIN APP ROUTER WITH LANDING PAGE
# ============================================================
if st.session_state["on_landing_page"]:
    render_landing_page()
elif st.session_state["app_mode"] == "Audit":
    render_audit_module()
else:
    render_creator_module()
