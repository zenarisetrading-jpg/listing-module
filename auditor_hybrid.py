import re
import json
import math
from collections import Counter

class HybridAuditor:
    def __init__(self, client):
        """
        client: OpenAI client (or compatible)
        """
        self.client = client

    # ============================================================
    # 1. HARD LOGIC (Rule-Based)
    # ============================================================

    def _score_structure(self, title, bullets, description):
        """Math-based structure scoring"""
        # Targets
        TARGET_BULLET_COUNT = 5
        TARGET_BULLET_LEN = 80 # Soft target, flexible
        TARGET_DESC_LEN = 400
        
        b_count = len(bullets)
        
        # Average bullet length (prevent div by zero)
        avg_b_len = sum(len(b) for b in bullets) / max(1, b_count)
        desc_len = len(description)

        # Formula from your model
        score = (
            0.4 * min(1.0, b_count / TARGET_BULLET_COUNT) +
            0.35 * min(1.0, avg_b_len / TARGET_BULLET_LEN) + # Cap at 1.0 so long bullets don't break math
            0.25 * min(1.0, desc_len / TARGET_DESC_LEN)
        )
        return min(1.0, round(score, 2))

    def _check_hard_compliance(self, text):
        """Checks for instant-fail words"""
        banned = ["cure", "treat", "prevent", "diagnose", "cancer", "covid"]
        text_lower = text.lower()
        for word in banned:
            if f" {word} " in text_lower:
                return 0.0, [word]
        return 1.0, []

    def _extract_numeric_claims(self, text):
        """Regex extraction for the 'Numeric Validation' layer"""
        # Patterns for: 24 hours, 500ml, 100%, 18/8, etc.
        patterns = [
            r'\d+\s?(?:hours|hrs|h)', 
            r'\d+\s?(?:ml|oz|l|gal)', 
            r'\d+(?:%| percent)', 
            r'\d+/\d+', # Fractions like 18/8
            r'\d+\s?(?:cm|mm|inch|")',
            r'\d+\s?(?:g|kg|lbs)'
        ]
        claims = []
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            claims.extend(matches)
        return list(set(claims))

    # ============================================================
    # 2. SOFT LOGIC (AI / Human Perspective)
    # ============================================================

    def _analyze_with_llm(self, title, bullets, description, keywords, brand_voice):
        """
        Asks the AI to grade Persuasion, Relevance, and Soft Compliance.
        Returns a JSON object.
        """
        
        listing_text = f"Title: {title}\nBullets: {bullets}\nDescription: {description}"
        
        system_prompt = """
        You are a ruthless Senior Amazon Editor. You grade listings based on human psychology and sales potential.
        
        Analyze the listing below and return a JSON object with these scores (0.00 to 1.00):
        
        1. relevance_score: Do they use the provided keywords naturally? Is it clearly written for the target customer?
        2. persuasion_score: Does it trigger desire? Does it use sensory language? Is it benefit-forward?
        3. soft_compliance_score: Is the tone safe? (0.0 = risky/scammy, 1.0 = professional/safe).
        
        Also return:
        - missing_keywords: List of provided keywords NOT found or used poorly.
        - critique: A 1-sentence summary of why you gave these scores.
        """
        
        user_prompt = json.dumps({
            "listing_content": listing_text,
            "target_keywords": keywords,
            "target_voice": brand_voice
        })

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Fast, cheap, good at semantic grading
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # Fallback if AI fails
            return {
                "relevance_score": 0.5, 
                "persuasion_score": 0.5, 
                "soft_compliance_score": 1.0, 
                "missing_keywords": [],
                "critique": "AI Analysis Failed."
            }

    # ============================================================
    # 3. THE MASTER AUDIT FUNCTION
    # ============================================================

    def audit_listing(self, title, bullets, description, target_keywords, supplier_specs=""):
        """
        Combines Hard Rules + AI Perspective into the Final LQI.
        """
        full_text = f"{title} {' '.join(bullets)} {description}"
        
        # --- STEP 1: HARD RULES ---
        structure_score = self._score_structure(title, bullets, description)
        
        # SEO Math (Title Length + Keyword Density)
        # We keep your formula: 0.55 * title_score + 0.45 * relevance (AI handled relevance)
        title_len_score = min(1.0, len(title) / 140) 
        
        # Hard Compliance Check
        hard_comp_score, banned_words = self._check_hard_compliance(full_text)

        # Numeric Claims Extraction
        found_claims = self._extract_numeric_claims(full_text)
        
        # --- STEP 2: AI PERSPECTIVE ---
        ai_result = self._analyze_with_llm(title, bullets, description, target_keywords, "Premium")
        
        # --- STEP 3: SCORING MERGE ---
        
        # (A) Relevance (AI)
        relevance_score = ai_result.get("relevance_score", 0.5)
        
        # (B) SEO Strength (Hybrid)
        # Hybrid: Math for length, AI for keyword usage relevance
        seo_score = (0.55 * title_len_score) + (0.45 * relevance_score)
        
        # (C) Persuasion (AI)
        persuasion_score = ai_result.get("persuasion_score", 0.5)
        
        # (E) Compliance (Hybrid)
        # If hard compliance fails (banned word), score is 0. Otherwise use AI's soft score.
        final_compliance = 0.0 if hard_comp_score == 0 else ai_result.get("soft_compliance_score", 1.0)

        # --- STEP 4: FINAL LQI CALCULATION ---
        # Formula: ((relevance + persuasion + seo + structure + compliance) / 5) * 10
        
        raw_avg = (relevance_score + persuasion_score + seo_score + structure_score + final_compliance) / 5
        lqi = round(raw_avg * 10, 2)

        # --- STEP 5: CLAIM VALIDATION (Simple Match) ---
        # Check found claims against supplier text (simple substring match for now)
        unsupported_claims = []
        if supplier_specs:
            for claim in found_claims:
                if claim not in supplier_specs: # Very strict, might need fuzzy match later
                    unsupported_claims.append(claim)

        return {
            "LQI": lqi,
            "breakdown": {
                "Relevance": round(relevance_score * 10, 1),
                "SEO": round(seo_score * 10, 1),
                "Persuasion": round(persuasion_score * 10, 1),
                "Structure": round(structure_score * 10, 1),
                "Compliance": round(final_compliance * 10, 1)
            },
            "details": {
                "ai_critique": ai_result.get("critique"),
                "banned_words_found": banned_words,
                "numeric_claims_extracted": found_claims,
                "potentially_unsupported_claims": unsupported_claims
            }
        }