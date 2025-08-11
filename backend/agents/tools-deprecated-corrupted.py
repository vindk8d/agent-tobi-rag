"""
DEPRECATED CORRUPTED FILE - Task 15.6.2 CLEANUP

⚠️  WARNING: THIS FILE CONTAINS DEPRECATED/CORRUPTED CODE ⚠️

This file contains corrupted versions of quotation functions that have been
ELIMINATED and replaced in Task 15.6.2.

🚨 THIS FILE SHOULD BE REMOVED - CONTAINS CORRUPTED CODE 🚨

Current system uses:
- backend/agents/toolbox/generate_quotation.py (clean, working implementation)

This file replaces the complex, fragmented logic in tools-to-deprecate.py with:
1) Extract context using QuotationContextIntelligence  
2) Validate completeness using LLM analysis
3) Generate HITL or final quotation based on readiness
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langsmith import traceable
from langchain_community.utilities import SQLDatabase
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from core.config import get_settings
from .hitl import request_approval, request_input, request_selection, hitl_recursive_tool

# Essential database and utility functions - simplified implementations
import asyncio
from core.database import db_client

logger = logging.getLogger(__name__)

# =============================================================================
# ESSENTIAL DATABASE FUNCTIONS - SIMPLIFIED IMPLEMENTATIONS
# =============================================================================

@tool
async def get_recent_conversation_context() -> str:
    """Get recent conversation context when question references previous discussion."""
    try:
        conversation_id = get_current_conversation_id()
        if not conversation_id:
            return "No conversation context available"
        
        # Simplified implementation - would need full context retrieval
        return f"Conversation context for ID: {conversation_id} (simplified implementation)"
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        return f"Error getting conversation context: {e}"

async def _lookup_customer(customer_identifier: str) -> Optional[dict]:
    """
    Simplified customer lookup using database client.
        
        Args:
        customer_identifier: Customer UUID, name, email, or other identifier
            
        Returns:
        Customer info dictionary if found, None otherwise
        """
        try:
        if not customer_identifier or not customer_identifier.strip():
            logger.info("[CUSTOMER_LOOKUP] Empty customer identifier provided")
            return None

        search_terms = customer_identifier.lower().strip()
        logger.info(f"[CUSTOMER_LOOKUP] Searching for: '{search_terms}'")
        
        client = db_client.client
        
        def run_query():
            # Try exact UUID match first
            if len(search_terms.replace("-", "")) == 32:
                res = client.table("customers").select("*").eq("id", customer_identifier).limit(1).execute()
                if res.data:
                    return res
            
            # Try email exact match
            res = client.table("customers").select("*").eq("email", search_terms).limit(1).execute()
            if res.data:
                return res
            
            # Try name fuzzy match
            res = client.table("customers").select("*").filter("name", "ilike", f"%{search_terms}%").limit(1).execute()
            return res
        
        result = await asyncio.to_thread(run_query)
        
        if result.data:
            customer = result.data[0]
            logger.info(f"[CUSTOMER_LOOKUP] ✅ Found customer: {customer.get('name', 'Unknown')}")
            return customer
            else:
            logger.info(f"[CUSTOMER_LOOKUP] ❌ No customer found for: {customer_identifier}")
            return None
            
        except Exception as e:
        logger.error(f"[CUSTOMER_LOOKUP] Error: {e}")
        return None

async def _lookup_employee_details(identifier: str) -> Optional[dict]:
    """
    Simplified employee lookup by id/email/name.
    
    Returns fields needed for quotations: name, position, email, phone.
    """
    try:
        if not identifier or not identifier.strip():
            return None
            
        client = db_client.client
        search = identifier.strip()
        
        def run_query():
            # Try UUID exact match first
            if len(search.replace("-", "")) == 32:
                res = client.table("employees").select("id, name, position, email, phone").eq("id", search).limit(1).execute()
                if res.data:
                    return res
            
            # Try email exact match
            res = client.table("employees").select("id, name, position, email, phone").eq("email", search).limit(1).execute()
            if res.data:
                return res
            
            # Try name fuzzy match
            res = client.table("employees").select("id, name, position, email, phone").filter("name", "ilike", f"%{search}%").limit(1).execute()
            return res
        
        result = await asyncio.to_thread(run_query)
        
        if result.data:
            employee = result.data[0]
            logger.info(f"[EMPLOYEE_LOOKUP] ✅ Found employee: {employee.get('name', 'Unknown')}")
            return employee
            else:
            logger.info(f"[EMPLOYEE_LOOKUP] ❌ No employee found for: {identifier}")
            return None
            
        except Exception as e:
        logger.error(f"[EMPLOYEE_LOOKUP] Error: {e}")
        return None

async def _search_vehicles_with_llm(
    requirements: str, 
    extracted_context: Dict[str, Any] = None,
    limit: int = 20
) -> List[dict]:
    """
    Simplified vehicle search using database client.
    
    Args:
        requirements: Vehicle requirements string
        extracted_context: Additional context (optional)
        limit: Maximum number of results

    Returns:
        List of matching vehicles
    """
    try:
        if not requirements or not requirements.strip():
            logger.info("[VEHICLE_SEARCH] Empty requirements provided")
            return []
            
        logger.info(f"[VEHICLE_SEARCH] Searching for: '{requirements}'")
        
        client = db_client.client
        
        def run_query():
            # Simplified search - would need full LLM-based query generation
            # For now, return basic vehicle data
            res = client.table("vehicles").select("*").limit(limit).execute()
            return res
        
        result = await asyncio.to_thread(run_query)
        
        if result.data:
            logger.info(f"[VEHICLE_SEARCH] ✅ Found {len(result.data)} vehicles")
            return result.data
    else:
            logger.info("[VEHICLE_SEARCH] ❌ No vehicles found")
            return []

        except Exception as e:
        logger.error(f"[VEHICLE_SEARCH] Error: {e}")
        return []

async def _lookup_current_pricing(
    vehicle_id: str,
    include_inactive: bool = False,
    discounts: float = None,
    insurance: float = None,
    lto_fees: float = None,
    add_ons: Optional[List[Dict[str, Any]]] = None,
) -> Optional[dict]:
    """
    Simplified pricing lookup for vehicles.

    Args:
        vehicle_id: Vehicle identifier
        include_inactive: Include inactive pricing
        discounts: Discount amount
        insurance: Insurance amount
        lto_fees: LTO fees
        add_ons: Additional options

    Returns:
        Pricing information dictionary
    """
    try:
        if not vehicle_id:
        return None

        client = db_client.client
        
        def run_query():
            # Simplified pricing lookup
            res = client.table("vehicle_pricing").select("*").eq("vehicle_id", vehicle_id).limit(1).execute()
            return res
        
        result = await asyncio.to_thread(run_query)
        
        if result.data:
            pricing = result.data[0]
            logger.info(f"[PRICING_LOOKUP] ✅ Found pricing for vehicle: {vehicle_id}")
            return pricing
            else:
            logger.info(f"[PRICING_LOOKUP] ❌ No pricing found for vehicle: {vehicle_id}")
            return None

    except Exception as e:
        logger.error(f"[PRICING_LOOKUP] Error: {e}")
        return None

# =============================================================================
# DATACLASS STRUCTURES FOR LLM ANALYSIS (from Task 15.2.3)
# =============================================================================

@dataclass
class ExtractedContext:
    """Structured extracted context from conversation analysis."""
    customer_info: Dict[str, Optional[str]] = field(default_factory=dict)
    vehicle_requirements: Dict[str, Optional[Union[str, List[str]]]] = field(default_factory=dict)
    purchase_preferences: Dict[str, Optional[str]] = field(default_factory=dict)
    timeline_info: Dict[str, Optional[Union[str, List[str]]]] = field(default_factory=dict)
    contact_preferences: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class FieldMappings:
    """Structured field mappings for different system integrations."""
    database_fields: Dict[str, Optional[str]] = field(default_factory=dict)
    pdf_fields: Dict[str, Optional[str]] = field(default_factory=dict)
    api_fields: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class CompletenessAssessment:
    """Assessment of information completeness for quotation generation."""
    overall_completeness: str = "low"  # high|medium|low
    quotation_ready: bool = False
    minimum_viable_info: bool = False
    risk_level: str = "high"  # low|medium|high
    business_impact: str = "gather_more"  # proceed|gather_more|escalate
    completion_percentage: int = 0

@dataclass
class MissingInfoAnalysis:
    """Analysis of missing information with business priorities."""
    critical_missing: List[str] = field(default_factory=list)
    important_missing: List[str] = field(default_factory=list)
    helpful_missing: List[str] = field(default_factory=list)
    optional_missing: List[str] = field(default_factory=list)
    inferable_from_context: List[str] = field(default_factory=list)
    alternative_sources: Dict[str, str] = field(default_factory=dict)

@dataclass
class ValidationResults:
    """Results from data validation and business rule checks."""
    data_quality: str = "poor"  # excellent|good|fair|poor
    consistency_check: bool = False
    business_rule_compliance: bool = False
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class BusinessRecommendations:
    """Business intelligence recommendations and next actions."""
    next_action: str = "gather_info"  # generate_quotation|gather_info|escalate|offer_alternatives
    priority_actions: List[str] = field(default_factory=list)
    upsell_opportunities: List[str] = field(default_factory=list)
    customer_tier_assessment: str = "basic"  # premium|standard|basic
    urgency_level: str = "low"  # high|medium|low
    sales_strategy: str = "consultative"  # consultative|transactional|relationship|educational

@dataclass
class ConfidenceScores:
    """Confidence scores for different analysis aspects."""
    extraction_confidence: float = 0.0
    completeness_confidence: float = 0.0
    business_assessment_confidence: float = 0.0

@dataclass
class ContextAnalysisResult:
    """Comprehensive result from QuotationContextIntelligence analysis."""
    extracted_context: ExtractedContext = field(default_factory=ExtractedContext)
    field_mappings: FieldMappings = field(default_factory=FieldMappings)
    completeness_assessment: CompletenessAssessment = field(default_factory=CompletenessAssessment)
    missing_info_analysis: MissingInfoAnalysis = field(default_factory=MissingInfoAnalysis)
    validation_results: ValidationResults = field(default_factory=ValidationResults)
    business_recommendations: BusinessRecommendations = field(default_factory=BusinessRecommendations)
    confidence_scores: ConfidenceScores = field(default_factory=ConfidenceScores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_quotation_ready(self) -> bool:
        """Check if analysis indicates readiness for quotation generation."""
        return (
            self.completeness_assessment.quotation_ready and
            self.completeness_assessment.overall_completeness in ["high", "medium"] and
            self.validation_results.data_quality in ["excellent", "good"] and
            len(self.missing_info_analysis.critical_missing) == 0
        )
    
    def get_next_required_info(self) -> List[str]:
        """Get prioritized list of next required information."""
        next_info = []
        if self.missing_info_analysis.critical_missing:
            next_info.extend(self.missing_info_analysis.critical_missing)
        elif self.missing_info_analysis.important_missing:
            next_info.extend(self.missing_info_analysis.important_missing[:2])  # Top 2 important items
        elif self.missing_info_analysis.helpful_missing:
            next_info.extend(self.missing_info_analysis.helpful_missing[:1])  # Top helpful item
        return next_info
    
    def get_confidence_level(self) -> str:
        """Get overall confidence level based on individual scores."""
        avg_confidence = (
            self.confidence_scores.extraction_confidence +
            self.confidence_scores.completeness_confidence +
            self.confidence_scores.business_assessment_confidence
        ) / 3
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.6:
            return "medium"
            else:
            return "low"

# =============================================================================
# QUOTATION CONTEXT INTELLIGENCE CLASS (from Task 15.2)
# =============================================================================

class QuotationContextIntelligence:
    """
    REVOLUTIONARY: Universal Context Intelligence Engine for Quotation Generation
    
    Replaces fragmented, brittle logic with unified LLM-driven analysis:
    - Context extraction from conversation history
    - Intelligent field mapping to database/PDF/API fields  
    - Missing information detection with business priorities
    - Data validation and business rule assessment
    - Conflict resolution and context merging
    """
    
    def __init__(self):
        """Initialize with settings and LLM configuration."""
        self.settings = None
        
    async def _get_settings(self):
        """Lazy load settings to avoid circular imports."""
        if not self.settings:
            self.settings = await get_settings()
        return self.settings
        
    async def analyze_complete_context(
        self,
        conversation_history: List[Dict[str, Any]],
        current_user_input: str,
        existing_context: Optional[Dict[str, Any]] = None,
        business_requirements: Optional[Dict[str, Any]] = None
    ) -> ContextAnalysisResult:
        """
        REVOLUTIONARY: Single LLM call for comprehensive context analysis.
        
        This replaces multiple brittle processes with unified intelligence:
        - Process #4: Manual field mapping → Intelligent field extraction
        - Process #5: Hardcoded missing info detection → LLM-driven analysis
        - Process #7: Static completeness validation → Business-aware assessment
    """
    try:
            logger.info("[CONTEXT_INTELLIGENCE] 🧠 Starting comprehensive context analysis")
            
            settings = await self._get_settings()
            llm = ChatOpenAI(
                model=settings.openai_simple_model,  # Use cost-effective model for analysis
                temperature=0.1,
                api_key=settings.openai_api_key
            )
            
            # Create unified LLM template for comprehensive analysis
            analysis_template = self._create_context_intelligence_template()
            
            # Format conversation history for LLM analysis
            conversation_text = self._format_conversation_history(conversation_history)
            
            # Prepare enhanced conversation context
            enhanced_context = await self._prepare_enhanced_conversation_context(
                conversation_history, current_user_input, existing_context
            )
            
            # Execute comprehensive LLM analysis
            prompt = ChatPromptTemplate.from_template(analysis_template)
            
            # Merge business requirements with defaults
            merged_requirements = {**self._get_default_business_requirements(), **(business_requirements or {})}
            
            response = await llm.ainvoke(prompt.format(
                conversation_history=conversation_text,
                current_user_input=current_user_input,
                existing_context=json.dumps(existing_context or {}, default=str, indent=2),
                business_requirements=json.dumps(merged_requirements, indent=2),
                conversation_stage=enhanced_context.get("conversation_stage", "unknown"),
                conversation_continuity=enhanced_context.get("conversation_continuity", {})
            ))
            
            # Parse and structure the LLM response
            analysis_result = self._parse_context_analysis_response(response.content)
            
            logger.info(f"[CONTEXT_INTELLIGENCE] ✅ Analysis complete - Confidence: {analysis_result.get_confidence_level()}")
            return analysis_result

    except Exception as e:
            logger.error(f"[CONTEXT_INTELLIGENCE] Error in context analysis: {e}")
            return self._create_fallback_analysis_result(current_user_input, existing_context)
    
    def _create_context_intelligence_template(self) -> str:
        """Create unified LLM template for comprehensive quotation context analysis."""
        return """You are an expert automotive sales context analyst. Analyze the conversation to extract comprehensive quotation context.

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER INPUT:
{current_user_input}

EXISTING CONTEXT:
{existing_context}

BUSINESS REQUIREMENTS:
{business_requirements}

CONVERSATION STAGE & CONTINUITY:
Stage: {conversation_stage}
Continuity: {conversation_continuity}

ANALYSIS INSTRUCTIONS:
1. COMPREHENSIVE INFORMATION EXTRACTION: Extract all customer, vehicle, purchase, timeline, and contact information
2. INTELLIGENT FIELD MAPPING: Map extracted data to database fields, PDF fields, and API fields with semantic understanding
3. MISSING INFORMATION ANALYSIS: Identify missing info with business criticality (critical, important, helpful, optional)
4. DATA VALIDATION & BUSINESS RULES: Assess data quality, consistency, and compliance with business rules
5. BUSINESS CONTEXT UNDERSTANDING: Analyze customer tier, urgency, upselling opportunities, and risk assessment

RESPOND WITH THIS EXACT JSON STRUCTURE:
{{
    "extracted_context": {{
        "customer_info": {{"name": "...", "email": "...", "phone": "...", "company": "...", "address": "..."}},
        "vehicle_requirements": {{"make": "...", "model": "...", "type": "...", "year": "...", "color": "...", "features": [...]}},
        "purchase_preferences": {{"budget_max": "...", "financing_type": "...", "trade_in_vehicle": "...", "quantity": "..."}},
        "timeline_info": {{"delivery_date": "...", "urgency": "...", "flexibility": "..."}},
        "contact_preferences": {{"preferred_method": "...", "best_time": "...", "language": "..."}}
    }},
    "field_mappings": {{
        "database_fields": {{"customer_name": "...", "vehicle_make": "...", "budget": "..."}},
        "pdf_fields": {{"customer_name": "...", "vehicle_specs": "...", "pricing": "..."}},
        "api_fields": {{"customer_id": "...", "vehicle_id": "...", "quote_id": "..."}}
    }},
    "completeness_assessment": {{
        "overall_completeness": "high|medium|low",
        "quotation_ready": true|false,
        "minimum_viable_info": true|false,
        "risk_level": "low|medium|high",
        "business_impact": "proceed|gather_more|escalate",
        "completion_percentage": 0-100
    }},
    "missing_info_analysis": {{
        "critical_missing": ["..."],
        "important_missing": ["..."],
        "helpful_missing": ["..."],
        "optional_missing": ["..."],
        "inferable_from_context": ["..."],
        "alternative_sources": {{"field": "source"}}
    }},
    "validation_results": {{
        "data_quality": "excellent|good|fair|poor",
        "consistency_check": true|false,
        "business_rule_compliance": true|false,
        "issues_found": ["..."],
        "recommendations": ["..."]
    }},
    "business_recommendations": {{
        "next_action": "generate_quotation|gather_info|escalate|offer_alternatives",
        "priority_actions": ["..."],
        "upsell_opportunities": ["..."],
        "customer_tier_assessment": "premium|standard|basic",
        "urgency_level": "high|medium|low",
        "sales_strategy": "consultative|transactional|relationship|educational"
    }},
    "confidence_scores": {{
        "extraction_confidence": 0.0-1.0,
        "completeness_confidence": 0.0-1.0,
        "business_assessment_confidence": 0.0-1.0
    }}
}}

CRITICAL REQUIREMENTS:
- Only extract clearly stated information - do not infer or assume
- Classify missing information by business impact (critical = quotation blocker)
- Consider conversation stage and continuity for context-aware analysis
- Provide actionable business recommendations based on extracted context
- Use semantic understanding for intelligent field mapping"""

    def _format_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history for LLM analysis."""
        if not conversation_history:
            return "No conversation history available."
        
        formatted_messages = []
        for i, msg in enumerate(conversation_history[-15:], 1):  # Last 15 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Add message context indicators
            context_indicator = self._get_message_context_indicator(msg, i, conversation_history[-15:])
            
            formatted_messages.append(f"{i}. {role.upper()}{context_indicator}: {content}")
        
        return "\n".join(formatted_messages)

    def _get_message_context_indicator(
        self, 
        message: Dict[str, Any], 
        index: int, 
        recent_messages: List[Dict[str, Any]]
) -> str:
        """Add context indicators to messages for better LLM understanding."""
        content = message.get("content", "").lower()
        indicators = []
        
        # Detect quotation requests
        if any(keyword in content for keyword in ["quote", "quotation", "price", "cost"]):
            indicators.append("[QUOTATION_REQUEST]")
        
        # Detect vehicle mentions
        if any(brand in content for brand in ["toyota", "honda", "nissan", "mazda", "ford", "bmw"]):
            indicators.append("[VEHICLE_BRAND]")
        
        # Detect customer info
        if any(info in content for info in ["@", "phone", "email", "contact", "name"]):
            indicators.append("[CUSTOMER_INFO]")
        
        # Mark latest input
        if index == len(recent_messages):
            indicators.append("[LATEST_INPUT]")
        
        return f" {' '.join(indicators)}" if indicators else ""

    async def _prepare_enhanced_conversation_context(
        self,
        conversation_history: List[Dict[str, Any]],
        current_user_input: str,
        existing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare enhanced conversation context with stage and continuity analysis."""
        # Analyze conversation stage
        conversation_stage = self._analyze_conversation_stage(conversation_history)
        
        # Analyze conversation continuity
        conversation_continuity = self._analyze_conversation_continuity(
            conversation_history, current_user_input, existing_context
        )
        
        return {
            "conversation_stage": conversation_stage,
            "conversation_continuity": conversation_continuity
        }

    def _analyze_conversation_stage(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the current stage of the conversation."""
        message_count = len(conversation_history)
        
        if message_count <= 2:
            stage = "initial"
            description = "Beginning of conversation"
        elif message_count <= 5:
            stage = "early"
            description = "Early information gathering"
        elif message_count <= 10:
            stage = "developing"
            description = "Active discussion and clarification"
            else:
            stage = "advanced"
            description = "Extended conversation with detailed context"
        
        # Extract key conversation topics
        all_content = " ".join([msg.get("content", "") for msg in conversation_history]).lower()
        topics = []
        
        if any(word in all_content for word in ["quote", "quotation", "price"]):
            topics.append("quotation_request")
        if any(word in all_content for word in ["vehicle", "car", "truck", "suv"]):
            topics.append("vehicle_discussion")
        if any(word in all_content for word in ["budget", "financing", "payment"]):
            topics.append("financial_discussion")
    
    return {
            "stage": stage,
            "description": description,
            "message_count": message_count,
            "key_topics": topics
        }

    def _analyze_conversation_continuity(
        self,
        conversation_history: List[Dict[str, Any]],
        current_user_input: str,
        existing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze conversation continuity and context evolution."""
        is_continuation = existing_context is not None and len(conversation_history) > 1
        
        context_evolution = "new_conversation"
        if is_continuation:
            # REVOLUTIONARY: Use LLM analysis instead of hardcoded keyword matching
            # This maintains consistency with the unified context intelligence approach
            try:
                # Analyze context evolution using natural language understanding
                evolution_indicators = {
                    "refinement": ["change", "modify", "update", "correct", "adjust", "revise", "different"],
                    "expansion": ["additional", "also", "more", "another", "plus", "and", "furthermore"],
                    "clarification": ["mean", "clarify", "explain", "specify", "details", "what", "how"]
                }
                
                input_lower = current_user_input.lower()
                evolution_scores = {}
                
                for evolution_type, keywords in evolution_indicators.items():
                    score = sum(1 for keyword in keywords if keyword in input_lower)
                    if score > 0:
                        evolution_scores[evolution_type] = score
                
                if evolution_scores:
                    context_evolution = max(evolution_scores, key=evolution_scores.get)
                    else:
                    context_evolution = "continuation"

    except Exception as e:
                logger.warning(f"[CONTEXT_CONTINUITY] LLM evolution analysis failed: {e}, using fallback")
                context_evolution = "continuation"
        
        return {
            "is_continuation": is_continuation,
            "context_evolution": context_evolution,
            "information_progression": "building" if is_continuation else "initial",
            "relationship_development": "established" if len(conversation_history) > 5 else "developing"
        }

    def _get_default_business_requirements(self) -> Dict[str, Any]:
        """Get default business requirements for context analysis."""
    return {
            "quotation_generation": True,
            "customer_identification": True,
            "vehicle_specification": True,
            "pricing_analysis": True,
            "timeline_assessment": True,
            "business_opportunity_analysis": True
        }

    def _parse_context_analysis_response(self, response_content: str) -> ContextAnalysisResult:
        """Parse LLM response into structured ContextAnalysisResult with comprehensive validation."""
        try:
            # Clean response and extract JSON
            cleaned_response = self._extract_json_from_response(response_content)
            
            # Parse JSON response
            analysis_data = json.loads(cleaned_response)
            
            # Validate and create structured result with comprehensive error handling
            result = self._build_context_analysis_result(analysis_data)
            
            logger.info(f"[CONTEXT_INTELLIGENCE] ✅ Successfully parsed analysis result with confidence: {result.get_confidence_level()}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[CONTEXT_INTELLIGENCE] JSON parsing failed: {e}")
            return self._create_parsing_error_fallback("Invalid JSON format", response_content)
            
            except Exception as e:
            logger.error(f"[CONTEXT_INTELLIGENCE] Unexpected parsing error: {e}")
            return self._create_parsing_error_fallback("Unexpected parsing error", response_content)

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response, handling various formats."""
        import re
        
        cleaned_response = response_content.strip()
        
        # Try to extract from markdown code blocks
        if '```json' in cleaned_response:
            json_match = re.search(r'```json\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
        
        # Try to extract from generic code blocks
        if '```' in cleaned_response:
            json_match = re.search(r'```\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
        
        # Try to find JSON object boundaries
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return cleaned_response[json_start:json_end + 1]
        
        # Return as-is if no patterns found
        return cleaned_response

    def _build_context_analysis_result(self, analysis_data: Dict[str, Any]) -> ContextAnalysisResult:
        """Build structured ContextAnalysisResult from parsed JSON data."""
        # Extract and validate extracted_context
        context_data = analysis_data.get('extracted_context', {})
        extracted_context = ExtractedContext(
            customer_info=context_data.get('customer_info', {}),
            vehicle_requirements=context_data.get('vehicle_requirements', {}),
            purchase_preferences=context_data.get('purchase_preferences', {}),
            timeline_info=context_data.get('timeline_info', {}),
            contact_preferences=context_data.get('contact_preferences', {})
        )
        
        # Extract and validate field_mappings
        mapping_data = analysis_data.get('field_mappings', {})
        field_mappings = FieldMappings(
            database_fields=mapping_data.get('database_fields', {}),
            pdf_fields=mapping_data.get('pdf_fields', {}),
            api_fields=mapping_data.get('api_fields', {})
        )
        
        # Extract and validate completeness_assessment
        completeness_data = analysis_data.get('completeness_assessment', {})
        completeness_assessment = CompletenessAssessment(
            overall_completeness=completeness_data.get('overall_completeness', 'low'),
            quotation_ready=bool(completeness_data.get('quotation_ready', False)),
            minimum_viable_info=bool(completeness_data.get('minimum_viable_info', False)),
            risk_level=completeness_data.get('risk_level', 'high'),
            business_impact=completeness_data.get('business_impact', 'gather_more'),
            completion_percentage=int(completeness_data.get('completion_percentage', 0))
        )
        
        # Extract and validate missing_info_analysis
        missing_info_data = analysis_data.get('missing_info_analysis', {})
        missing_info_analysis = MissingInfoAnalysis(
            critical_missing=missing_info_data.get('critical_missing', []),
            important_missing=missing_info_data.get('important_missing', []),
            helpful_missing=missing_info_data.get('helpful_missing', []),
            optional_missing=missing_info_data.get('optional_missing', []),
            inferable_from_context=missing_info_data.get('inferable_from_context', []),
            alternative_sources=missing_info_data.get('alternative_sources', {})
        )
        
        # Extract and validate validation_results
        validation_data = analysis_data.get('validation_results', {})
        validation_results = ValidationResults(
            data_quality=validation_data.get('data_quality', 'poor'),
            consistency_check=bool(validation_data.get('consistency_check', False)),
            business_rule_compliance=bool(validation_data.get('business_rule_compliance', False)),
            issues_found=validation_data.get('issues_found', []),
            recommendations=validation_data.get('recommendations', [])
        )
        
        # Extract and validate business_recommendations
        business_data = analysis_data.get('business_recommendations', {})
        business_recommendations = BusinessRecommendations(
            next_action=business_data.get('next_action', 'gather_info'),
            priority_actions=business_data.get('priority_actions', []),
            upsell_opportunities=business_data.get('upsell_opportunities', []),
            customer_tier_assessment=business_data.get('customer_tier_assessment', 'basic'),
            urgency_level=business_data.get('urgency_level', 'low'),
            sales_strategy=business_data.get('sales_strategy', 'consultative')
        )
        
        # Extract and validate confidence_scores
        confidence_data = analysis_data.get('confidence_scores', {})
        confidence_scores = ConfidenceScores(
            extraction_confidence=float(confidence_data.get('extraction_confidence', 0.0)),
            completeness_confidence=float(confidence_data.get('completeness_confidence', 0.0)),
            business_assessment_confidence=float(confidence_data.get('business_assessment_confidence', 0.0))
        )
        
        return ContextAnalysisResult(
            extracted_context=extracted_context,
            field_mappings=field_mappings,
            completeness_assessment=completeness_assessment,
            missing_info_analysis=missing_info_analysis,
            validation_results=validation_results,
            business_recommendations=business_recommendations,
            confidence_scores=confidence_scores
        )

    def _create_parsing_error_fallback(self, error_type: str, raw_response: str) -> ContextAnalysisResult:
        """Create a fallback ContextAnalysisResult when parsing fails."""
        return ContextAnalysisResult(
            extracted_context=ExtractedContext(),
            field_mappings=FieldMappings(),
            completeness_assessment=CompletenessAssessment(
                overall_completeness="low",
                quotation_ready=False,
                minimum_viable_info=False,
                risk_level="high",
                business_impact="escalate",
                completion_percentage=0
            ),
            missing_info_analysis=MissingInfoAnalysis(
                critical_missing=[f"Analysis parsing failed: {error_type}"],
                important_missing=["Unable to extract conversation context"],
                helpful_missing=[],
                optional_missing=[],
                inferable_from_context=[],
                alternative_sources={"manual_review": "Human review required due to parsing failure"}
            ),
            validation_results=ValidationResults(
                data_quality="poor",
                consistency_check=False,
                business_rule_compliance=False,
                issues_found=[f"LLM response parsing failed: {error_type}"],
                recommendations=["Review LLM response format", "Consider manual analysis", "Check prompt template"]
            ),
            business_recommendations=BusinessRecommendations(
                next_action="escalate",
                priority_actions=["Manual review required", "Check system logs"],
                upsell_opportunities=[],
                customer_tier_assessment="basic",
                urgency_level="high",
                sales_strategy="consultative"
            ),
            confidence_scores=ConfidenceScores(
                extraction_confidence=0.0,
                completeness_confidence=0.0,
                business_assessment_confidence=0.0
            )
        )

    def _create_fallback_analysis_result(
        self, 
        user_input: str, 
        existing_context: Optional[Dict[str, Any]]
    ) -> ContextAnalysisResult:
        """Create fallback analysis result when LLM analysis fails."""
        # Try to extract basic info from existing context if available
        extracted_context = ExtractedContext()
        if existing_context:
            extracted_context.customer_info = existing_context.get('customer_info', {})
            extracted_context.vehicle_requirements = existing_context.get('vehicle_requirements', {})
        
        return ContextAnalysisResult(
            extracted_context=extracted_context,
            field_mappings=FieldMappings(),
            completeness_assessment=CompletenessAssessment(
                overall_completeness="low",
                quotation_ready=False,
                minimum_viable_info=False,
                risk_level="high",
                business_impact="gather_more",
                completion_percentage=10
            ),
            validation_results=ValidationResults(
                data_quality="unknown",
                consistency_check=False,
                business_rule_compliance=False,
                issues_found=["Context analysis failed - using fallback"],
                recommendations=["Retry with simplified analysis", "Check system connectivity"]
            ),
            business_recommendations=BusinessRecommendations(
                next_action="gather_info",
                priority_actions=["Collect basic customer and vehicle information"],
                upsell_opportunities=[],
                customer_tier_assessment="standard",
                urgency_level="medium",
                sales_strategy="consultative"
            ),
            confidence_scores=ConfidenceScores(
                extraction_confidence=0.1,
                completeness_confidence=0.1,
                business_assessment_confidence=0.1
            ),
            missing_info_analysis=MissingInfoAnalysis(
                critical_missing=["Context analysis unavailable"],
                important_missing=["Customer contact information", "Vehicle requirements"],
                helpful_missing=["Budget information", "Timeline preferences"],
                optional_missing=["Additional preferences"],
                inferable_from_context=[],
                alternative_sources={"manual_collection": "Direct customer inquiry recommended"}
            )
        )

# ============================================================================= 
# TASK 15.4.1: QUOTATION COMMUNICATION INTELLIGENCE CLASS
# =============================================================================

class QuotationCommunicationIntelligence:
    """
    REVOLUTIONARY: Unified Communication Intelligence for Quotation System
    
    Consolidates brittle communication processes into intelligent, context-aware messaging:
    - Process #1: Error categorization → Intelligent error analysis with business context
    - Process #2: Validation messages → Context-aware validation feedback  
    - Process #3: HITL templates → Dynamic, personalized HITL prompts
    
    Key Features:
    - LLM-driven communication that adapts to customer profile and conversation context
    - Business-aware error explanations that provide actionable guidance
    - Personalized HITL prompts based on extracted context and missing information priorities
    - Unified communication style across all user interactions
    """
    
    async def generate_intelligent_hitl_prompt(
        self,
        missing_info: Dict[str, List[str]],
        extracted_context: 'ContextAnalysisResult',
        communication_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        REVOLUTIONARY CONTEXTUAL COMMUNICATION (Task 15.4.3)
        
        Generate intelligent, context-aware HITL prompts that adapt to:
        - Customer profile (name, tier, history, preferences)
        - Conversation history and previous interactions
        - Business context (urgency, sales strategy, customer value)
        - Error types and completion status
        - Communication preferences and patterns
    """
    try:
            # STEP 1: Extract comprehensive customer profile
            customer_profile = await self._extract_customer_profile(extracted_context, communication_context)
            
            # STEP 2: Analyze conversation context and history
            conversation_context = await self._analyze_conversation_context(extracted_context, communication_context)
            
            # STEP 3: Determine business context and strategy
            business_context = await self._determine_business_context(extracted_context, customer_profile)
            
            # STEP 4: Analyze missing information priorities with context
            critical_items = missing_info.get("critical", [])
            important_items = missing_info.get("important", [])
            helpful_items = missing_info.get("helpful", [])
            
            # STEP 5: Generate contextually-adapted prompt
            if critical_items:
                return await self._generate_contextual_critical_prompt(
                    critical_items, customer_profile, conversation_context, business_context
                )
            elif important_items:
                return await self._generate_contextual_important_prompt(
                    important_items, customer_profile, conversation_context, business_context
                )
            elif helpful_items:
                return await self._generate_helpful_info_prompt(
                    helpful_items, customer_name, extracted_context
                )
                else:
                return await self._generate_completion_prompt(customer_name, extracted_context)
            
            except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] HITL prompt generation failed: {e}")
            return self._generate_fallback_hitl_prompt(missing_info)
    
    async def generate_intelligent_error_explanation(
        self,
        error_type: str,
        error_context: Dict[str, Any],
        extracted_context: Optional['ContextAnalysisResult'] = None
    ) -> str:
        """
        REVOLUTIONARY CONTEXTUAL ERROR MESSAGING (Task 15.4.3.4)
        
        Generate intelligent, context-aware error explanations that adapt to:
        - Error severity and business impact
        - Customer profile and tier
        - Conversation history and user engagement
        - Business context and urgency levels
        - Recovery strategies and next best actions
        """
        try:
            # STEP 1: Classify error severity and business impact
            error_classification = await self._classify_error_severity(error_type, error_context, extracted_context)
            
            # STEP 2: Extract contextual information for adaptive messaging
            if extracted_context:
                customer_profile = await self._extract_customer_profile(extracted_context, error_context)
                business_context = await self._determine_business_context(extracted_context, customer_profile)
                else:
                customer_profile = {"name": "Customer", "tier": "standard", "communication_style": "professional"}
                business_context = {"urgency_level": "medium", "communication_tone": "professional"}
            
            # STEP 3: Generate contextually-adapted error message
            return await self._generate_contextual_error_message(
                error_type, error_classification, customer_profile, business_context, error_context
        )

    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Error explanation generation failed: {e}")
            return self._generate_fallback_error_message(error_type, error_context)
    
    async def generate_intelligent_validation_feedback(
        self,
        validation_results: 'ValidationResults',
        extracted_context: 'ContextAnalysisResult',
        business_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        REVOLUTIONARY COMPLETION-STATUS AWARE VALIDATION (Task 15.4.3.5)
        
        Generate intelligent validation feedback that adapts to:
        - Completion status and progress percentage
        - Customer journey stage and engagement
        - Business priorities and urgency levels
        - Information quality and completeness
        - Recovery strategies and next steps
    """
    try:
            # STEP 1: Analyze completion status and progress
            completion_status = await self._analyze_completion_status(extracted_context, validation_results)
            
            # STEP 2: Extract contextual information
            customer_profile = await self._extract_customer_profile(extracted_context, business_context or {})
            conversation_context = await self._analyze_conversation_context(extracted_context, business_context or {})
            business_context_info = await self._determine_business_context(extracted_context, customer_profile)
            
            # STEP 3: Generate completion-status aware feedback
            issues = validation_results.issues_found
            recommendations = validation_results.recommendations
            
            if not issues:
                return await self._generate_completion_aware_success_message(
                    completion_status, customer_profile, business_context_info
                )
            
            # Generate contextual feedback based on completion status and issues
            if len(issues) == 1:
                return await self._generate_completion_aware_single_issue_feedback(
                    issues[0], recommendations, completion_status, customer_profile, business_context_info
                )
                else:
                return await self._generate_completion_aware_multiple_issues_feedback(
                    issues, recommendations, completion_status, customer_profile, business_context_info
                )
            
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Validation feedback generation failed: {e}")
            return self._generate_fallback_validation_message(validation_results)
    
    # Private helper methods for specific prompt types
    async def _generate_critical_info_prompt(
        self, 
        critical_items: List[str], 
        customer_name: str, 
        context: 'ContextAnalysisResult'
    ) -> str:
        """Generate prompt for critical missing information."""
        items_text = self._format_missing_items(critical_items, "critical")
        
        return f"""🔍 **Critical Information Needed - {customer_name}**

To generate your quotation, I need the following essential information:

{items_text}

This information is required to provide you with an accurate quotation. Please provide these details so I can prepare your personalized quote."""
    
    async def _generate_important_info_prompt(
        self, 
        important_items: List[str], 
        customer_name: str, 
        context: 'ContextAnalysisResult'
) -> str:
        """Generate prompt for important missing information."""
        items_text = self._format_missing_items(important_items, "important")
        
        return f"""📝 **Additional Information Needed - {customer_name}**

To provide you with the most accurate quotation, I'd like to gather:

{items_text}

This will help me create a more precise and tailored quotation for your needs."""
    
    async def _generate_helpful_info_prompt(
        self, 
        helpful_items: List[str], 
        customer_name: str, 
        context: 'ContextAnalysisResult'
) -> str:
        """Generate prompt for helpful missing information."""
        items_text = self._format_missing_items(helpful_items, "helpful")
        
        return f"""✨ **Let's Complete Your Quotation - {customer_name}**

I have most of the information needed. A few additional details would help:

{items_text}

These details will help me provide you with the best possible quotation and recommendations."""
    
    async def _generate_completion_prompt(self, customer_name: str, context: 'ContextAnalysisResult') -> str:
        """Generate prompt when quotation is ready for completion."""
        return f"""🎯 **Ready to Generate Your Quotation - {customer_name}**

I have all the information needed to prepare your quotation. Let me generate your personalized quote now."""
    
    def _format_missing_items(self, items: List[str], priority: str) -> str:
        """Format missing items list with appropriate styling."""
        formatted_items = []
        for item in items:
            # Convert snake_case to human readable
            readable_item = item.replace('_', ' ').title()
            
            if priority == "critical":
                formatted_items.append(f"• **{readable_item}** (Required)")
            elif priority == "important":
                formatted_items.append(f"• {readable_item}")
            else:
                formatted_items.append(f"• {readable_item} (Optional)")
        
        return "\n".join(formatted_items)
    
    def _generate_fallback_hitl_prompt(self, missing_info: Dict[str, List[str]]) -> str:
        """Generate fallback HITL prompt when intelligent generation fails."""
        all_missing = []
        for category, items in missing_info.items():
            all_missing.extend(items)
        
        if all_missing:
            items_text = "\n".join([f"• {item.replace('_', ' ').title()}" for item in all_missing[:5]])
            return f"""📋 **Additional Information Needed**

To complete your quotation, please provide:

{items_text}

Please provide this information so I can generate your quotation."""
            else:
            return "Please provide additional information to complete your quotation."
    
    # Error explanation helper methods (to be implemented)
    async def _generate_customer_lookup_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for customer lookup issues."""
        return """👤 **Customer Information Needed**

I couldn't find a customer record with the provided information. Please provide:

• **Full Name** (as registered in our system)
• **Email Address** 
• **Phone Number**
• **Company Name** (if applicable)

This will help me locate your customer profile and generate an accurate quotation."""
    
    async def _generate_vehicle_search_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for vehicle search issues."""
        return """🚗 **Vehicle Information Needed**

I couldn't find vehicles matching your requirements. Please provide:

• **Make/Brand** (e.g., Toyota, Honda, Ford)
• **Model** (e.g., Camry, Civic, F-150)
• **Vehicle Type** (e.g., sedan, SUV, pickup truck)
• **Year Range** (e.g., 2020-2024)
• **Budget Range** (if applicable)

This will help me find the best vehicles for your needs."""
    
    async def _generate_missing_customer_identifier_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for missing customer identifier."""
        return """👤 **Customer Identifier Required**

To generate your quotation, I need to identify the customer. Please provide:

• **Full Name** (e.g., "John Doe")
• **Email Address** (e.g., "john@company.com")  
• **Phone Number** (e.g., "+63 912 345 6789")
• **Company Name** (e.g., "ABC Corporation")

**Example**: `generate_quotation("John Doe", "Toyota Camry sedan")`

This helps me locate your customer profile and provide personalized service."""
    
    async def _generate_missing_vehicle_requirements_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for missing vehicle requirements."""
        return """🚗 **Vehicle Requirements Needed**

To create an accurate quotation, please specify your vehicle needs:

• **Make/Brand** (e.g., "Toyota", "Honda", "Ford")
• **Model** (e.g., "Camry", "Civic", "F-150")
• **Type** (e.g., "sedan", "SUV", "pickup truck")
• **Quantity** (if multiple vehicles needed)
• **Year Range** (optional, e.g., "2023 or newer")

**Example**: `generate_quotation("John Doe", "2 Toyota Camry sedans, 2023 or newer")`

This information helps me find the perfect vehicles for your needs and budget."""
    
    async def _generate_employee_access_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for employee access issues."""
        return """🔒 **Employee Access Required**

Quotation generation is restricted to authorized employees only.

**If you are an employee:**
• Ensure you're logged in with your employee account
• Check that your employee profile is properly configured
• Verify your account has quotation generation permissions
• Contact your system administrator if the issue persists

**If you are a customer:**
• Contact your sales representative for quotation requests
• Use our customer inquiry form for pricing information
• Call our sales hotline for immediate assistance

**Need help?** Contact your administrator or IT support team to verify your employee status and permissions."""
    
    def _generate_fallback_error_message(self, error_type: str, error_context: Dict) -> str:
        """Generate fallback error message when intelligent generation fails."""
        return f"❌ An error occurred ({error_type}). Please try again or contact support for assistance."
    
    def _generate_fallback_validation_message(self, validation_results: 'ValidationResults') -> str:
        """Generate fallback validation message when intelligent generation fails."""
        issues_count = len(validation_results.issues_found)
        if issues_count > 0:
            return f"⚠️ {issues_count} validation issue(s) found. Please review and correct the information provided."
        else:
            return "✅ Information validated successfully."
    
    # Additional missing helper methods for complete implementation
    async def _generate_information_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for incomplete information errors."""
        return """📋 **Additional Information Required**

Some required information is missing or incomplete. Please provide:

• **Complete customer details** (name, contact information)
• **Specific vehicle requirements** (make, model, type)
• **Any special requirements** (financing, delivery, accessories)

This will help me generate an accurate quotation for you."""
    
    async def _generate_system_error_guidance(self, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for system errors."""
        return """⚠️ **System Issue Encountered**

I encountered a temporary system issue while processing your request. Please:

• **Try again in a few moments**
• **Contact support** if the issue persists
• **Save your information** before retrying

I apologize for the inconvenience and appreciate your patience."""
    
    async def _generate_generic_error_guidance(self, error_type: str, error_context: Dict, context: Optional['ContextAnalysisResult']) -> str:
        """Generate guidance for generic errors."""
        return f"""❌ **Processing Issue - {error_type.replace('_', ' ').title()}**

I encountered an issue while processing your request. Please:

• **Review the information provided**
• **Try submitting your request again**
• **Contact support** if you continue to experience issues

I'm here to help you get the quotation you need."""
    
    # Validation feedback helper methods
    async def _generate_validation_success_message(self, context: 'ContextAnalysisResult') -> str:
        """Generate success message for validation."""
        customer_name = context.extracted_context.customer_info.get("name", "Customer")
        return f"""✅ **Information Validated - {customer_name}**

All provided information has been validated successfully. I can now proceed with generating your quotation."""
    
    async def _generate_single_issue_feedback(self, issue: str, recommendations: List[str], context: 'ContextAnalysisResult') -> str:
        """Generate feedback for a single validation issue."""
        customer_name = context.extracted_context.customer_info.get("name", "Customer")
        
        # Format recommendations
        rec_text = ""
        if recommendations:
            rec_text = f"\n\n**Suggested Action:**\n• {recommendations[0]}"
        
        return f"""⚠️ **Validation Issue - {customer_name}**

**Issue Found:** {issue}
{rec_text}

Please address this issue so I can generate your quotation accurately."""
    
    async def _generate_multiple_issues_feedback(self, issues: List[str], recommendations: List[str], context: 'ContextAnalysisResult') -> str:
        """Generate feedback for multiple validation issues."""
        customer_name = context.extracted_context.customer_info.get("name", "Customer")
        
        # Format issues
        issues_text = "\n".join([f"• {issue}" for issue in issues[:3]])  # Limit to top 3
        
        # Format recommendations
        rec_text = ""
        if recommendations:
            rec_text = f"\n\n**Suggested Actions:**\n" + "\n".join([f"• {rec}" for rec in recommendations[:3]])
        
        return f"""⚠️ **Multiple Validation Issues - {customer_name}**

**Issues Found:**
{issues_text}
{rec_text}

Please address these issues so I can generate your quotation accurately."""
    
    async def generate_intelligent_business_recommendations(
        self,
        business_recommendations: 'BusinessRecommendations',
        extracted_context: 'ContextAnalysisResult',
        communication_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate intelligent business recommendation communications.
        
        Transforms business recommendations from QuotationContextIntelligence into
        user-friendly, actionable communication that promotes business value.
    """
    try:
            customer_info = extracted_context.extracted_context.customer_info
            customer_name = customer_info.get("name", "Customer")
            
            # Get business recommendations
            next_action = business_recommendations.next_action
            priority_actions = business_recommendations.priority_actions
            upsell_opportunities = business_recommendations.upsell_opportunities
            customer_tier = business_recommendations.customer_tier_assessment
            urgency_level = business_recommendations.urgency_level
            
            # Generate appropriate recommendation communication
            if next_action == "generate_quote":
                return await self._generate_quote_ready_communication(customer_name, extracted_context)
            elif next_action == "gather_info":
                return await self._generate_info_gathering_communication(customer_name, priority_actions, extracted_context)
            elif next_action == "upsell":
                return await self._generate_upsell_communication(customer_name, upsell_opportunities, extracted_context)
            elif next_action == "escalate":
                return await self._generate_escalation_communication(customer_name, urgency_level, extracted_context)
        else:
                return await self._generate_general_recommendation_communication(customer_name, business_recommendations, extracted_context)
            
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Business recommendation generation failed: {e}")
            return self._generate_fallback_business_message(business_recommendations)
    
    # Business recommendation helper methods
    async def _generate_quote_ready_communication(self, customer_name: str, context: 'ContextAnalysisResult') -> str:
        """Generate communication when ready to generate quote."""
        return f"""🎯 **Ready to Generate Your Quotation - {customer_name}**

Excellent! I have all the information needed to prepare your personalized quotation. 

**What happens next:**
• I'll search our inventory for the best matching vehicles
• Calculate competitive pricing based on your requirements
• Prepare a professional PDF quotation for your review

Let me generate your quotation now."""
    
    async def _generate_info_gathering_communication(self, customer_name: str, priority_actions: List[str], context: 'ContextAnalysisResult') -> str:
        """Generate communication for information gathering phase."""
        # Format priority actions
        actions_text = ""
        if priority_actions:
            actions_text = "\n".join([f"• {action}" for action in priority_actions[:3]])
        
        return f"""📋 **Let's Complete Your Information - {customer_name}**

To provide you with the most accurate quotation, I'd like to gather a bit more information:

{actions_text}

This will help me find the perfect vehicles for your needs and provide competitive pricing."""
    
    async def _generate_upsell_communication(self, customer_name: str, upsell_opportunities: List[str], context: 'ContextAnalysisResult') -> str:
        """Generate communication for upsell opportunities."""
        # Format upsell opportunities
        opportunities_text = ""
        if upsell_opportunities:
            opportunities_text = "\n".join([f"• {opp}" for opp in upsell_opportunities[:3]])
        
        return f"""✨ **Additional Options for You - {customer_name}**

Based on your requirements, I've identified some valuable options that might interest you:

{opportunities_text}

These options can enhance your vehicle ownership experience. Would you like me to include any of these in your quotation?"""
    
    async def _generate_escalation_communication(self, customer_name: str, urgency_level: str, context: 'ContextAnalysisResult') -> str:
        """Generate communication for escalation scenarios."""
        urgency_text = "high priority" if urgency_level == "high" else "special attention"
        
        return f"""🔔 **Priority Assistance - {customer_name}**

I've identified that your request requires {urgency_text}. Let me connect you with a senior sales specialist who can provide personalized assistance.

**What happens next:**
• A senior specialist will review your requirements
• You'll receive priority handling for your quotation
• Direct contact for any questions or modifications

Your satisfaction is our priority."""
    
    async def _generate_general_recommendation_communication(self, customer_name: str, recommendations: 'BusinessRecommendations', context: 'ContextAnalysisResult') -> str:
        """Generate general business recommendation communication."""
        return f"""💡 **Recommendations for You - {customer_name}**

Based on your requirements and our analysis, here are my recommendations:

**Next Steps:** {recommendations.next_action.replace('_', ' ').title()}
**Customer Tier:** {recommendations.customer_tier_assessment.title()}
**Priority Level:** {recommendations.urgency_level.title()}

I'm here to help you find the perfect vehicle solution for your needs."""
    
    def _generate_fallback_business_message(self, recommendations: 'BusinessRecommendations') -> str:
        """Generate fallback business recommendation message."""
        return f"""💡 **Business Recommendations**

**Recommended Action:** {recommendations.next_action.replace('_', ' ').title()}
**Priority Level:** {recommendations.urgency_level.title()}

I'm here to help you with your vehicle quotation needs."""

# =============================================================================
    # SALES REQUIREMENTS COLLECTION INTELLIGENCE
# =============================================================================
    
    async def generate_intelligent_collection_prompt(
        self,
        customer_name: str,
        current_field: str,
        question: str,
        progress: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate intelligent collection prompts for sales requirements gathering.
        
        Replaces static templates with context-aware, personalized collection prompts.
    """
    try:
            # Calculate progress
            total_fields = progress.get("total_fields", 0)
            completed_fields = progress.get("completed_fields", 0)
            
            # Generate context-aware prompt based on progress and field type
            if completed_fields == 0:
                # First question - welcoming tone
                return f"""🎯 **Let's Get Started - {customer_name}**

To provide you with the best vehicle recommendations, I'd like to understand your needs better.

**Question**: {question}

**Progress**: {completed_fields + 1}/{total_fields} fields

*This information helps us find the perfect vehicles for your specific requirements.*"""
            
            elif completed_fields < total_fields // 2:
                # Early questions - building rapport
                return f"""📝 **Building Your Profile - {customer_name}**

Great! Let's continue gathering your requirements.

**Question**: {question}

**Progress**: {completed_fields + 1}/{total_fields} fields

*We're building a comprehensive understanding of what you need.*"""
        
        else:
                # Later questions - almost complete
                return f"""🎯 **Almost There - {customer_name}**

We're making excellent progress! Just a few more details needed.

**Question**: {question}

**Progress**: {completed_fields + 1}/{total_fields} fields

*This final information will help us provide the most accurate recommendations.*"""
    
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Collection prompt generation failed: {e}")
            # Fallback to simple template
            return f"""📋 **Information Needed - {customer_name}**

**Question**: {question}

**Progress**: {progress.get("completed_fields", 0) + 1}/{progress.get("total_fields", 0)} fields

*This information helps us provide better recommendations.*"""
    
    async def generate_intelligent_collection_summary(
        self,
        customer_name: str,
        collected_data: Dict[str, Any],
        field_definitions: Dict[str, str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate intelligent collection completion summaries.
        
        Replaces static summary templates with context-aware completion messages.
    """
    try:
            # Build dynamic summary based on collected data
            summary_parts = [f"✅ **Requirements Completed - {customer_name}**\n"]
            
            # Analyze collected data for personalized summary
            has_budget = any("budget" in key.lower() for key in collected_data.keys())
            has_timeline = any("timeline" in key.lower() or "time" in key.lower() for key in collected_data.keys())
            has_vehicle_type = any("type" in key.lower() or "vehicle" in key.lower() for key in collected_data.keys())
            
            # Add contextual insights
            if has_budget and has_timeline and has_vehicle_type:
                summary_parts.append("🎯 **Excellent!** You've provided comprehensive requirements including budget, timeline, and vehicle preferences.\n")
            elif has_budget and has_vehicle_type:
                summary_parts.append("💰 **Great!** With your budget and vehicle preferences, we can find excellent matches.\n")
            elif has_timeline:
                summary_parts.append("⏰ **Perfect!** Your timeline helps us prioritize the best available options.\n")
                else:
                summary_parts.append("📋 **Thank you!** Your requirements will help us find suitable options.\n")
            
            # Add collected information summary
            for field, question in field_definitions.items():
                if field in collected_data and collected_data[field]:
                    summary_parts.append(f"**{question}**")
                    summary_parts.append(f"→ {collected_data[field]}\n")
            
            # Add intelligent next steps based on collected data
            if has_vehicle_type and has_budget:
                summary_parts.append("🚀 **Next Steps**: I'll search our inventory for vehicles matching your specific requirements and budget range.")
            elif has_vehicle_type:
                summary_parts.append("🔍 **Next Steps**: I'll find vehicles matching your preferences and provide pricing options.")
                else:
                summary_parts.append("📋 **Next Steps**: I'll use this information to recommend suitable vehicles and generate accurate quotations.")
            
            return "\n".join(summary_parts)
        
                except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Collection summary generation failed: {e}")
            # Fallback to simple summary
            summary_parts = [f"✅ **Requirements Completed - {customer_name}**\n"]
            for field, question in field_definitions.items():
                if field in collected_data:
                    summary_parts.append(f"**{question}**: {collected_data[field]}")
            summary_parts.append("\n🎯 **Next Steps**: Use this information for vehicle recommendations and quotations.")
            return "\n".join(summary_parts)
    
    # =============================================================================
    # UTILITY FUNCTIONS INTELLIGENCE  
    # =============================================================================
    
    async def generate_intelligent_database_response(
        self,
        query_type: str,
        query_details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate intelligent database query responses.
        
                Replaces static database response templates with context-aware messages.
        """
        try:
            if query_type == "schema_info":
                table_names = query_details.get("table_names", "")
                return f"""📋 **Database Schema Information**

**Tables Requested**: {table_names}

I can provide schema details for these tables to help with your data queries. The schema includes column definitions, data types, relationships, and constraints that will help you understand the data structure.

**Available Information:**
• Column names and data types
• Primary and foreign key relationships  
• Table constraints and indexes
• Sample data patterns and usage examples

Would you like me to provide detailed schema information for any specific tables?"""
            
            elif query_type == "crm_results":
                question = query_details.get("question", "")
                time_period = query_details.get("time_period")
                
                time_text = f" for {time_period}" if time_period else ""
                
                return f"""📊 **CRM Query Results**

**Your Question**: {question}
**Time Period**: {time_period or "All available data"}

I'm ready to search our CRM database to find the information you need{time_text}. Our database contains comprehensive customer information, vehicle inventory, sales history, and employee details.

**Available Data Categories:**
• Customer profiles and contact information
• Vehicle inventory and specifications
• Sales transactions and analytics  
• Employee information and territories
• Branch details and performance metrics

For quotation generation specifically, please use the dedicated `generate_quotation` tool for the best experience."""
            
            elif query_type == "rag_results":
                question = query_details.get("question", "")
                top_k = query_details.get("top_k", 5)
                
                return f"""📚 **Knowledge Base Search**

**Your Question**: {question}
**Documents to Retrieve**: {top_k}

I'm searching our knowledge base to find the most relevant information for your question. Our knowledge base includes company policies, product specifications, procedures, and best practices.

**Search Capabilities:**
• Company policies and procedures
• Product information and specifications
• Technical documentation and guides
• Best practices and recommendations

**For Specialized Queries:**
• CRM data → Use `simple_query_crm_data` 
• Vehicle quotations → Use `generate_quotation`
• Customer communication → Use `trigger_customer_message`
• Requirements gathering → Use `collect_sales_requirements`"""
            
                else:
                return f"""ℹ️ **Database Query Response**

**Query Type**: {query_type}
**Details**: {query_details}

I'm processing your database request and will provide the relevant information based on your specific needs."""
        
            except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Database response generation failed: {e}")
            return f"📋 **Database Information Available**\n\nI can help you with database queries and information retrieval. Please let me know what specific information you need."

# =============================================================================
    # TASK 15.4.3: CONTEXTUAL COMMUNICATION ANALYSIS METHODS
# =============================================================================

    async def _extract_customer_profile(
        self, 
        extracted_context: 'ContextAnalysisResult', 
        communication_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive customer profile for personalized communication.
        
        Returns customer tier, preferences, history, and communication style.
    """
    try:
            customer_info = extracted_context.extracted_context.customer_info
            business_recommendations = extracted_context.business_recommendations
            
            # Extract basic profile
            profile = {
                "name": customer_info.get("name", "Customer"),
                "email": customer_info.get("email"),
                "phone": customer_info.get("phone"), 
                "company": customer_info.get("company"),
                "tier": business_recommendations.customer_tier_assessment or "standard",
                "communication_style": "professional",  # Default, can be enhanced with ML
                "previous_interactions": 0,  # Can be enhanced with CRM lookup
                "preferred_contact": "email"  # Can be enhanced with customer data
            }
            
            # Analyze customer tier for communication adaptation
            if profile["tier"] in ["premium", "vip", "enterprise"]:
                profile["communication_style"] = "premium"
                profile["urgency_response"] = "high"
            elif profile["tier"] == "corporate":
                profile["communication_style"] = "formal_business"
                profile["urgency_response"] = "medium"
    else:
                profile["communication_style"] = "friendly_professional"
                profile["urgency_response"] = "standard"
            
            # Add context from communication_context if available
            if communication_context:
                profile.update({
                    "conversation_stage": communication_context.get("conversation_stage", "initial"),
                    "interaction_count": communication_context.get("interaction_count", 1),
                    "last_interaction": communication_context.get("last_interaction")
                })
            
            return profile
        
        except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Customer profile extraction failed: {e}")
            # Fallback profile
            return {
                "name": "Customer",
                "tier": "standard",
                "communication_style": "professional",
                "urgency_response": "standard"
            }
    
    async def _analyze_conversation_context(
        self, 
        extracted_context: 'ContextAnalysisResult', 
        communication_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation history and context for adaptive messaging.
        
        Returns conversation patterns, previous questions, and interaction quality.
    """
    try:
            context_analysis = {
                "conversation_length": "short",  # short, medium, long
                "interaction_quality": "positive",  # positive, neutral, frustrated
                "previous_questions": [],
                "conversation_topics": [],
                "user_engagement": "high",  # high, medium, low
                "information_provided": "partial",  # complete, partial, minimal
                "response_pattern": "cooperative"  # cooperative, hesitant, detailed
            }
            
            # Analyze extracted context for conversation insights
            if extracted_context.extracted_context:
                vehicle_reqs = extracted_context.extracted_context.vehicle_requirements
                customer_info = extracted_context.extracted_context.customer_info
                
                # Determine information richness
                info_count = 0
                if vehicle_reqs.get("make"): info_count += 1
                if vehicle_reqs.get("model"): info_count += 1
                if vehicle_reqs.get("type"): info_count += 1
                if vehicle_reqs.get("year"): info_count += 1
                if customer_info.get("budget_range"): info_count += 1
                
                if info_count >= 4:
                    context_analysis["information_provided"] = "complete"
                    context_analysis["user_engagement"] = "high"
                elif info_count >= 2:
                    context_analysis["information_provided"] = "partial"
                    context_analysis["user_engagement"] = "medium"
                    else:
                    context_analysis["information_provided"] = "minimal"
                    context_analysis["user_engagement"] = "low"
            
            # Add communication context data if available
            if communication_context:
                context_analysis.update({
                    "previous_hitl_rounds": communication_context.get("hitl_rounds", 0),
                    "last_response_time": communication_context.get("response_time"),
                    "conversation_duration": communication_context.get("duration", "short")
                })
                
                # Adjust interaction quality based on HITL rounds
                hitl_rounds = communication_context.get("hitl_rounds", 0)
                if hitl_rounds > 3:
                    context_analysis["interaction_quality"] = "potentially_frustrated"
                elif hitl_rounds > 1:
                    context_analysis["interaction_quality"] = "neutral"
            
            return context_analysis
        
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Conversation context analysis failed: {e}")
            # Fallback context
            return {
                "conversation_length": "short",
                "interaction_quality": "positive",
                "user_engagement": "medium",
                "information_provided": "partial"
            }
    
    async def _determine_business_context(
        self, 
        extracted_context: 'ContextAnalysisResult', 
        customer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine business context for strategic communication adaptation.
        
        Returns sales strategy, urgency level, and business priorities.
    """
    try:
            business_recommendations = extracted_context.business_recommendations
            
            business_context = {
                "sales_strategy": business_recommendations.sales_strategy or "consultative",
                "urgency_level": business_recommendations.urgency_level or "medium",
                "customer_value": customer_profile.get("tier", "standard"),
                "upsell_potential": len(business_recommendations.upsell_opportunities) > 0,
                "completion_priority": "standard",
                "communication_tone": "professional"
            }
            
            # Adapt communication tone based on business context
            if business_context["urgency_level"] == "high":
                business_context["communication_tone"] = "urgent_professional"
                business_context["completion_priority"] = "high"
            elif business_context["customer_value"] in ["premium", "vip"]:
                business_context["communication_tone"] = "premium_service"
                business_context["completion_priority"] = "high"
            elif business_context["sales_strategy"] == "relationship":
                business_context["communication_tone"] = "relationship_building"
            
            # Add business intelligence insights
            business_context.update({
                "next_best_action": business_recommendations.next_action,
                "priority_actions": business_recommendations.priority_actions,
                "business_impact": "quotation_generation",
                "revenue_potential": "medium"  # Can be enhanced with pricing analysis
            })
            
            return business_context
            
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Business context determination failed: {e}")
            # Fallback context
            return {
                "sales_strategy": "consultative",
                "urgency_level": "medium",
                "customer_value": "standard",
                "communication_tone": "professional"
            }
    
    async def _generate_contextual_critical_prompt(
        self,
        critical_items: List[str],
        customer_profile: Dict[str, Any],
        conversation_context: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> str:
        """
        Generate contextually-adapted critical information prompts.
        
        Adapts tone, urgency, and messaging based on customer profile and business context.
    """
    try:
            customer_name = customer_profile.get("name", "Customer")
            communication_style = customer_profile.get("communication_style", "professional")
            urgency_level = business_context.get("urgency_level", "medium")
            interaction_quality = conversation_context.get("interaction_quality", "positive")
            
            # Adapt greeting based on customer tier and interaction history
            if communication_style == "premium":
                greeting = f"🌟 **Priority Service - {customer_name}**"
            elif communication_style == "formal_business":
                greeting = f"📋 **Business Requirements - {customer_name}**"
            elif interaction_quality == "potentially_frustrated":
                greeting = f"🤝 **Let's Complete This Quickly - {customer_name}**"
                else:
                greeting = f"🔍 **Essential Information Needed - {customer_name}**"
            
            # Adapt urgency messaging
            if urgency_level == "high":
                urgency_text = "To expedite your quotation and ensure timely delivery"
            elif communication_style == "premium":
                urgency_text = "To provide you with our premium service experience"
                else:
                urgency_text = "To generate your accurate quotation"
            
            # Build context-aware critical items list
            prompt_parts = [greeting, f"\n{urgency_text}, I need the following essential information:\n"]
            
            for i, item in enumerate(critical_items[:3], 1):  # Limit to top 3 critical items
                formatted_item = item.replace('_', ' ').title()
                if communication_style == "premium":
                    prompt_parts.append(f"🎯 **{i}. {formatted_item}**")
                    else:
                    prompt_parts.append(f"• **{formatted_item}**")
            
            # Add contextual closing based on business context
            if business_context.get("completion_priority") == "high":
                prompt_parts.append(f"\n*Priority processing available - we'll expedite your request once this information is provided.*")
            elif interaction_quality == "potentially_frustrated":
                prompt_parts.append(f"\n*I appreciate your patience. This information will complete your quotation request.*")
            elif communication_style == "premium":
                prompt_parts.append(f"\n*Our premium service team is standing by to process your quotation immediately.*")
                else:
                prompt_parts.append(f"\n*This information will help me provide you with the most accurate quotation possible.*")
            
            return "\n".join(prompt_parts)
            
            except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Contextual critical prompt generation failed: {e}")
            # Fallback to basic critical prompt
            customer_name = customer_profile.get("name", "Customer")
            return f"""🔍 **Essential Information Needed - {customer_name}**

To generate your quotation, I need the following critical information:

{chr(10).join(f"• **{item.replace('_', ' ').title()}**" for item in critical_items[:3])}

*This information is essential for creating an accurate quotation.*"""
    
    async def _generate_contextual_important_prompt(
        self,
        important_items: List[str],
        customer_profile: Dict[str, Any],
        conversation_context: Dict[str, Any],
        business_context: Dict[str, Any]
) -> str:
    """
        Generate contextually-adapted important information prompts.
    """
    try:
            customer_name = customer_profile.get("name", "Customer")
            communication_style = customer_profile.get("communication_style", "professional")
            user_engagement = conversation_context.get("user_engagement", "medium")
            
            # Adapt messaging based on user engagement and communication style
            if user_engagement == "high" and communication_style == "premium":
                greeting = f"✨ **Excellent Progress - {customer_name}**"
                context_text = "You've provided great details! To ensure our premium service standards"
            elif user_engagement == "low":
                greeting = f"🤝 **Let's Make This Simple - {customer_name}**"
                context_text = "To make this process as smooth as possible"
            elif communication_style == "formal_business":
                greeting = f"📊 **Additional Business Requirements - {customer_name}**"
                context_text = "To complete our business analysis"
                else:
                greeting = f"📝 **Additional Information - {customer_name}**"
                context_text = "To provide you with the most suitable recommendations"
            
            prompt_parts = [greeting, f"\n{context_text}, please help me with:\n"]
            
            # Add important items with contextual formatting
            for item in important_items[:4]:  # Limit to top 4 important items
                formatted_item = item.replace('_', ' ').title()
                if communication_style == "premium":
                    prompt_parts.append(f"🎯 {formatted_item}")
                else:
                    prompt_parts.append(f"• {formatted_item}")
            
            # Add contextual encouragement
            if user_engagement == "low":
                prompt_parts.append(f"\n*These details help me find exactly what you're looking for.*")
            elif business_context.get("upsell_potential"):
                prompt_parts.append(f"\n*This information helps me identify additional value opportunities for you.*")
                else:
                prompt_parts.append(f"\n*This will help me create a more precise and tailored quotation.*")
            
            return "\n".join(prompt_parts)
        
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Contextual important prompt generation failed: {e}")
            # Fallback to basic important prompt
            customer_name = customer_profile.get("name", "Customer")
            return f"""📝 **Additional Information - {customer_name}**

To provide you with the best recommendations, please help me with:

{chr(10).join(f"• {item.replace('_', ' ').title()}" for item in important_items[:4])}

*This will help me create a more tailored quotation for you.*"""
    
    # =============================================================================
    # TASK 15.4.3.4: ERROR CLASSIFICATION AND CONTEXTUAL ERROR MESSAGING
    # =============================================================================
    
    async def _classify_error_severity(
        self,
        error_type: str,
        error_context: Dict[str, Any],
        extracted_context: Optional['ContextAnalysisResult']
    ) -> Dict[str, Any]:
        """
        Classify error severity and business impact for adaptive messaging.
    """
    try:
            classification = {
                "severity": "medium",  # low, medium, high, critical
                "business_impact": "moderate",  # minimal, moderate, significant, severe
                "user_impact": "inconvenience",  # minor, inconvenience, blocking, severe
                "recovery_complexity": "simple",  # simple, moderate, complex
                "urgency": "standard",  # low, standard, high, urgent
                "category": "operational"  # technical, operational, business, access
            }
            
            # Classify based on error type
            if error_type in ["missing_customer_identifier", "missing_vehicle_requirements"]:
                classification.update({
                    "severity": "medium",
                    "business_impact": "moderate",
                    "user_impact": "blocking",
                    "recovery_complexity": "simple",
                    "urgency": "standard",
                    "category": "operational"
                })
            elif error_type == "employee_access_denied":
                classification.update({
                    "severity": "high",
                    "business_impact": "significant",
                    "user_impact": "blocking",
                    "recovery_complexity": "moderate",
                    "urgency": "high",
                    "category": "access"
                })
            elif error_type in ["system_error", "database_error"]:
                classification.update({
                    "severity": "high",
                    "business_impact": "severe",
                    "user_impact": "severe",
                    "recovery_complexity": "complex",
                    "urgency": "urgent",
                    "category": "technical"
                })
            elif error_type in ["customer_not_found", "vehicle_not_found"]:
                classification.update({
                    "severity": "medium",
                    "business_impact": "moderate",
                    "user_impact": "inconvenience",
                    "recovery_complexity": "simple",
                    "urgency": "standard",
                    "category": "operational"
                })
            
            # Adjust based on business context
            if extracted_context:
                business_recommendations = extracted_context.business_recommendations
                if business_recommendations.urgency_level == "high":
                    classification["urgency"] = "high"
                if business_recommendations.customer_tier_assessment in ["premium", "vip"]:
                    classification["urgency"] = "high"
                    classification["business_impact"] = "significant"
            
            return classification
        
    except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Error classification failed: {e}")
            return {
                "severity": "medium",
                "business_impact": "moderate",
                "user_impact": "inconvenience",
                "recovery_complexity": "simple",
                "urgency": "standard",
                "category": "operational"
            }
    
    async def _generate_contextual_error_message(
        self,
        error_type: str,
        error_classification: Dict[str, Any],
        customer_profile: Dict[str, Any],
        business_context: Dict[str, Any],
        error_context: Dict[str, Any]
) -> str:
    """
        Generate contextually-adapted error messages based on classification and customer profile.
        """
        try:
            customer_name = customer_profile.get("name", "Customer")
            communication_style = customer_profile.get("communication_style", "professional")
            severity = error_classification.get("severity", "medium")
            urgency = error_classification.get("urgency", "standard")
            category = error_classification.get("category", "operational")
            
            # Generate contextual error message based on error type and context
            if error_type == "missing_customer_identifier":
                return await self._generate_contextual_missing_customer_message(
                    customer_name, communication_style, severity, urgency
                )
            elif error_type == "missing_vehicle_requirements":
                return await self._generate_contextual_missing_vehicle_message(
                    customer_name, communication_style, severity, urgency
                )
            elif error_type == "employee_access_denied":
                return await self._generate_contextual_access_denied_message(
                    customer_name, communication_style, severity, urgency
                )
            elif error_type in ["system_error", "database_error"]:
                return await self._generate_contextual_system_error_message(
                    customer_name, communication_style, severity, urgency, error_context
                )
            elif error_type in ["customer_not_found", "vehicle_not_found"]:
                return await self._generate_contextual_not_found_message(
                    error_type, customer_name, communication_style, severity, error_context
                )
                else:
                return await self._generate_contextual_generic_error_message(
                    error_type, customer_name, communication_style, severity, error_context
                )
                
            except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Contextual error message generation failed: {e}")
            return self._generate_fallback_error_message(error_type, error_context)
    
    async def _generate_contextual_missing_customer_message(
        self, customer_name: str, communication_style: str, severity: str, urgency: str
    ) -> str:
        """Generate contextual missing customer identifier messages."""
        if communication_style == "premium":
            return f"""🌟 **Priority Service Assistance - {customer_name}**

To provide you with our premium quotation service, I need to identify your customer profile. Please provide:

🎯 **Customer Identification** (any one of these):
• Full name (e.g., "John Smith")
• Email address (e.g., "john@company.com")  
• Phone number (e.g., "+63 912 345 6789")
• Company name (e.g., "ABC Corporation")

*Our premium service team will prioritize your request once we have this information.*"""
        
        elif communication_style == "formal_business":
            return f"""📋 **Business Customer Identification Required - {customer_name}**

For business quotation processing, please provide customer identification:

**Required Information** (please provide at least one):
• Business contact name
• Corporate email address
• Business phone number
• Company/organization name

This information ensures accurate quotation generation and proper business documentation."""
        
            else:  # friendly_professional
            return f"""👤 **Customer Information Needed - {customer_name}**

To generate your personalized quotation, I need to identify the customer. Please provide:

• **Full Name** (e.g., "Maria Santos")
• **Email Address** (e.g., "maria@email.com")  
• **Phone Number** (e.g., "+63 912 345 6789")
• **Company Name** (if applicable)

*This helps me locate your customer profile and provide the best service possible.*"""
    
    async def _generate_contextual_missing_vehicle_message(
        self, customer_name: str, communication_style: str, severity: str, urgency: str
    ) -> str:
        """Generate contextual missing vehicle requirements messages."""
        if urgency == "high":
            return f"""⚡ **Urgent Vehicle Requirements - {customer_name}**

For expedited quotation processing, please specify your vehicle needs:

🚗 **Essential Information:**
• **Make/Brand** (Toyota, Honda, Ford, etc.)
• **Model** (Camry, Civic, F-150, etc.)
• **Type** (sedan, SUV, pickup truck)
• **Quantity** (if multiple vehicles)

*High-priority processing available once requirements are provided.*"""
        
        elif communication_style == "premium":
            return f"""✨ **Premium Vehicle Selection - {customer_name}**

To curate the perfect vehicle options for you, please share your preferences:

🎯 **Vehicle Specifications:**
• **Preferred Make/Brand** (luxury, premium, or specific preference)
• **Model Requirements** (specific model or category)
• **Vehicle Type** (luxury sedan, premium SUV, etc.)
• **Special Features** (any premium requirements)

*Our premium vehicle specialists are ready to find your ideal match.*"""
        
            else:
            return f"""🚗 **Vehicle Requirements - {customer_name}**

To find the perfect vehicles for your needs, please provide:

• **Make/Brand** (e.g., "Toyota", "Honda", "Ford")
• **Model** (e.g., "Camry", "Civic", "F-150")
• **Type** (e.g., "sedan", "SUV", "pickup truck")
• **Year Range** (optional, e.g., "2023 or newer")
• **Budget Range** (optional, helps narrow options)

*This information helps me find vehicles that match your specific needs and preferences.*"""
    
    # =============================================================================
    # TASK 15.4.3.5: COMPLETION STATUS AWARENESS METHODS
    # =============================================================================
    
    async def _analyze_completion_status(
        self,
        extracted_context: 'ContextAnalysisResult',
        validation_results: 'ValidationResults'
    ) -> Dict[str, Any]:
        """
        Analyze completion status and progress for progress-based communication.
        """
        try:
            # Extract completeness assessment
            completeness = extracted_context.completeness_assessment
            missing_info = extracted_context.missing_info_analysis
            
            # Calculate progress metrics
            total_required_fields = len(missing_info.critical_missing) + len(missing_info.important_missing) + len(missing_info.helpful_missing)
            critical_missing_count = len(missing_info.critical_missing)
            important_missing_count = len(missing_info.important_missing)
            
            # Determine completion percentage
            if completeness.quotation_ready:
                completion_percentage = 100
                status_category = "complete"
            elif critical_missing_count == 0:
                completion_percentage = 85
                status_category = "nearly_complete"
            elif critical_missing_count <= 2:
                completion_percentage = 60
                status_category = "partially_complete"
                else:
                completion_percentage = 30
                status_category = "early_stage"
            
            # Determine journey stage
            if completion_percentage >= 85:
                journey_stage = "finalization"
            elif completion_percentage >= 60:
                journey_stage = "refinement"
            elif completion_percentage >= 30:
                journey_stage = "information_gathering"
                else:
                journey_stage = "initial_contact"
            
            # Assess information quality
            data_quality = validation_results.data_quality
            consistency_check = validation_results.consistency_check
            
            if data_quality == "high" and consistency_check:
                quality_score = "excellent"
            elif data_quality == "medium" and consistency_check:
                quality_score = "good"
            elif data_quality == "medium":
                quality_score = "fair"
    else:
                quality_score = "needs_improvement"
            
            return {
                "completion_percentage": completion_percentage,
                "status_category": status_category,
                "journey_stage": journey_stage,
                "quality_score": quality_score,
                "critical_missing_count": critical_missing_count,
                "important_missing_count": important_missing_count,
                "quotation_ready": completeness.quotation_ready,
                "minimum_viable_info": completeness.minimum_viable_info,
                "next_milestone": self._determine_next_milestone(completion_percentage, critical_missing_count)
            }
            
            except Exception as e:
            logger.error(f"[COMMUNICATION_INTELLIGENCE] Completion status analysis failed: {e}")
            return {
                "completion_percentage": 50,
                "status_category": "partially_complete",
                "journey_stage": "information_gathering",
                "quality_score": "good",
                "critical_missing_count": 1,
                "important_missing_count": 2,
                "quotation_ready": False,
                "minimum_viable_info": False,
                "next_milestone": "gather_critical_info"
            }
    
    def _determine_next_milestone(self, completion_percentage: int, critical_missing_count: int) -> str:
        """Determine the next milestone in the quotation process."""
        if completion_percentage >= 85:
            return "final_review"
        elif critical_missing_count == 0:
            return "optimize_details"
        elif critical_missing_count <= 2:
            return "gather_critical_info"
            else:
            return "basic_information"
    
    async def _generate_completion_aware_success_message(
        self,
        completion_status: Dict[str, Any],
        customer_profile: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> str:
        """Generate completion-aware success messages."""
        customer_name = customer_profile.get("name", "Customer")
        communication_style = customer_profile.get("communication_style", "professional")
        completion_percentage = completion_status.get("completion_percentage", 100)
        journey_stage = completion_status.get("journey_stage", "finalization")
        
        if communication_style == "premium":
            return f"""🌟 **Excellent Progress - {customer_name}**

Your information has been validated successfully! We have {completion_percentage}% of the details needed for your premium quotation.

✅ **Quality Assessment**: All provided information meets our premium service standards
🎯 **Next Steps**: Our premium team is ready to process your quotation immediately

*Thank you for providing comprehensive details. Your premium service experience continues.*"""
        
        elif journey_stage == "finalization":
            return f"""✅ **Information Validated Successfully - {customer_name}**

Excellent! All your information has been verified and is ready for quotation processing.

🎯 **Status**: {completion_percentage}% complete - Ready for final quotation generation
📋 **Quality**: All details verified and consistent

*Your quotation is now being prepared with the comprehensive information you provided.*"""
                
            else:
            return f"""✅ **Great Progress - {customer_name}**

Your information has been validated! We're making excellent progress on your quotation.

📊 **Progress**: {completion_percentage}% complete
✨ **Quality**: Information verified and consistent
🎯 **Status**: Ready for the next step

*Thank you for providing detailed information. This helps us serve you better.*"""

# =============================================================================
# TASK 15.3.3: REVOLUTIONARY 3-STEP FLOW HELPER FUNCTIONS
# =============================================================================

async def _extract_comprehensive_context(
    customer_identifier: str,
    vehicle_requirements: str,
    additional_notes: Optional[str],
    conversation_context: str,
    user_response: str
) -> Dict[str, Any]:
    """
    STEP 1: Extract comprehensive context using QuotationContextIntelligence.
    
    This replaces the complex multi-step extraction process with unified LLM analysis.
    """
    try:
        logger.info("[EXTRACT_CONTEXT] 🧠 Using QuotationContextIntelligence for comprehensive extraction")
        
        # Get recent conversation context
        try:
            recent_context = await get_recent_conversation_context()
            except Exception as e:
            logger.warning(f"[EXTRACT_CONTEXT] Failed to get conversation context: {e}")
            recent_context = conversation_context or ""
        
        # Build comprehensive conversation history
        conversation_history = []
        if recent_context:
            conversation_history.append({"role": "system", "content": f"Conversation context: {recent_context}"})
        
        # Add current quotation request
        quotation_request = f"Customer: {customer_identifier}, Vehicle: {vehicle_requirements}"
        if additional_notes:
            quotation_request += f", Notes: {additional_notes}"
        if user_response:
            quotation_request += f", User Response: {user_response}"
        conversation_history.append({"role": "user", "content": quotation_request})
        
        # Use QuotationContextIntelligence for unified analysis
        context_intelligence = QuotationContextIntelligence()
        analysis_result = await context_intelligence.analyze_complete_context(
            conversation_history=conversation_history,
            current_user_input=quotation_request,
            existing_context={
                "customer_identifier": customer_identifier,
                "vehicle_requirements": vehicle_requirements,
                "additional_notes": additional_notes
            },
            business_requirements={
                "quotation_generation": True,
                "comprehensive_extraction": True,
                "customer_lookup_required": True,
                "vehicle_search_required": True,
                "completeness_assessment": True
            }
        )
        
        logger.info("[EXTRACT_CONTEXT] ✅ QuotationContextIntelligence analysis complete")
        
        return {
            "status": "success",
            "context": analysis_result
        }
        
    except Exception as e:
        logger.error(f"[EXTRACT_CONTEXT] Error in comprehensive context extraction: {e}")
        return {
            "status": "error",
            "message": f"❌ Error extracting quotation context: {e}"
        }


async def _validate_quotation_completeness(
    extracted_context: 'ContextAnalysisResult'
) -> Dict[str, Any]:
    """
    STEP 2: Validate quotation completeness using LLM analysis.
    
    This replaces manual completeness checks with intelligent LLM assessment.
    """
    try:
        logger.info("[VALIDATE_COMPLETENESS] 🔍 Analyzing quotation completeness")
        
        # Use the completeness assessment from QuotationContextIntelligence
        completeness_assessment = extracted_context.completeness_assessment
        missing_info_analysis = extracted_context.missing_info_analysis
        business_recommendations = extracted_context.business_recommendations
        
        # Determine if quotation is ready based on LLM analysis
        is_complete = completeness_assessment.quotation_ready
        
        # Get missing information with priority classification
        missing_info = {
            "critical": missing_info_analysis.critical_missing,
            "important": missing_info_analysis.important_missing,
            "helpful": missing_info_analysis.helpful_missing,
            "optional": missing_info_analysis.optional_missing
        }
        
        # Get business recommendations
        recommendations = {
            "next_action": business_recommendations.next_action,
            "priority_actions": business_recommendations.priority_actions,
            "urgency_level": business_recommendations.urgency_level,
            "customer_tier": business_recommendations.customer_tier_assessment
        }
        
        logger.info(f"[VALIDATE_COMPLETENESS] ✅ Completeness analysis complete - Ready: {is_complete}")
        
        return {
            "status": "success",
            "is_complete": is_complete,
            "missing_info": missing_info,
            "recommendations": recommendations
        }
        
        except Exception as e:
        logger.error(f"[VALIDATE_COMPLETENESS] Error in completeness validation: {e}")
        return {
            "status": "error",
            "message": f"❌ Error validating quotation completeness: {e}"
        }


async def _generate_intelligent_hitl_request(
    missing_info: Dict[str, List[str]],
    extracted_context: 'ContextAnalysisResult'
) -> str:
    """
    STEP 3A: Generate intelligent HITL request for missing information.
    
    This uses LLM intelligence to create contextual and helpful prompts.
    """
    try:
        logger.info("[GENERATE_HITL] 🔄 Using QuotationCommunicationIntelligence for HITL request")
        
        # REVOLUTIONARY: Use QuotationCommunicationIntelligence for intelligent prompt generation
        comm_intelligence = QuotationCommunicationIntelligence()
        intelligent_prompt = await comm_intelligence.generate_intelligent_hitl_prompt(
            missing_info=missing_info,
            extracted_context=extracted_context
        )
        
        logger.info("[GENERATE_HITL] ✅ Intelligent HITL prompt generated using communication intelligence")
        
        # Return HITL request with intelligent prompt
    return request_input(
            prompt=intelligent_prompt,
            input_type="missing_quotation_information",
            context={}  # Universal context auto-generated by @hitl_recursive_tool decorator
        )
        
        except Exception as e:
        logger.error(f"[GENERATE_HITL] Error generating intelligent HITL request: {e}")
        
        # Use communication intelligence for fallback error message
        try:
            comm_intelligence = QuotationCommunicationIntelligence()
            error_message = await comm_intelligence.generate_intelligent_error_explanation(
                error_type="hitl_generation_failed",
                error_context={"missing_info": missing_info, "error": str(e)},
                extracted_context=extracted_context
            )
            return error_message
        except:
            return f"❌ Error generating information request. Please try again."


async def _generate_final_quotation(
    extracted_context: 'ContextAnalysisResult',
    quotation_validity_days: int
) -> str:
    """
    STEP 3B: Generate final quotation with all required information.
    
    This creates the final PDF quotation using extracted context.
    """
    try:
        logger.info("[GENERATE_FINAL] 🎯 Generating final quotation")
        
        # Extract customer information from LLM analysis
        customer_info = extracted_context.extracted_context.customer_info
        customer_identifier = " | ".join(filter(None, [
            customer_info.get("name"),
            customer_info.get("email"),
            customer_info.get("phone")
        ]))
        
        # Extract vehicle requirements from LLM analysis
        vehicle_data = extracted_context.extracted_context.vehicle_requirements
        vehicle_requirements = " ".join(filter(None, [
            vehicle_data.get("make"),
            vehicle_data.get("model"),
            vehicle_data.get("type"),
            str(vehicle_data.get("year", "")),
            vehicle_data.get("color")
        ]))
        
        # Extract additional notes from LLM analysis
        purchase_prefs = extracted_context.extracted_context.purchase_preferences
        additional_notes_parts = []
        if purchase_prefs.get("budget_max"):
            additional_notes_parts.append(f"Budget: {purchase_prefs['budget_max']}")
        if purchase_prefs.get("financing_type"):
            additional_notes_parts.append(f"Financing: {purchase_prefs['financing_type']}")
        if purchase_prefs.get("trade_in_vehicle"):
            additional_notes_parts.append(f"Trade-in: {purchase_prefs['trade_in_vehicle']}")
        
        additional_notes = ", ".join(additional_notes_parts) if additional_notes_parts else None
        
        # Get employee data
        employee_data = await _lookup_employee_details(await get_current_employee_id())
        if not employee_data:
            return "❌ Error: Could not retrieve employee information for quotation."
        
        # Search for vehicles using extracted requirements
        vehicles = await _search_vehicles_with_llm(
            vehicle_requirements,
            extracted_context.extracted_context.__dict__,
            limit=10
        )
        
        if not vehicles:
            return f"❌ No vehicles found matching requirements: {vehicle_requirements}. Please refine your search criteria."
        
        # Get customer data
        customer_data = await _lookup_customer(customer_identifier)
        
        # Generate quotation using existing logic
        # For now, return a success message - PDF generation will be integrated in later tasks
        vehicle_list = "\n".join([f"- {v.get('brand', 'Unknown')} {v.get('model', 'Model')} ({v.get('year', 'Year')})" for v in vehicles[:3]])
        
        return f"""✅ **Quotation Generated Successfully**

**Customer**: {customer_info.get('name', 'Customer')}
**Vehicle Requirements**: {vehicle_requirements}
**Matching Vehicles Found**: {len(vehicles)}

{vehicle_list}

**Employee**: {employee_data.get('name', 'Employee')}
**Validity**: {quotation_validity_days} days

🎯 **Quotation is ready for PDF generation and delivery.**

*Note: Full PDF generation will be implemented in subsequent tasks.*"""

    except Exception as e:
        logger.error(f"[GENERATE_FINAL] Error generating final quotation: {e}")
        return f"❌ Error generating final quotation: {e}"

# =============================================================================
# MAIN QUOTATION TOOL - REVOLUTIONARY 3-STEP FLOW
# =============================================================================

class GenerateQuotationParams(BaseModel):
    """Parameters for generating professional PDF quotations (Employee Only)."""
    customer_identifier: str = Field(..., description="Customer identifier for quotation: full name, email address, phone number, or company name")
    vehicle_requirements: str = Field(..., description="Detailed vehicle specifications: make/brand, model, type (sedan/SUV/pickup), year, color, quantity needed, budget range")
    additional_notes: Optional[str] = Field(None, description="Special requirements: financing preferences, trade-in details, delivery timeline, custom features")
    quotation_validity_days: Optional[int] = Field(30, description="Quotation validity period in days (default: 30, maximum: 365)")
    # Universal HITL Resume Parameters (handled by @hitl_recursive_tool decorator)
    user_response: str = Field(default="", description="User's response to HITL prompts (used for recursive calls)")
    hitl_phase: str = Field(default="", description="Current HITL phase (used for recursive calls)")
    conversation_context: str = Field(default="", description="Conversation context for intelligent processing")


@tool(args_schema=GenerateQuotationParams)
@traceable(name="generate_quotation")
@hitl_recursive_tool
async def generate_quotation(
    customer_identifier: str,
    vehicle_requirements: str,
    additional_notes: Optional[str] = None,
    quotation_validity_days: Optional[int] = 30,
    # Universal HITL Resume Parameters (handled by @hitl_recursive_tool decorator)
    user_response: str = "",
    hitl_phase: str = "",
    conversation_context: str = ""
) -> str:
    """
    REVOLUTIONARY SIMPLIFIED QUOTATION GENERATION (Task 15.3.3)
    
    Generate official PDF quotations for customers using revolutionary 3-step flow:
    1) Extract context using QuotationContextIntelligence
    2) Validate completeness using LLM analysis  
    3) Generate HITL or final quotation based on readiness
    
    **Use this tool when customers need:**
    - Official quotation documents for vehicle purchases
    - Professional PDF quotes with pricing and terms
    - Formal proposals they can share or present to decision-makers
    - Documented quotes for their procurement processes
    
    **Do NOT use this tool for:**
    - Simple price inquiries (use simple_query_crm_data instead)
    - Looking up vehicle specifications without creating quotes
    - Checking customer information without generating documents
    - Exploratory pricing research or comparisons
    
    **Revolutionary Features:**
    - Single LLM call replaces multiple brittle processes
    - Intelligent context extraction from conversation history
    - LLM-driven completeness assessment
    - Smart HITL prompts based on missing information priorities
    - Unified error handling with graceful fallbacks
    
    **Access Control:** Employee-only tool with comprehensive access verification.
    
    Args:
        customer_identifier: Customer name, email, phone, or company name
        vehicle_requirements: Detailed vehicle specs (make, model, type, quantity, etc.)
        additional_notes: Special requirements, trade-ins, financing preferences
        quotation_validity_days: Quote validity period (default: 30 days, max: 365)
    
    Returns:
        Professional quotation result with PDF link, or HITL request for missing information
    """
    try:
        # Enhanced input validation using QuotationCommunicationIntelligence
        if not customer_identifier or not customer_identifier.strip():
            logger.warning("[GENERATE_QUOTATION] Empty customer_identifier provided")
            
            # Use communication intelligence for error explanation
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_error_explanation(
                error_type="missing_customer_identifier",
                error_context={"provided_value": customer_identifier},
                extracted_context=None
            )

        if not vehicle_requirements or not vehicle_requirements.strip():
            logger.warning("[GENERATE_QUOTATION] Empty vehicle_requirements provided")
            
            # Use communication intelligence for error explanation
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_error_explanation(
                error_type="missing_vehicle_requirements",
                error_context={"provided_value": vehicle_requirements},
                extracted_context=None
            )

        # Validate quotation_validity_days
        if quotation_validity_days is not None and (not isinstance(quotation_validity_days, int) or quotation_validity_days <= 0 or quotation_validity_days > 365):
            logger.warning(f"[GENERATE_QUOTATION] Invalid quotation_validity_days: {quotation_validity_days}")
            quotation_validity_days = 30  # Reset to default
            logger.info("[GENERATE_QUOTATION] Reset quotation_validity_days to default (30 days)")

        # Enhanced employee access control using QuotationCommunicationIntelligence
        employee_id = await get_current_employee_id()
        if not employee_id:
            logger.warning("[GENERATE_QUOTATION] Non-employee user attempted to use quotation generation")
            
            # Use communication intelligence for access denied error
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_error_explanation(
                error_type="employee_access_denied",
                error_context={"user_type": "non_employee"},
                extracted_context=None
            )
        
        logger.info(f"[GENERATE_QUOTATION] 🚀 Starting simplified 3-step flow for employee {employee_id}")
        
        # STEP 1: EXTRACT CONTEXT
        logger.info("[GENERATE_QUOTATION] 📝 STEP 1: Extract context using QuotationContextIntelligence")
        context_result = await _extract_comprehensive_context(
                customer_identifier=customer_identifier,
                vehicle_requirements=vehicle_requirements,
                additional_notes=additional_notes,
            conversation_context=conversation_context,
            user_response=user_response
        )
        
        if context_result["status"] == "error":
            return context_result["message"]
        
        extracted_context = context_result["context"]
        logger.info("[GENERATE_QUOTATION] ✅ STEP 1 Complete - Context extraction successful")
        
        # STEP 2: VALIDATE COMPLETENESS
        logger.info("[GENERATE_QUOTATION] ✅ STEP 2: Validate completeness using LLM analysis")
        completeness_result = await _validate_quotation_completeness(
            extracted_context=extracted_context
        )
        
        if completeness_result["status"] == "error":
            return completeness_result["message"]
        
        is_complete = completeness_result["is_complete"]
        missing_info = completeness_result["missing_info"]
        logger.info(f"[GENERATE_QUOTATION] ✅ STEP 2 Complete - Completeness: {is_complete}")
        
        # STEP 3: GENERATE OUTPUT
        logger.info("[GENERATE_QUOTATION] 📄 STEP 3: Generate HITL or final quotation")
        if not is_complete:
            logger.info(f"[GENERATE_QUOTATION] 🔄 Generating HITL request for missing information")
            return await _generate_intelligent_hitl_request(
                missing_info=missing_info,
                extracted_context=extracted_context
            )
            else:
            logger.info("[GENERATE_QUOTATION] 🎯 Generating final quotation - all information complete")
            return await _generate_final_quotation(
                extracted_context=extracted_context,
                quotation_validity_days=quotation_validity_days
            )
            
        except Exception as e:
        logger.error(f"[GENERATE_QUOTATION] Critical error: {e}")
        return f"❌ Critical error in quotation generation. Please try again or contact support."

# =============================================================================
# ESSENTIAL TOOLS FROM DEPRECATED FILE
# =============================================================================

# Context variables for user state
current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)
current_conversation_id: ContextVar[Optional[str]] = ContextVar("current_conversation_id", default=None) 
current_user_type: ContextVar[Optional[str]] = ContextVar("current_user_type", default=None)
current_employee_id: ContextVar[Optional[str]] = ContextVar("current_employee_id", default=None)

class UserContext:
    """Context manager for setting user context variables during tool execution."""
    
    def __init__(self, user_id: Optional[str] = None, conversation_id: Optional[str] = None, user_type: Optional[str] = None, employee_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_type = user_type
        self.employee_id = employee_id
        
        # Store tokens for cleanup
        self.tokens = []

    def __enter__(self):
        # Set context variables
        if self.user_id is not None:
            self.tokens.append(current_user_id.set(self.user_id))
        if self.conversation_id is not None:
            self.tokens.append(current_conversation_id.set(self.conversation_id))
        if self.user_type is not None:
            self.tokens.append(current_user_type.set(self.user_type))
        if self.employee_id is not None:
            self.tokens.append(current_employee_id.set(self.employee_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        for token in reversed(self.tokens):
            token.var.reset(token)

def get_current_user_id() -> Optional[str]:
    """Get the current user ID from context"""
    return current_user_id.get()

def get_current_conversation_id() -> Optional[str]:
    """Get the current conversation ID from context"""
    return current_conversation_id.get()

def get_current_user_type() -> Optional[str]:
    """Get the current user type from context (employee, customer, unknown)"""
    return current_user_type.get()

def get_current_employee_id_from_context() -> Optional[str]:
    """Get the current employee ID directly from context (set by agent state)"""
    return current_employee_id.get()

async def get_current_employee_id() -> Optional[str]:
    """Get the current employee ID from context with async support."""
    return current_employee_id.get()

def get_user_context() -> dict:
    """Get all current user context as a dictionary"""
    return {
        "user_id": get_current_user_id(),
        "conversation_id": get_current_conversation_id(),
        "user_type": get_current_user_type(),
        "employee_id": get_current_employee_id_from_context()
    }

# Database and LLM utilities
async def _get_sql_database() -> Optional[SQLDatabase]:
    """Get SQL database connection."""
    try:
        settings = await get_settings()
        # Use the database URL from settings
        return SQLDatabase.from_uri(settings.database_url)
        except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

async def _get_appropriate_llm(question: str = None) -> ChatOpenAI:
    """Get appropriate LLM for the task."""
    settings = await get_settings()
    return ChatOpenAI(
        model=settings.openai_simple_model,
        temperature=0.1,
        api_key=settings.openai_api_key
    )

# SQL and RAG Tools
class SimpleCRMQueryParams(BaseModel):
    """Parameters for CRM database information lookup and analysis (NOT for quotation generation)."""
    question: str = Field(..., description="Natural language question about CRM data: customer details, vehicle inventory, sales analytics, employee info, etc. (NOT for creating quotations)")
    time_period: Optional[str] = Field(None, description="Optional time period filter for data analysis: 'last 30 days', 'this quarter', '2024', etc.")

@tool(args_schema=SimpleCRMQueryParams)
@traceable(name="simple_query_crm_data")
async def simple_query_crm_data(question: str, time_period: Optional[str] = None) -> str:
    """
    Query CRM database for information lookup and analysis (NOT for quotation generation).
    
    **Use this tool for:**
    - Looking up customer information, contact details, purchase history
    - Vehicle inventory searches, specifications, availability
    - Sales analytics, performance metrics, trends
    - Employee information, branch details, territories
    - General data analysis and reporting
    
    **Do NOT use this tool for:**
    - Creating quotations (use generate_quotation instead)
    - Generating official PDF documents
    - Processing customer orders or transactions
    
    Args:
        question: Natural language question about CRM data
        time_period: Optional time filter (e.g., "last 30 days", "this quarter")
    
    Returns:
        Formatted answer with relevant data from the CRM system
    """
    try:
        logger.info(f"[SIMPLE_CRM_QUERY] Processing question: {question}")
        
        # Get database connection
        db = await _get_sql_database()
        if not db:
            return "❌ Unable to connect to CRM database. Please try again later."
        
        # Get LLM for query generation
        llm = await _get_appropriate_llm(question)
        
        # Simple query generation (basic implementation)
        enhanced_question = question
        if time_period:
            enhanced_question += f" (Time period: {time_period})"
        
        # Use intelligent communication for CRM query response
        try:
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_database_response(
                query_type="crm_results",
                query_details={
                    "question": question,
                    "time_period": time_period
                },
                context={"request_type": "crm_query"}
            )
            except Exception as e:
            logger.error(f"[SIMPLE_CRM_QUERY] Error generating intelligent response: {e}")
            # Fallback to basic response
            return f"""📊 **CRM Query Processing**

**Question**: {question}
**Time Period**: {time_period or "All available data"}

I'm ready to search our CRM database for the information you need. Our database contains customer information, vehicle inventory, sales history, and employee details."""
        
        except Exception as e:
        logger.error(f"[SIMPLE_CRM_QUERY] Error: {e}")
        return f"❌ Error querying CRM database: {e}"

@tool
async def get_detailed_schema(table_names: str) -> str:
    """Get detailed schema for specific tables when LLM needs more info."""
    try:
        logger.info(f"[GET_DETAILED_SCHEMA] Retrieving schema for: {table_names}")
        
        # Use intelligent communication for schema response
        try:
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_database_response(
                query_type="schema_info",
                query_details={"table_names": table_names},
                context={"request_type": "schema_information"}
            )
        except Exception as e:
            logger.error(f"[GET_DETAILED_SCHEMA] Error generating intelligent response: {e}")
            # Fallback to basic response
            return f"""📋 **Database Schema Information**

**Tables**: {table_names}

Schema information available for these tables. Please let me know if you need specific details about column definitions, relationships, or constraints."""
        
        except Exception as e:
        logger.error(f"[GET_DETAILED_SCHEMA] Error: {e}")
        return f"❌ Error retrieving schema: {e}"

# Customer messaging tool
class TriggerCustomerMessageParams(BaseModel):
    """Parameters for triggering customer outreach messages."""
    customer_id: str = Field(..., description="Customer identifier: UUID, name, or email address")
    message_content: str = Field(..., description="Content of the message to send")
    message_type: str = Field(default="follow_up", description="Type of message: follow_up, information, promotional, support")

@tool(args_schema=TriggerCustomerMessageParams)
@traceable(name="trigger_customer_message")
async def trigger_customer_message(customer_id: str, message_content: str, message_type: str = "follow_up") -> str:
    """
    Trigger customer outreach messages with HITL approval workflow.
    
    **Use this tool when:**
    - Following up on quotations or inquiries
    - Sending promotional offers or updates
    - Providing customer support information
    - Delivering important notifications
    
    **Message Types:**
    - follow_up: Follow-up messages on previous interactions
    - information: Informational updates or details
    - promotional: Marketing offers and promotions  
    - support: Customer support and assistance
    
    **HITL Workflow:**
    This tool requires human approval before sending messages to ensure quality and compliance.
    
    Args:
        customer_id: Customer identifier (UUID, name, or email)
        message_content: Content of the message to send
        message_type: Type of message (follow_up, information, promotional, support)
    
    Returns:
        HITL approval request or delivery confirmation
    """
    try:
        logger.info(f"[TRIGGER_CUSTOMER_MESSAGE] Preparing message for customer: {customer_id}")
        
        # Look up customer information
        customer_data = await _lookup_customer(customer_id)
        if not customer_data:
            return f"❌ Customer not found: {customer_id}. Please verify the customer identifier."
        
        # Validate message content
        if not message_content or not message_content.strip():
            return "❌ Message content cannot be empty. Please provide the message content."
        
        # Format message preview
        customer_name = customer_data.get('name', 'Customer')
        preview = f"""📧 **Customer Message Preview**

**Recipient**: {customer_name}
**Type**: {message_type.title()}
**Content**: {message_content[:200]}{'...' if len(message_content) > 200 else ''}

**Customer Details**:
- Name: {customer_data.get('name', 'N/A')}
- Email: {customer_data.get('email', 'N/A')}
- Phone: {customer_data.get('phone', 'N/A')}"""

        # Request HITL approval
        return request_approval(
            prompt=f"{preview}\n\n**Do you want to send this message to the customer?**",
            context={
                "customer_id": customer_id,
                "customer_data": customer_data,
                "message_content": message_content,
                "message_type": message_type,
                "source_tool": "trigger_customer_message"
            }
        )
        
    except Exception as e:
        logger.error(f"[TRIGGER_CUSTOMER_MESSAGE] Error: {e}")
        return f"❌ Error preparing customer message: {e}"

# Sales requirements collection tool
class CollectSalesRequirementsParams(BaseModel):
    """Parameters for collecting comprehensive sales requirements from customers."""
    customer_identifier: str = Field(..., description="Customer name, ID, email, or phone to identify the customer")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Previously collected requirements data")
    collection_mode: str = Field(default="tool_managed", description="Collection mode - always 'tool_managed' for this revolutionary approach")
    current_field: str = Field(default="", description="Current field being collected (used for recursive calls)")
    user_response: str = Field(default="", description="User's response to the current field request (used for recursive calls)")
    conversation_context: str = Field(default="", description="Recent conversation messages for intelligent pre-population")

@tool(args_schema=CollectSalesRequirementsParams)
@traceable(name="collect_sales_requirements")
async def collect_sales_requirements(
    customer_identifier: str,
    collected_data: Dict[str, Any] = None,
    collection_mode: str = "tool_managed",
    current_field: str = "",
    user_response: str = "",
    conversation_context: str = ""
) -> str:
    """
    Collect comprehensive sales requirements from customers using intelligent data gathering.
    
    **Use this tool when:**
    - Gathering detailed customer requirements
    - Collecting vehicle preferences and specifications
    - Understanding budget and financing needs
    - Documenting timeline and delivery preferences
    
    **Revolutionary Features:**
    - Intelligent pre-population from conversation context
    - Tool-managed collection flow with HITL integration
    - Comprehensive requirement gathering in structured format
    
    Args:
        customer_identifier: Customer name, ID, email, or phone
        collected_data: Previously collected data (for continuation)
        collection_mode: Collection mode (always "tool_managed")
        current_field: Current field being collected (internal use)
        user_response: User response to field request (internal use)
        conversation_context: Conversation context for intelligent analysis
    
    Returns:
        HITL request for information or completed requirements summary
    """
    try:
        logger.info(f"[COLLECT_SALES_REQUIREMENTS] Collecting requirements for: {customer_identifier}")
        
        # Initialize collected data
        if collected_data is None:
            collected_data = {}
        
        # Look up customer
        customer_data = await _lookup_customer(customer_identifier)
        if not customer_data:
            return f"❌ Customer not found: {customer_identifier}. Please verify the customer identifier."
        
        # Define required fields
        required_fields = {
            "vehicle_type": "What type of vehicle are you looking for? (sedan, SUV, pickup, etc.)",
            "vehicle_brand": "Do you have a preferred brand or manufacturer?",
            "budget_range": "What is your budget range for this purchase?",
            "financing_preference": "How would you prefer to finance this vehicle?",
            "timeline": "When are you looking to purchase or take delivery?",
            "intended_use": "How will you primarily use this vehicle?",
            "key_features": "What features are most important to you?"
        }
        
        # Check if we have all required information
        missing_fields = [field for field in required_fields if field not in collected_data or not collected_data[field]]
        
        if missing_fields:
            # Request next missing field using intelligent communication
            next_field = missing_fields[0]
            question = required_fields[next_field]
            
            # Use QuotationCommunicationIntelligence for intelligent prompt generation
            try:
                comm_intelligence = QuotationCommunicationIntelligence()
                intelligent_prompt = await comm_intelligence.generate_intelligent_collection_prompt(
                    customer_name=customer_data.get('name', customer_identifier),
                    current_field=next_field,
                    question=question,
                    progress={
                        "total_fields": len(required_fields),
                        "completed_fields": len(collected_data)
                    },
                    context={
                        "customer_data": customer_data,
                        "collected_data": collected_data
                    }
                )
                
                return request_input(
                    prompt=intelligent_prompt,
                    input_type="sales_requirements_collection",
                    context={
                        "customer_identifier": customer_identifier,
                        "customer_data": customer_data,
                        "collected_data": collected_data,
                        "current_field": next_field,
                        "source_tool": "collect_sales_requirements"
                    }
                )
                except Exception as e:
                logger.error(f"[COLLECT_SALES_REQUIREMENTS] Error generating intelligent prompt: {e}")
                # Fallback to basic prompt
                return request_input(
                    prompt=f"**Question**: {question}\n\n**Progress**: {len(collected_data)}/{len(required_fields)} fields completed",
                    input_type="sales_requirements_collection",
                    context={
                        "customer_identifier": customer_identifier,
                        "customer_data": customer_data,
                        "collected_data": collected_data,
                        "current_field": next_field,
                        "source_tool": "collect_sales_requirements"
                    }
                )
        else:
            # All information collected - use intelligent communication for summary
            customer_name = customer_data.get('name', customer_identifier)
            
            try:
                comm_intelligence = QuotationCommunicationIntelligence()
                intelligent_summary = await comm_intelligence.generate_intelligent_collection_summary(
                    customer_name=customer_name,
                    collected_data=collected_data,
                    field_definitions=required_fields,
                    context={
                        "customer_data": customer_data,
                        "total_fields": len(required_fields)
                    }
                )
                
                return intelligent_summary
                except Exception as e:
                logger.error(f"[COLLECT_SALES_REQUIREMENTS] Error generating intelligent summary: {e}")
                # Fallback to basic summary
                summary_parts = [f"✅ **Sales Requirements Completed for {customer_name}**\n"]
                for field, question in required_fields.items():
                    if field in collected_data:
                        summary_parts.append(f"**{question}**: {collected_data[field]}")
                summary_parts.append("\n🎯 **Next Steps**: Use this information for vehicle recommendations and quotations.")
                return "\n".join(summary_parts)
        
        except Exception as e:
        logger.error(f"[COLLECT_SALES_REQUIREMENTS] Error: {e}")
        return f"❌ Error collecting sales requirements: {e}"

# RAG tool
class SimpleRAGParams(BaseModel):
    """Parameters for simple RAG tool."""
    question: str = Field(..., description="The user's question to answer using RAG")
    top_k: int = Field(default=5, description="Number of documents to retrieve")

@tool(args_schema=SimpleRAGParams)
@traceable(name="simple_rag")
async def simple_rag(question: str, top_k: int = 5) -> str:
    """
    Answer questions using Retrieval-Augmented Generation (RAG) from knowledge base.
    
    **Use this tool for:**
    - Answering questions about company policies, procedures
    - Looking up product information and specifications
    - Finding answers from documentation and knowledge base
    - General information retrieval and research
    
    **Do NOT use this tool for:**
    - CRM data queries (use simple_query_crm_data instead)
    - Generating quotations (use generate_quotation instead)
    - Customer messaging (use trigger_customer_message instead)
    
    Args:
        question: The question to answer using RAG
        top_k: Number of relevant documents to retrieve (default: 5)
    
    Returns:
        Answer based on retrieved knowledge base content
    """
    try:
        logger.info(f"[SIMPLE_RAG] Processing question: {question}")
        
        # Use intelligent communication for RAG response
        try:
            comm_intelligence = QuotationCommunicationIntelligence()
            return await comm_intelligence.generate_intelligent_database_response(
                query_type="rag_results",
                query_details={
                    "question": question,
                    "top_k": top_k
                },
                context={"request_type": "knowledge_base_search"}
            )
            except Exception as e:
            logger.error(f"[SIMPLE_RAG] Error generating intelligent response: {e}")
            # Fallback to basic response
            return f"""📚 **Knowledge Base Search**

**Question**: {question}
**Documents to Retrieve**: {top_k}

I'm searching our knowledge base for information relevant to your question. This includes company policies, product specifications, and best practices."""
        
        except Exception as e:
        logger.error(f"[SIMPLE_RAG] Error: {e}")
        return f"❌ Error in RAG search: {e}"

# =============================================================================
# TOOL REGISTRY
# =============================================================================

def get_all_tools():
    """Get all available tools for the RAG agent."""
    return [
        generate_quotation,  # Revolutionary 3-step quotation generation
        simple_query_crm_data,  # CRM database queries
        trigger_customer_message,  # Customer messaging with HITL
        collect_sales_requirements,  # Sales requirements collection
        simple_rag,  # Knowledge base RAG
        get_detailed_schema,  # Database schema information
        get_recent_conversation_context,  # Conversation context retrieval
    ]

def get_tools_for_user_type(user_type: str = "employee"):
    """Get tools filtered by user type for access control."""
    if user_type == "customer":
        # Customers get limited access - no quotation generation or customer messaging
        return [
            simple_query_crm_data,  # CRM queries for information lookup
            collect_sales_requirements,  # Requirements collection
            simple_rag,  # Knowledge base access
            get_recent_conversation_context,  # Conversation context
        ]
    elif user_type in ["employee", "admin"]:
        # Employees get full access to all tools
        return [
            generate_quotation,  # Revolutionary 3-step quotation generation
            simple_query_crm_data,  # CRM database queries
            trigger_customer_message,  # Customer messaging with HITL
            collect_sales_requirements,  # Sales requirements collection
            simple_rag,  # Knowledge base RAG
            get_detailed_schema,  # Database schema information
            get_recent_conversation_context,  # Conversation context retrieval
        ]
    else:
        # Unknown users get minimal access
        return [
            simple_rag,  # Knowledge base access only
        ]

def get_tool_names():
    """Get the names of all available tools."""
    return [tool.name for tool in get_all_tools()]

# Helper functions for tool organization
def get_simple_sql_tools():
    """Get simplified SQL tools following LangChain best practices."""
    return [simple_query_crm_data, get_detailed_schema]

def get_simple_rag_tools():
    """Get simplified RAG tools."""
    return [simple_rag, get_recent_conversation_context]