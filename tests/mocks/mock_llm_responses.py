"""
Mock LLM Responses for Token-Efficient Testing.

This module provides realistic mock responses for LLM operations
without consuming API tokens, supporting the concurrent testing requirements:

- Mock summary generation responses
- Mock embedding generation  
- Mock conversation analysis
- Deterministic responses for consistent testing
- Configurable response variations for realistic testing

Used by test_concurrent_background_tasks.py for 1000+ task testing.
"""

import hashlib
import random
from typing import List, Dict, Any, Optional
from datetime import datetime


class MockSummaryGenerator:
    """
    Generates realistic conversation summaries without LLM API calls.
    Provides varied, contextual responses for comprehensive testing.
    """
    
    # Template patterns for different conversation types
    SUMMARY_TEMPLATES = {
        "customer_support": [
            "Customer contacted support regarding {topic}. Discussion covered {details} with {sentiment} tone. Resolution: {outcome}.",
            "Support conversation about {topic}. Customer expressed {sentiment} about {details}. Status: {outcome}.",
            "Customer inquiry focused on {topic}. Key points discussed: {details}. Interaction was {sentiment}."
        ],
        "sales": [
            "Sales conversation about {topic}. Customer showed {sentiment} interest in {details}. Next steps: {outcome}.",
            "Product discussion covering {topic}. Customer questions about {details} were {sentiment}. Follow-up: {outcome}.",
            "Sales inquiry regarding {topic}. Customer engagement was {sentiment} with focus on {details}."
        ],
        "technical": [
            "Technical discussion about {topic}. Issues with {details} were addressed with {sentiment} outcome.",
            "Support session for {topic}. Technical details: {details}. Resolution was {sentiment}.",
            "Technical consultation on {topic}. Customer needed help with {details}. Session was {sentiment}."
        ],
        "general": [
            "Conversation covered {topic} with discussion of {details}. Overall tone was {sentiment}.",
            "General inquiry about {topic}. Customer interested in {details}. Interaction: {sentiment}.",
            "Discussion focused on {topic}. Key aspects: {details}. Conversation was {sentiment}."
        ]
    }
    
    # Context-aware vocabulary
    TOPICS = {
        "customer_support": [
            "billing issues", "account problems", "service interruption", "refund request",
            "password reset", "login troubles", "subscription changes", "technical difficulties"
        ],
        "sales": [
            "product pricing", "feature comparison", "upgrade options", "demo request",
            "trial extension", "volume discounts", "implementation timeline", "custom solutions"
        ],
        "technical": [
            "API integration", "configuration setup", "performance optimization", "troubleshooting",
            "system requirements", "data migration", "security settings", "compatibility issues"
        ],
        "general": [
            "general inquiry", "information request", "feedback submission", "feature suggestion",
            "usage questions", "best practices", "documentation", "training resources"
        ]
    }
    
    DETAILS = {
        "customer_support": [
            "payment processing, account verification",
            "service restoration, troubleshooting steps",
            "refund policy, billing cycle",
            "security measures, account recovery"
        ],
        "sales": [
            "pricing tiers, feature benefits",
            "implementation process, timeline",
            "ROI calculations, cost savings",
            "integration capabilities, scalability"
        ],
        "technical": [
            "API endpoints, authentication",
            "database configuration, performance tuning",
            "error handling, logging setup",
            "security protocols, access controls"
        ],
        "general": [
            "product features, usage guidelines",
            "best practices, optimization tips",
            "documentation updates, training materials",
            "community resources, support channels"
        ]
    }
    
    SENTIMENTS = ["positive", "neutral", "constructive", "professional", "collaborative"]
    OUTCOMES = ["resolved", "pending follow-up", "escalated", "scheduled callback", "documentation sent"]
    
    @classmethod
    def generate_summary(cls, message_count: int = 5, conversation_type: str = "general", 
                        user_type: str = "customer") -> str:
        """
        Generate a contextual conversation summary.
        
        Args:
            message_count: Number of messages in conversation
            conversation_type: Type of conversation (customer_support, sales, technical, general)
            user_type: Type of user (customer, employee)
            
        Returns:
            Generated summary string
        """
        # Select appropriate templates and vocabulary
        templates = cls.SUMMARY_TEMPLATES.get(conversation_type, cls.SUMMARY_TEMPLATES["general"])
        topics = cls.TOPICS.get(conversation_type, cls.TOPICS["general"])
        details = cls.DETAILS.get(conversation_type, cls.DETAILS["general"])
        
        # Generate contextual content
        template = random.choice(templates)
        topic = random.choice(topics)
        detail = random.choice(details)
        sentiment = random.choice(cls.SENTIMENTS)
        outcome = random.choice(cls.OUTCOMES)
        
        # Create summary with message count context
        if message_count <= 3:
            summary = template.format(topic=topic, details=detail, sentiment=sentiment, outcome=outcome)
            summary += f" Brief exchange with {message_count} messages."
        elif message_count <= 8:
            summary = template.format(topic=topic, details=detail, sentiment=sentiment, outcome=outcome)
            summary += f" Standard conversation with {message_count} messages exchanged."
        else:
            summary = template.format(topic=topic, details=detail, sentiment=sentiment, outcome=outcome)
            summary += f" Extended discussion with {message_count} messages covering multiple aspects."
        
        return summary
    
    @classmethod
    def generate_summary_with_keywords(cls, messages: List[Dict[str, Any]]) -> str:
        """
        Generate summary based on actual message content (still mock, but more realistic).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated summary with extracted keywords
        """
        if not messages:
            return "Empty conversation with no messages."
        
        # Extract simple keywords from message content
        all_content = " ".join([msg.get("content", "") for msg in messages])
        words = all_content.lower().split()
        
        # Simple keyword extraction (mock approach)
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Determine conversation type based on keywords
        if any(word in ["price", "cost", "buy", "purchase", "sale"] for word in keywords):
            conv_type = "sales"
        elif any(word in ["help", "support", "problem", "issue", "error"] for word in keywords):
            conv_type = "customer_support"
        elif any(word in ["api", "code", "setup", "config", "technical"] for word in keywords):
            conv_type = "technical"
        else:
            conv_type = "general"
        
        # Generate contextual summary
        summary = cls.generate_summary(len(messages), conv_type)
        
        # Add keyword context if available
        if keywords:
            top_keywords = list(set(keywords[:5]))  # Top 5 unique keywords
            summary += f" Key terms: {', '.join(top_keywords)}."
        
        return summary


class MockEmbeddingGenerator:
    """
    Generates consistent mock embedding vectors without API calls.
    Provides deterministic embeddings for testing and caching.
    """
    
    EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-small dimension
    
    @classmethod
    def generate_embedding(cls, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Generate deterministic mock embedding vector.
        
        Args:
            text: Text to generate embedding for
            model: Embedding model name (for compatibility)
            
        Returns:
            List of floats representing the embedding vector
        """
        # Create deterministic hash-based embedding
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Generate embedding values from hash
        embedding = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= cls.EMBEDDING_DIMENSION:
                break
                
            hex_pair = text_hash[i:i+2]
            # Convert hex to float between -1 and 1
            float_val = (int(hex_pair, 16) - 128) / 128.0
            embedding.append(float_val)
        
        # Pad or extend to exact dimension
        while len(embedding) < cls.EMBEDDING_DIMENSION:
            # Repeat pattern to reach target dimension
            remaining = cls.EMBEDDING_DIMENSION - len(embedding)
            to_add = min(remaining, len(embedding))
            embedding.extend(embedding[:to_add])
        
        # Ensure exact dimension
        embedding = embedding[:cls.EMBEDDING_DIMENSION]
        
        # Normalize vector (optional, for realism)
        import math
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    @classmethod
    def generate_batch_embeddings(cls, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Generate multiple embeddings efficiently.
        
        Args:
            texts: List of texts to generate embeddings for
            model: Embedding model name
            
        Returns:
            List of embedding vectors
        """
        return [cls.generate_embedding(text, model) for text in texts]
    
    @classmethod
    def calculate_similarity(cls, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        # Cosine similarity calculation
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class MockConversationAnalyzer:
    """
    Analyzes conversations and extracts insights without LLM calls.
    Provides structured analysis for testing conversation processing.
    """
    
    @classmethod
    def analyze_conversation(cls, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation and extract key metrics.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary with conversation analysis
        """
        if not messages:
            return {
                "message_count": 0,
                "participant_count": 0,
                "conversation_length": "empty",
                "sentiment": "neutral",
                "topics": [],
                "key_phrases": [],
                "resolution_status": "unknown"
            }
        
        # Basic metrics
        message_count = len(messages)
        participants = set(msg.get("role", "unknown") for msg in messages)
        participant_count = len(participants)
        
        # Conversation length classification
        if message_count <= 3:
            length = "brief"
        elif message_count <= 8:
            length = "standard"
        else:
            length = "extended"
        
        # Simple sentiment analysis (mock)
        all_content = " ".join([msg.get("content", "") for msg in messages]).lower()
        positive_words = ["thank", "great", "good", "excellent", "helpful", "solved", "works"]
        negative_words = ["problem", "issue", "error", "broken", "failed", "wrong", "bad"]
        
        positive_count = sum(1 for word in positive_words if word in all_content)
        negative_count = sum(1 for word in negative_words if word in all_content)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Topic extraction (simple keyword-based)
        topics = []
        if "price" in all_content or "cost" in all_content:
            topics.append("pricing")
        if "support" in all_content or "help" in all_content:
            topics.append("support")
        if "technical" in all_content or "api" in all_content:
            topics.append("technical")
        if "account" in all_content or "login" in all_content:
            topics.append("account")
        
        # Key phrases (mock extraction)
        words = all_content.split()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        key_phrases = [word for word in words if len(word) > 4 and word not in common_words][:5]
        
        # Resolution status (mock determination)
        if "resolved" in all_content or "fixed" in all_content or "solved" in all_content:
            resolution = "resolved"
        elif "pending" in all_content or "follow" in all_content:
            resolution = "pending"
        elif "escalate" in all_content or "manager" in all_content:
            resolution = "escalated"
        else:
            resolution = "in_progress"
        
        return {
            "message_count": message_count,
            "participant_count": participant_count,
            "conversation_length": length,
            "sentiment": sentiment,
            "topics": topics,
            "key_phrases": key_phrases,
            "resolution_status": resolution,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    @classmethod
    def extract_action_items(cls, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract action items from conversation (mock implementation).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of action items
        """
        action_items = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            
            # Simple action item detection
            if "will send" in content:
                action_items.append("Send requested information to customer")
            elif "follow up" in content:
                action_items.append("Schedule follow-up contact")
            elif "escalate" in content:
                action_items.append("Escalate issue to appropriate team")
            elif "schedule" in content:
                action_items.append("Schedule appointment or callback")
        
        # Add generic action items if none found
        if not action_items and messages:
            action_items.append("Review conversation and determine next steps")
        
        return action_items[:3]  # Limit to top 3 action items


# Export main classes for easy importing
__all__ = [
    "MockSummaryGenerator",
    "MockEmbeddingGenerator", 
    "MockConversationAnalyzer"
]
