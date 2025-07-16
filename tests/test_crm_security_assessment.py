"""
CRM Security Assessment Test

This test demonstrates the security vulnerabilities in the current CRM system
and provides actionable recommendations for improvement.

Key Findings:
1. No Row Level Security (RLS) on CRM tables
2. All CRM data accessible to all users
3. No user authentication context in queries
4. Potential for data exfiltration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import os
import sys
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.config import get_settings
from backend.database import get_db_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCRMSecurityAssessment:
    """Assessment of CRM security posture."""
    
    async def setup_method(self):
        """Initialize database connection."""
        # Initialize instance variables
        self.db = None
        self.settings = None
        self.findings = []
        self.recommendations = []
        
        # Initialize database connection
        self.settings = await get_settings()
        self.db = await get_db_client().async_client()
        
    async def check_rls_policies(self) -> Dict[str, Any]:
        """Check Row Level Security policies on CRM tables."""
        logger.info("Checking Row Level Security policies...")
        
        crm_tables = [
            'branches', 'employees', 'customers', 'vehicles', 
            'opportunities', 'transactions', 'pricing', 'activities'
        ]
        
        rls_status = {}
        
        async with self.db.connect() as conn:
            for table in crm_tables:
                # Check if RLS is enabled
                result = await conn.execute(f"""
                    SELECT relrowsecurity 
                    FROM pg_class 
                    WHERE relname = '{table}'
                """)
                rls_enabled = result.scalar()
                
                # Check for policies
                policies_result = await conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM pg_policies 
                    WHERE tablename = '{table}'
                """)
                policy_count = policies_result.scalar()
                
                rls_status[table] = {
                    'rls_enabled': rls_enabled,
                    'policy_count': policy_count,
                    'secure': rls_enabled and policy_count > 0
                }
        
        return rls_status
    
    @pytest.mark.asyncio
    async def test_crm_security_assessment(self):
        """Main test method for comprehensive CRM security assessment."""
        await self.setup_method()
        results = await self.run_security_assessment()
        
        # Assess critical security findings
        critical_findings = results.get('critical_findings', [])
        security_score = results.get('security_score', 0)
        
        # Print results
        print("\n" + "="*80)
        print("CRM SECURITY ASSESSMENT RESULTS")
        print("="*80)
        print(f"Security Score: {security_score}/100")
        print(f"Critical Findings: {len(critical_findings)}")
        print("="*80)
        
        # This test documents findings but doesn't fail on security issues
        # to allow for comprehensive reporting
        print_security_report(results)
        
        # Save results
        await save_assessment_report(results)
    
    async def run_security_assessment(self) -> Dict[str, Any]:
        """Run comprehensive security assessment."""
        rls_results = await self.check_rls_policies()
        access_results = await self.test_data_access_patterns()
        user_results = await self.test_user_context()
        attack_results = await self.simulate_attack_scenarios()
        
        return {
            'rls_policies': rls_results,
            'data_access': access_results,
            'user_context': user_results,
            'attack_scenarios': attack_results,
            'security_score': self.calculate_security_score(rls_results, access_results, user_results, attack_results),
            'critical_findings': self.identify_critical_findings(rls_results, access_results, user_results, attack_results),
            'recommendations': self.generate_recommendations(rls_results, access_results, user_results, attack_results)
        }
    
    def calculate_security_score(self, rls_results, access_results, user_results, attack_results) -> int:
        """Calculate overall security score."""
        score = 0
        
        # RLS policies (30 points)
        if rls_results.get('enabled_tables', 0) > 0:
            score += 30
        
        # Data access controls (25 points)
        if access_results.get('unauthorized_access_blocked', 0) > 0:
            score += 25
        
        # User context (20 points)
        if user_results.get('user_context_available', False):
            score += 20
        
        # Attack resistance (25 points)
        successful_attacks = attack_results.get('successful_attacks', 0)
        if successful_attacks == 0:
            score += 25
        
        return score
    
    def identify_critical_findings(self, rls_results, access_results, user_results, attack_results) -> List[Dict[str, Any]]:
        """Identify critical security findings."""
        findings = []
        
        if rls_results.get('enabled_tables', 0) == 0:
            findings.append({
                'severity': 'CRITICAL',
                'title': 'No Row Level Security',
                'description': 'No RLS policies found on any CRM tables'
            })
        
        if not user_results.get('user_context_available', False):
            findings.append({
                'severity': 'CRITICAL',
                'title': 'No User Context',
                'description': 'Unable to identify current user context'
            })
        
        if attack_results.get('successful_attacks', 0) > 0:
            findings.append({
                'severity': 'CRITICAL',
                'title': 'Successful Attack Simulations',
                'description': f"{attack_results.get('successful_attacks', 0)} attack scenarios succeeded"
            })
        
        return findings
    
    def generate_recommendations(self, rls_results, access_results, user_results, attack_results) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if rls_results.get('enabled_tables', 0) == 0:
            recommendations.append("Implement Row Level Security policies on all CRM tables")
        
        if not user_results.get('user_context_available', False):
            recommendations.append("Configure Supabase Auth with proper JWT token handling")
        
        if attack_results.get('successful_attacks', 0) > 0:
            recommendations.append("Implement additional access controls and monitoring")
        
        return recommendations
    
    async def test_data_access_patterns(self) -> Dict[str, Any]:
        """Test different data access patterns to identify vulnerabilities."""
        logger.info("Testing data access patterns...")
        
        test_queries = [
            {
                'name': 'All Customer Data',
                'query': 'SELECT * FROM customers',
                'sensitivity': 'HIGH',
                'should_be_restricted': True
            },
            {
                'name': 'Employee Information',
                'query': 'SELECT name, email, position FROM employees',
                'sensitivity': 'MEDIUM',
                'should_be_restricted': True
            },
            {
                'name': 'Financial Transactions',
                'query': 'SELECT * FROM transactions',
                'sensitivity': 'CRITICAL',
                'should_be_restricted': True
            },
            {
                'name': 'Vehicle Pricing',
                'query': 'SELECT * FROM pricing',
                'sensitivity': 'MEDIUM',
                'should_be_restricted': False
            },
            {
                'name': 'Sales Opportunities',
                'query': 'SELECT * FROM opportunities',
                'sensitivity': 'HIGH',
                'should_be_restricted': True
            }
        ]
        
        access_results = []
        
        async with self.db.connect() as conn:
            for test in test_queries:
                try:
                    result = await conn.execute(test['query'])
                    row_count = len(result.fetchall())
                    
                    access_results.append({
                        'name': test['name'],
                        'query': test['query'],
                        'sensitivity': test['sensitivity'],
                        'accessible': True,
                        'row_count': row_count,
                        'should_be_restricted': test['should_be_restricted'],
                        'security_violation': test['should_be_restricted']
                    })
                    
                except Exception as e:
                    access_results.append({
                        'name': test['name'],
                        'query': test['query'],
                        'sensitivity': test['sensitivity'],
                        'accessible': False,
                        'error': str(e),
                        'should_be_restricted': test['should_be_restricted'],
                        'security_violation': False
                    })
        
        return access_results
    
    async def check_user_context(self) -> Dict[str, Any]:
        """Check if user context is available in database queries."""
        logger.info("Checking user context availability...")
        
        try:
            async with self.db.connect() as conn:
                # Check if there's any user context
                result = await conn.execute("SELECT current_user")
                db_user = result.scalar()
                
                # Check if auth.uid() is available (Supabase RLS function)
                try:
                    result = await conn.execute("SELECT auth.uid()")
                    auth_uid = result.scalar()
                    has_auth_context = auth_uid is not None
                except:
                    has_auth_context = False
                
                # Check if there are any user-related functions
                result = await conn.execute("""
                    SELECT COUNT(*) 
                    FROM pg_proc 
                    WHERE proname LIKE '%user%' OR proname LIKE '%auth%'
                """)
                user_functions = result.scalar()
                
                return {
                    'db_user': db_user,
                    'has_auth_context': has_auth_context,
                    'user_functions_count': user_functions,
                    'context_available': has_auth_context
                }
        except Exception as e:
            return {
                'error': str(e),
                'context_available': False
            }
    
    async def simulate_attack_scenarios(self) -> List[Dict[str, Any]]:
        """Simulate common attack scenarios."""
        logger.info("Simulating attack scenarios...")
        
        scenarios = [
            {
                'name': 'Data Exfiltration - All Customer Data',
                'description': 'Attempt to extract all customer information',
                'query': 'SELECT name, email, phone, company FROM customers',
                'risk_level': 'CRITICAL'
            },
            {
                'name': 'Competitor Intelligence',
                'description': 'Access pricing and inventory data',
                'query': '''
                    SELECT v.brand, v.model, p.base_price, p.final_price, v.stock_quantity
                    FROM vehicles v 
                    JOIN pricing p ON v.id = p.vehicle_id
                ''',
                'risk_level': 'HIGH'
            },
            {
                'name': 'Sales Performance Analysis',
                'description': 'Analyze employee sales performance',
                'query': '''
                    SELECT e.name, COUNT(o.id) as opportunities, 
                           SUM(t.total_amount) as revenue
                    FROM employees e
                    LEFT JOIN opportunities o ON e.id = o.opportunity_salesperson_ae_id
                    LEFT JOIN transactions t ON o.id = t.opportunity_id
                    GROUP BY e.id, e.name
                ''',
                'risk_level': 'MEDIUM'
            },
            {
                'name': 'Customer Contact Harvesting',
                'description': 'Harvest customer contact information for spam',
                'query': 'SELECT name, email, phone, mobile_number FROM customers WHERE email IS NOT NULL',
                'risk_level': 'CRITICAL'
            }
        ]
        
        attack_results = []
        
        async with self.db.connect() as conn:
            for scenario in scenarios:
                try:
                    result = await conn.execute(scenario['query'])
                    rows = result.fetchall()
                    
                    attack_results.append({
                        'name': scenario['name'],
                        'description': scenario['description'],
                        'risk_level': scenario['risk_level'],
                        'successful': True,
                        'rows_returned': len(rows),
                        'sample_data': [dict(row) for row in rows[:3]] if rows else []
                    })
                    
                except Exception as e:
                    attack_results.append({
                        'name': scenario['name'],
                        'description': scenario['description'],
                        'risk_level': scenario['risk_level'],
                        'successful': False,
                        'error': str(e)
                    })
        
        return attack_results
    
    def generate_security_recommendations(self, rls_status: Dict, access_results: List, 
                                        user_context: Dict, attack_results: List) -> List[Dict[str, Any]]:
        """Generate comprehensive security recommendations."""
        
        recommendations = []
        
        # RLS Recommendations
        unsecured_tables = [table for table, status in rls_status.items() if not status['secure']]
        if unsecured_tables:
            recommendations.append({
                'category': 'Row Level Security',
                'priority': 'CRITICAL',
                'title': 'Implement RLS on CRM Tables',
                'description': f'Tables {", ".join(unsecured_tables)} lack Row Level Security',
                'implementation': [
                    'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;',
                    'CREATE POLICY for appropriate user roles',
                    'Test policies with different user contexts'
                ],
                'affected_tables': unsecured_tables
            })
        
        # Access Control Recommendations
        violations = [r for r in access_results if r.get('security_violation', False)]
        if violations:
            recommendations.append({
                'category': 'Access Control',
                'priority': 'HIGH',
                'title': 'Implement Role-Based Access Control',
                'description': 'Sensitive data is accessible without proper authorization',
                'implementation': [
                    'Create user roles: sales_agent, manager, admin',
                    'Implement role-based RLS policies',
                    'Add user authentication middleware'
                ],
                'violations': len(violations)
            })
        
        # User Context Recommendations
        if not user_context.get('context_available', False):
            recommendations.append({
                'category': 'Authentication',
                'priority': 'CRITICAL',
                'title': 'Implement User Context in Database',
                'description': 'No user authentication context available for queries',
                'implementation': [
                    'Configure Supabase Auth',
                    'Use auth.uid() in RLS policies',
                    'Implement JWT-based authentication'
                ]
            })
        
        # Attack Prevention Recommendations
        critical_attacks = [a for a in attack_results if a.get('successful') and a.get('risk_level') == 'CRITICAL']
        if critical_attacks:
            recommendations.append({
                'category': 'Attack Prevention',
                'priority': 'CRITICAL',
                'title': 'Prevent Data Exfiltration',
                'description': 'System vulnerable to data exfiltration attacks',
                'implementation': [
                    'Implement query result size limits',
                    'Add rate limiting on database queries',
                    'Monitor for suspicious query patterns',
                    'Implement data loss prevention (DLP)'
                ],
                'vulnerable_scenarios': len(critical_attacks)
            })
        
        # Additional Security Measures
        recommendations.append({
            'category': 'Monitoring & Auditing',
            'priority': 'MEDIUM',
            'title': 'Implement Security Monitoring',
            'description': 'Add comprehensive security monitoring and auditing',
            'implementation': [
                'Enable query logging and monitoring',
                'Implement audit trails for data access',
                'Set up alerts for suspicious activities',
                'Regular security assessments'
            ]
        })
        
        return recommendations
    
    def generate_rls_policies(self, rls_status: Dict) -> Dict[str, List[str]]:
        """Generate sample RLS policies for CRM tables."""
        
        policies = {}
        
        # Customer table policies
        policies['customers'] = [
            """
            -- Allow sales agents to view customers assigned to them
            CREATE POLICY "sales_agents_own_customers" ON customers
            FOR SELECT TO sales_agent
            USING (
                id IN (
                    SELECT customer_id FROM opportunities 
                    WHERE opportunity_salesperson_ae_id = auth.uid()
                )
            );
            """,
            """
            -- Allow managers to view all customers in their branch
            CREATE POLICY "managers_branch_customers" ON customers
            FOR SELECT TO manager
            USING (
                id IN (
                    SELECT DISTINCT o.customer_id 
                    FROM opportunities o
                    JOIN employees e ON o.opportunity_salesperson_ae_id = e.id
                    WHERE e.branch_id = (
                        SELECT branch_id FROM employees WHERE id = auth.uid()
                    )
                )
            );
            """
        ]
        
        # Employee table policies
        policies['employees'] = [
            """
            -- Allow employees to view their own information
            CREATE POLICY "employees_own_info" ON employees
            FOR SELECT TO authenticated
            USING (id = auth.uid());
            """,
            """
            -- Allow managers to view employees in their branch
            CREATE POLICY "managers_branch_employees" ON employees
            FOR SELECT TO manager
            USING (
                branch_id = (
                    SELECT branch_id FROM employees WHERE id = auth.uid()
                )
            );
            """
        ]
        
        # Transaction table policies
        policies['transactions'] = [
            """
            -- Only allow managers and above to view transactions
            CREATE POLICY "managers_view_transactions" ON transactions
            FOR SELECT TO manager, director, admin
            USING (true);
            """,
            """
            -- Sales agents can only view their own transactions
            CREATE POLICY "sales_agents_own_transactions" ON transactions
            FOR SELECT TO sales_agent
            USING (opportunity_salesperson_ae_id = auth.uid());
            """
        ]
        
        return policies
    
    async def run_comprehensive_assessment(self) -> Dict[str, Any]:
        """Run the complete security assessment."""
        logger.info("Starting comprehensive CRM security assessment...")
        
        await self.setup_method() # Changed from setup() to setup_method()
        
        # Run all checks
        rls_status = await self.check_rls_policies()
        access_results = await self.test_data_access_patterns()
        user_context = await self.check_user_context()
        attack_results = await self.simulate_attack_scenarios()
        
        # Generate recommendations
        recommendations = self.generate_security_recommendations(
            rls_status, access_results, user_context, attack_results
        )
        
        # Generate sample RLS policies
        sample_policies = self.generate_rls_policies(rls_status)
        
        # Calculate security score
        total_tables = len(rls_status)
        secured_tables = sum(1 for status in rls_status.values() if status['secure'])
        security_score = (secured_tables / total_tables) * 100 if total_tables > 0 else 0
        
        successful_attacks = sum(1 for attack in attack_results if attack.get('successful', False))
        critical_vulnerabilities = sum(1 for rec in recommendations if rec['priority'] == 'CRITICAL')
        
        return {
            'assessment_date': datetime.now().isoformat(),
            'security_score': security_score,
            'summary': {
                'total_crm_tables': total_tables,
                'secured_tables': secured_tables,
                'unsecured_tables': total_tables - secured_tables,
                'successful_attacks': successful_attacks,
                'critical_vulnerabilities': critical_vulnerabilities,
                'overall_risk_level': self.calculate_risk_level(security_score, successful_attacks, critical_vulnerabilities)
            },
            'rls_status': rls_status,
            'access_patterns': access_results,
            'user_context': user_context,
            'attack_scenarios': attack_results,
            'recommendations': recommendations,
            'sample_rls_policies': sample_policies,
            'next_steps': [
                '1. Implement Row Level Security on all CRM tables',
                '2. Configure user authentication with Supabase Auth',
                '3. Create role-based access control policies',
                '4. Add query monitoring and rate limiting',
                '5. Implement audit logging for all data access',
                '6. Regular security assessments and penetration testing'
            ]
        }
    
    def calculate_risk_level(self, security_score: float, successful_attacks: int, 
                           critical_vulnerabilities: int) -> str:
        """Calculate overall risk level."""
        if security_score == 0 or critical_vulnerabilities >= 3:
            return 'CRITICAL'
        elif security_score < 50 or successful_attacks >= 2:
            return 'HIGH'
        elif security_score < 80 or successful_attacks >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'


async def run_security_assessment():
    """Run the comprehensive security assessment."""
    assessment = TestCRMSecurityAssessment()
    await assessment.setup_method()
    return await assessment.run_comprehensive_assessment()


async def save_assessment_report(results: Dict[str, Any], filename: str = None):
    """Save assessment results to a file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crm_security_assessment_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Security assessment report saved to {filename}")


def print_security_report(results: Dict[str, Any]):
    """Print a formatted security report."""
    print("\n" + "="*80)
    print("CRM SECURITY ASSESSMENT REPORT")
    print("="*80)
    print(f"Assessment Date: {results['assessment_date']}")
    print(f"Security Score: {results['security_score']:.1f}/100")
    print(f"Overall Risk Level: {results['summary']['overall_risk_level']}")
    
    print("\n" + "-"*50)
    print("EXECUTIVE SUMMARY")
    print("-"*50)
    summary = results['summary']
    print(f"• Total CRM Tables: {summary['total_crm_tables']}")
    print(f"• Secured Tables: {summary['secured_tables']}")
    print(f"• Unsecured Tables: {summary['unsecured_tables']}")
    print(f"• Successful Attack Simulations: {summary['successful_attacks']}")
    print(f"• Critical Vulnerabilities: {summary['critical_vulnerabilities']}")
    
    print("\n" + "-"*50)
    print("🚨 CRITICAL FINDINGS")
    print("-"*50)
    
    # Show unsecured tables
    unsecured = [table for table, status in results['rls_status'].items() if not status['secure']]
    if unsecured:
        print(f"❌ No Row Level Security on tables: {', '.join(unsecured)}")
    
    # Show user context issues
    if not results['user_context'].get('context_available', False):
        print("❌ No user authentication context available")
    
    # Show successful attacks
    critical_attacks = [a for a in results['attack_scenarios'] 
                       if a.get('successful') and a.get('risk_level') == 'CRITICAL']
    if critical_attacks:
        print(f"❌ {len(critical_attacks)} critical attack scenarios successful")
    
    print("\n" + "-"*50)
    print("📋 PRIORITY RECOMMENDATIONS")
    print("-"*50)
    
    critical_recs = [r for r in results['recommendations'] if r['priority'] == 'CRITICAL']
    high_recs = [r for r in results['recommendations'] if r['priority'] == 'HIGH']
    
    print("CRITICAL PRIORITY:")
    for rec in critical_recs:
        print(f"  • {rec['title']}: {rec['description']}")
    
    print("\nHIGH PRIORITY:")
    for rec in high_recs:
        print(f"  • {rec['title']}: {rec['description']}")
    
    print("\n" + "-"*50)
    print("🔧 NEXT STEPS")
    print("-"*50)
    for i, step in enumerate(results['next_steps'], 1):
        print(f"{i}. {step}")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)


if __name__ == "__main__":
    async def main():
        print("🔍 Starting CRM Security Assessment...")
        
        try:
            # Run comprehensive assessment
            results = await run_security_assessment()
            
            # Save detailed report
            await save_assessment_report(results)
            
            # Print summary report
            print_security_report(results)
            
        except Exception as e:
            print(f"❌ Assessment failed: {e}")
            logger.error(f"Assessment error: {e}", exc_info=True)
    
    # Run the assessment
    asyncio.run(main()) 