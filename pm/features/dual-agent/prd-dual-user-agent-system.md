# Product Requirements Document: Dual User Agent System

## Introduction/Overview

This feature extends our existing RAG agent system to support conversations with both **employees** and **customers**, replacing the current employee-only access model. The system will intelligently route users to appropriate agent nodes based on their user type, providing differentiated experiences while maintaining security and access controls.

Currently, our agent system only serves employees through a single verification gate. This enhancement will create a dual-path system where employees continue to have full access to internal tools and data, while customers receive a tailored experience with limited but relevant capabilities.

## Goals

1. **Extend User Access**: Support both employee and customer user types in the conversation flow
2. **Maintain Security**: Ensure customers cannot access internal company data or employee-specific tools
3. **Preserve Employee Experience**: Keep full functionality for employees while adding customer capabilities
4. **Implement Customer Outreach**: Enable employees to trigger follow-up messages to customers with proper confirmation workflows
5. **Follow LangGraph Best Practices**: Implement using LangGraph conditional edges, interrupts, and state management patterns
6. **Maintain System Simplicity**: Keep the implementation robust yet straightforward

## User Stories

### Primary User Stories

**As an employee**, I want to continue using the agent system with full access to CRM data, RAG documents, and all current tools, so that my workflow remains unchanged and efficient.

**As a customer**, I want to interact with an agent that can help me with general inquiries using company documents, so that I can get assistance without exposing internal company data.

**As an employee**, I want to trigger follow-up messages to specific customers through the agent system, so that I can efficiently manage customer relationships and outreach.

**As an employee**, I want the system to ask for confirmation before sending messages to customers, so that I can review and approve communications before they are sent.

### Secondary User Stories

**As a system administrator**, I want users who are neither employees nor customers to receive a graceful fallback message, so that the system handles edge cases professionally.

**As a customer**, I want to receive relevant information from company documents without being exposed to employee-specific data or internal operations.

## Functional Requirements

### Core User Verification and Routing

1. **Replace Employee Verification Node**: Transform `employee_verification_node` into `user_verification_node` that identifies user type from the `users` table
2. **User Type Detection**: System must distinguish between `employee`, `customer`, and `unknown` user types using the existing database schema
3. **Conditional Routing**: Implement conditional edges to route verified users to appropriate agent nodes:
   - Employees → `employee_agent` node  
   - Customers → `customer_agent` node
   - Unknown users → Graceful fallback response

### Agent Node Differentiation

4. **Employee Agent Node**: Maintain current functionality with full tool access including:
   - `simple_query_crm_data` for CRM database queries
   - `simple_rag` for document retrieval
   - All existing memory management and context features
   - New customer messaging tool

5. **Customer Agent Node**: Provide limited but useful functionality including:
   - `simple_rag` for customer-relevant document retrieval
   - `simple_query_crm_data` with restricted access to vehicles and pricing tables only (no employees, opportunities, or customer data)
   - Same memory management capabilities for conversation continuity

6. **Tool Access Control**: Implement tool filtering based on user type:
   - Customers: Allow `simple_rag` (full access) and `simple_query_crm_data` (vehicles/pricing tables only)
   - Employees: Full access to all tools including complete CRM data and administrative functions
   - Block customer access to: employee data, customer records, sales opportunities, internal operations

### Customer Outreach System

7. **Customer Messaging Tool**: Create a new tool available only to employee agents for triggering customer outreach
8. **Interrupt-Based Confirmation with Delivery Feedback**: Use LangGraph's `interrupt()` mechanism to:
   - Pause execution when customer messaging is triggered
   - Present confirmation dialog to employee with message details and action options
   - If confirmed: Attempt actual message delivery to the customer
   - Collect success/failure status of the delivery attempt
   - Resume execution only after gathering delivery feedback, providing the employee with:
     - Success confirmation and delivery details
     - Cancellation acknowledgment (if user cancelled)
     - Failure notification with error details (if delivery failed)

### Memory and State Management
9. **User Context Enhancement**: Extend existing user context system in `tools.py` to cache user type after first verification lookup, providing efficient access control without polluting the persistent AgentState

10. **Memory Isolation**: Ensure customer conversations don't interfere with employee memory and vice versa

### Error Handling and Fallbacks

11. **Unknown User Handling**: For users not found in the database or with unsupported user types, return a professional message directing them to appropriate channels

12. **Database Error Handling**: Gracefully handle database connection issues during user verification

13. **Tool Execution Safety**: Ensure customer agents cannot execute restricted tools even if attempted

## Non-Goals (Out of Scope)

1. **Customer Self-Registration**: This feature does not include customer signup or account creation workflows
2. **Advanced Customer Features**: No customer-specific advanced features like appointment booking, order tracking, etc.
3. **Multi-Channel Support**: Focus remains on the existing conversation interface, no new channels (email, SMS, etc.)
4. **Customer Data Management**: No customer profile editing or data management capabilities  
5. **Real-Time Notifications**: Customer messaging will not include real-time delivery notifications or read receipts
6. **Bulk Customer Messaging**: One-to-one messaging only, no broadcast or bulk messaging features

## Technical Considerations

### Database Schema
- Leverage existing `users` table structure with `user_type`, `employee_id`, and `customer_id` columns
- No database schema changes required

### LangGraph Implementation
- Use conditional edges for user type routing
- Implement interrupt patterns for customer messaging confirmation
- Maintain existing checkpointing and memory management

### Tool Architecture  
- Extend existing tool system with user-context-aware filtering
- Create new customer messaging tool using existing tool patterns
- Maintain backward compatibility with current employee tools

### Security
- Implement access control at the tool level, not just UI level
- Ensure customer context cannot access employee-specific data
- Validate user permissions on every tool execution

## Success Metrics

### Functional Success
- **User Routing Accuracy**: 100% correct routing of employees vs customers to appropriate agent nodes
- **Security Compliance**: 0% successful customer access attempts to restricted CRM tools
- **Employee Workflow Continuity**: 100% preservation of existing employee functionality

### Performance Success
- **Response Time**: Customer agent responses within 3 seconds average
- **Employee Tool Access**: No degradation in employee tool response times
- **Memory Management**: Successful conversation persistence for both user types

### User Experience Success
- **Customer Satisfaction**: Positive feedback on customer agent interactions from beta testing
- **Employee Adoption**: 100% of current employee users successfully transitioned to dual system
- **Error Rate**: Less than 1% fallback to unknown user handling in production

## Open Questions

1. **Customer Document Filtering**: Should customers have access to all company documents or only a filtered subset? (Recommendation: Start with all, add filtering if needed)

2. **Customer Message Templates**: Should the customer messaging tool include predefined templates or be completely free-form? (Recommendation: Start free-form, add templates based on usage patterns)

3. **Interrupt Timeout**: How long should the system wait for employee confirmation during customer messaging interrupts? (Recommendation: 5 minutes with option to extend)

4. **Customer Rate Limiting**: Should there be rate limits on customer queries to prevent abuse? (Recommendation: Monitor initially, implement if needed)

5. **Cross-User Context**: Should customers be able to reference previous interactions with different employees? (Recommendation: Yes, through existing user context system)

## Implementation Priority

### Phase 1 (Core Functionality)
- User verification and routing system
- Basic customer agent with RAG access
- Employee agent with existing functionality
- Unknown user fallback handling

### Phase 2 (Enhanced Features)  
- Customer messaging tool with interrupt system
- Confirmation workflows and notifications
- Tool access control refinements

### Phase 3 (Optimization)
- Performance monitoring and optimization
- Enhanced error handling
- User experience improvements based on feedback 