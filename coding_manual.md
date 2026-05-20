# Coding manual
## Information Dimension Taxonomy

### Dimension: Problem Understanding and Comprehension

**Definition**: Information that helps beginners comprehend the issue scope, reproduce problems, understand the technical context, expected behavior, underlying causes, and validate problem existence through descriptions, root cause analysis, behavioral evidence, and reproduction guidance


**Solving Stage**: Problem Understanding


**Cognitive Function**: Enables newcomers to grasp technical problems, identify what needs fixing, understand why current behavior is incorrect, verify issue existence, and establish context for effective problem diagnosis and analysis


#### Category: Problem Reproduction and Validation

Information that enables beginners to replicate issues, verify problem existence, validate problem status, and understand root causes through concrete examples, evidence, and validation steps


- **Visual Evidence**: Screenshots, videos, or GIFs that visually demonstrate the issue manifestation and current behavior

- **Bug Reproduction Steps and Guidance**: Step-by-step instructions and guidance to recreate the issue in a development environment with specific conditions and configurations

- **Environment-Specific Reproduction**: Instructions tailored to specific platforms, browsers, versions, or environmental conditions where the issue occurs

- **Error Reproduction Examples**: Concrete code snippets and demonstrations showing how to reproduce the reported issue or error

- **Status Confirmation**: Information confirming whether the issue is reproducible, already fixed, or requires specific actions

- **Problem Diagnosis**: Assessment clarifying whether the issue represents a bug, documentation error, or expected behavior


#### Category: Context and Root Cause Analysis

Information that provides technical background, system behavior context, historical context, domain knowledge, and explains the underlying reasons for the issue


- **Root Cause Analysis**: Explanation of the underlying technical reasons, mechanisms, and fundamental causes behind the issue occurrence

- **Problem Context and Impact**: Background information explaining why the issue matters, its real-world consequences, and user impact

- **Implementation Context**: Background information about existing code patterns, system capabilities, technical constraints, and architectural decisions

- **Behavioral Context**: Explanations of system behavior, user workflows, real-world usage scenarios, and functional requirements

- **Historical Context**: References to previous implementations, related issues, code change history, and evolution of functionality

- **Domain Terminology Guidance**: Explanation of project-specific terms, conceptual frameworks, and domain knowledge essential for understanding the problem

- **Behavioral Evidence**: Demonstrations of current software behavior through code examples, visual evidence, and concrete examples showing discrepancies between expected and actual outcomes


#### Category: Expected Behavior and Scope

Clear specifications of what the system should do when functioning correctly, including success criteria, acceptance criteria, and scope boundaries


- **Problem Specification**: Clear statements defining the exact functionality gap, standard violation, or unexpected behavior that needs resolution

- **Behavior Specification**: Explicit descriptions of correct system functionality, acceptance criteria, and target outcomes

- **Scope Definition**: Clear boundaries of what should be included in the solution, what should not, and implementation constraints


#### Category: Code Location Specification

Information that precisely identifies where issues occur in the code through specific file paths, line numbers, and components requiring modification


- **Code Location Specification**: Specific file paths, line numbers, code segments, and components that require modification


### Dimension: Solution Design and Approach

**Definition**: Information types that guide beginners in designing appropriate solutions, including technical strategies, implementation patterns, architectural decisions, and validation approaches


**Solving Stage**: Solution Design


**Cognitive Function**: Helps newcomers design appropriate solutions by providing implementation patterns, architectural guidance, solution validation criteria, and technical decision-making support


#### Category: Solution Strategy Guidance

High-level approaches, technical strategies, and architectural decisions for solving the problem effectively


- **Solution Approach**: High-level technical strategies, implementation methodologies, and solution directions

- **Architectural Guidance**: System design decisions, code organization patterns, architectural constraints, and integration approaches

- **Problem-Solving Strategy**: Methodologies for overcoming obstacles, structured approaches to complex problems, and troubleshooting guidance

- **Alternative Solution Evaluation**: Comparison of different implementation approaches with rationale for preferred choices and tradeoff analysis

- **Implementation Strategy**: Guidance on the overall technical approach, design patterns, and architectural considerations for the solution

- **Solution Goals Clarification**: Clear definition of success criteria, project priorities, implementation constraints, and functional requirements for the solution

- **Solution Validation and Approval**: Confirmation that proposed solutions work effectively, alignment with maintainer expectations, and validation of solution approaches


#### Category: Implementation Reference and Planning

Existing code examples, patterns, references, and templates that demonstrate proven implementation approaches and design consistency, along with structured implementation plans


- **Example Reference**: Concrete, working examples from the codebase that demonstrate implementation patterns, syntax, and usage

- **Design Consistency Guidance**: Feedback and guidance on maintaining visual, functional, and architectural consistency with existing patterns and project standards

- **Implementation Techniques and Patterns**: Specific coding patterns, methods, technical approaches, and best practices to use in implementation

- **Implementation Plan**: Structured breakdown of required changes across codebase components with specific steps and sequences

- **Alternative Solution Options**: Multiple potential approaches, variations, or alternative methods for solving the same problem

- **Prior Work Reference**: Links to existing implementations, similar fixes, related work, or code patterns that can serve as templates

- **Workaround Demonstration**: Alternative implementation approaches that bypass the immediate problem when direct fixes are complex, or temporary solutions

- **Implementation Examples**: Working code examples, snippets, and reference implementations demonstrating correct approaches


#### Category: Solution Validation

Information that helps validate solution approaches, confirm implementation correctness, and ensure alignment with project standards


- **Implementation Approval**: Formal validation and authorization of solution approaches from project maintainers and experts

- **Solution Validation Criteria**: Specific criteria, test cases, verification methods, and success metrics to verify solution correctness

- **Validation and Testing Strategy**: Approaches for verifying solution correctness, performance characteristics, and quality assurance methods


### Dimension: Implementation and Verification Support

**Definition**: Information types that provide specific implementation details, code location guidance, testing procedures, verification steps, and quality assurance processes


**Solving Stage**: Implementation and Verification


**Cognitive Function**: Enables newcomers to efficiently navigate codebases, make correct code changes, set up necessary development environments, execute implementations, and verify solution correctness


#### Category: Technical Implementation Guidance

Specific technical instructions, step-by-step implementation details, and code modification guidance


- **Technical Implementation Steps**: Specific, actionable instructions for implementing the solution in code with detailed procedures and commands

- **Code Location Guidance**: Specific file paths, function names, code locations, and repository structure requiring modification

- **Solution Correction**: Feedback identifying incorrect implementation approaches, syntax errors, and providing specific corrections

- **Exact Code Fix Suggestion**: Precise code changes, function calls, implementation details, or patches needed to resolve the issue

- **Task Clarification**: Explicit statements defining specific changes required, scope boundaries, implementation expectations, and work requirements

- **Implementation Guidance**: Technical direction on how to implement solutions using appropriate patterns, methods, and technical approaches


#### Category: Testing and Verification

Information about testing methodologies, verification requirements, quality assurance processes, and debugging procedures


- **Testing Configuration**: Specific testing setups, environment configurations, debugging tips, and test execution procedures

- **Verification Requirements**: Specific evidence, validation steps, success criteria, and proof needed to demonstrate the solution works correctly

- **Test Implementation Guidance**: Instructions on where and how to implement tests in the codebase, including structure, assertions, and project conventions

- **Test Case Construction**: Instructions for building appropriate test cases, including expected outputs, validation scenarios, and test inputs

- **Testing Procedure**: Specific commands, procedures, and methodologies for executing tests and validating results

- **Reproducible Test Cases**: Minimal, executable code examples that reliably demonstrate the issue for validation and verification

- **Validation Tool Reference**: References to scripts, tools, procedures, or automated systems used to validate solution correctness

- **Solution Verification Methods**: Approaches and techniques for testing, validating, and confirming that solutions work correctly in real scenarios

- **Error Identification**: Specific error messages, failure conditions, stack traces, and diagnostic information that help identify problems

- **Prevention Guidance**: Instructions for avoiding common pitfalls, implementation mistakes, and maintaining code standards


#### Category: Environment and Process Support

Information about contribution workflows, project procedures, collaboration processes, and development environment configuration


- **Process Guidance**: Instructions about project contribution workflows, commit standards, submission processes, and administrative procedures

- **Collaboration Support**: Offers of help, community resources, support channels, mentorship, and collaborative guidance for beginners

- **Environment Configuration**: Instructions for setting up development environments, configuring tools, dependencies, and resolving setup issues

- **Setup Instructions**: Step-by-step procedures for configuring development environments, installing dependencies, and preparing systems for work

- **Tool Usage Guidance**: Instructions for using development tools, debugging utilities, and project-specific software effectively


### Dimension: Project Environment and Knowledge Foundation

**Definition**: Information types that help newcomers understand project conventions, documentation standards, contribution workflows, and maintain code quality


**Solving Stage**: Cross-stage


**Cognitive Function**: Helps newcomers navigate project ecosystems, understand community standards, follow established processes, and maintain consistent quality across all contribution stages


#### Category: Quality and Standards

Information about code quality requirements, project standards, and quality assurance processes


- **Code Quality and Standards**: Requirements for code style, formatting, linting rules, and quality benchmarks

- **Contribution Requirements**: Specific project requirements for contributions, including legal, procedural, and quality gates

- **Quality Assurance Requirements**: Specific testing, validation, and quality assurance processes required for contributions

- **Coding Standards**: Project-specific coding conventions, style guidelines, and implementation patterns

- **Technical Conventions**: Project-specific technical practices, architectural patterns, and implementation conventions


#### Category: Documentation and Reference

Information sources, documentation references, and knowledge resources for understanding project context and implementation details


- **Documentation References**: Links to relevant documentation, API references, and external resources for understanding project functionality

- **Resource Links**: References to external resources, tutorials, documentation, and learning materials relevant to the problem

- **Process Documentation Reference**: References to project documentation explaining contribution processes, workflow procedures, and administrative guidelines

- **Documentation Standards**: Requirements and guidelines for documentation quality, format, and content standards

- **Existing Documentation Reference**: Pointers to existing project documentation, README files, and established knowledge resources


#### Category: Workflow and Process Support

Information about contribution workflows, collaboration processes, task management, and project procedures


- **Contribution Process Guidance**: Instructions for navigating contribution workflows, submission procedures, and project-specific processes

- **Task Management and Prioritization**: Guidance on task sequencing, priority assessment, and workload management for efficient contribution

- **Workflow Best Practice**: Recommended approaches for efficient workflow management, collaboration patterns, and productivity tips

- **Process Guidance**: Instructions for following project-specific procedures, administrative workflows, and contribution protocols

- **Task Assignment and Collaboration**: Coordination of work distribution, responsibility assignment, and collaborative development processes

