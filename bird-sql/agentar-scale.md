**Pseudocode: Agentar-Scale-SQL with**

**Input:**

- *Munified*: The single LLM used for all reasoning, generation, and selection tasks.
- *Qu*: The user question (Natural Language).
- *Eu*: Any associated evidence/context.
- *D*: The target database.

**Offline Resources (Preprocessed):**

- *SchemaDDL*: Database metadata in standard DDL format.
- *SchemaLight*: Database metadata in Markdown-based light schema format.
- *VDcell*: Vector store for retrieving relevant database cell values.
- *VDexample*: Vector store for retrieving few-shot ICL examples.

**Step 1: Task Understanding**

This stage establishes context by retrieving relevant database cells and few-shot examples.

```
FUNCTION Task_Understanding(Q_u, E_u, Schema_Light):
    // 1. Keyword Extraction (LLM used for initial analysis)
    Keywords = M_unified.generate_keywords(Prompt: "Extract keywords from Q_u and E_u.") [7]

    // 2. Retrieve Relevant Cells (using keywords and vector store V_Dcell)
    Relevant_Cells = V_Dcell.retrieve_context(Keywords) [7]

    // 3. Skeleton Extraction (LLM used to identify query structure)
    Question_Skeleton = M_unified.generate_skeleton(Prompt: "Extract the SQL structure/skeleton implied by Q_u.") [7]

    // 4. Retrieve Few-shot Examples (using skeleton and vector store V_Dexample)
    Few_Shot_Examples = V_Dexample.retrieve_examples(Question_Skeleton) [7]

    RETURN Relevant_Cells, Few_Shot_Examples

```

**Step 2: SQL Generation Scaling**

This stage uses Parallel Scaling (Diverse Synthesis) and Sequential Scaling (Iterative Refinement) to create a diverse and refined pool of SQL candidates, *C*.

```
FUNCTION SQL_Generation_Scaling(Q_u, Relevant_Cells, Few_Shot_Examples, Schema_DDL, Schema_Light):
    C = Empty List
    N_candidates = 17 // Total candidates generated (e.g., 9 ICL + 8 Reasoning, as in sources) [4]

    // --- Parallel Scaling: Diverse Synthesis --- [2]

    // A. Simulate Reasoning Generator ($M_{reasoning}$) - Focus on DDL schema and intrinsic reasoning
    FOR i FROM 1 TO N_reasoning_candidates:
        // Use a prompt emphasizing deep, step-by-step reasoning (Internal Scaling) [9]
        Prompt_Reasoning = "Generate SQL using DDL schema and step-by-step reasoning. Context: Schema_DDL, Relevant_Cells, Q_u."
        SQL_R = M_unified.generate(Prompt_Reasoning, Schema_DDL, temperature=0.2) // Low temperature for high precision/reasoning
        C.append(SQL_R)

    // B. Simulate ICL Generator ($M_{ICL}$) - Focus on Light Schema and few-shot examples
    FOR j FROM 1 TO N_ICL_candidates:
        // Vary prompt style (Direct, CoT, Decomposition) and temperature for diversity [5]
        Prompt_ICL = Choose_Prompt_Style(j) + "Generate SQL using Light Schema and Few_Shot_Examples. Context: Schema_Light, Relevant_Cells, Q_u."
        SQL_I = M_unified.generate(Prompt_ICL, Schema_Light, temperature=Choose_Temperature(j))
        C.append(SQL_I)

    // --- Sequential Scaling: Iterative Refinement --- [1, 2, 10]

    Refined_C = Empty List
    FOR each SQL_Candidate in C:
        Execution_Result = Execute_SQL(SQL_Candidate, D)

        // 1. Syntax Repair (Simulating SQL Fixer) [10]
        IF Syntax_Error(Execution_Result):
            Prompt_Fixer = "The SQL failed due to a syntax error. Fix the syntax error in SQL_Candidate."
            SQL_Fixed = M_unified.generate(Prompt_Fixer, SQL_Candidate)
            SQL_Candidate = SQL_Fixed
            Execution_Result = Execute_SQL(SQL_Candidate, D)

        // 2. Semantic Edits (Simulating SQL Reviser) [10]
        IF Semantic_Error_Suspected(Execution_Result): // Requires heuristic check or comparison
            Prompt_Revisor = "The SQL executed but the result seems logically flawed/unexpected. Revise SQL_Candidate based on Q_u and Relevant_Cells."
            SQL_Revised = M_unified.generate(Prompt_Revisor, SQL_Candidate)
            SQL_Candidate = SQL_Revised

        Refined_C.append(SQL_Candidate)

    RETURN Refined_C

```

**Step 3: SQL Selection Scaling**

This stage consolidates candidates based on execution results and uses a Tournament Selection mechanism driven by *Munified* acting as the Reasoning Selector.

```
FUNCTION SQL_Selection_Scaling(Refined_C, Q_u, Schema_Light):
    // 1. Candidate Consolidation
    Grouped_Candidates = Group_by_Execution_Result(Refined_C, D) // Execution on database D
    C_prime = Select_Representative(Grouped_Candidates) // Select one representative from each group [11]

    // 2. Tournament Selection (Pairwise Comparative Ranking) [11]
    Scores = Initialize_Scores(C_prime)

    FOR each pair (c_i, c_j) in C_prime:
        // Prompt M_unified to act as the Reasoning Selector ($M_{selection}$) [12]
        // This requires M_unified to compare two SQL candidates based on context
        Prompt_Selector = "You are a Reasoning Selector. Which SQL query, c_i or c_j, better answers Q_u given the Schema_Light? Justify your choice."

        // Context includes Q_u, Schema_Light, and execution results of c_i and c_j [12]
        Winner = M_unified.generate(Prompt_Selector, Q_u, Schema_Light, c_i, c_j)

        // 3. Update scores
        Scores[Winner] += 1

    // 4. Final Selection
    C_final = argmax(Scores) [12]

    RETURN C_final

```

**Main Agentar-Scale-SQL Flow**

```
PROCEDURE Agentar_Scale_SQL_Inference(Q_u, E_u, D, M_unified):
    // 1. Task Understanding
    Relevant_Cells, Few_Shot_Examples = Task_Understanding(Q_u, E_u, Schema_Light) [8]

    // 2. SQL Generation Scaling (Diverse Synthesis and Iterative Refinement)
    SQL_Candidates = SQL_Generation_Scaling(Q_u, Relevant_Cells, Few_Shot_Examples, Schema_DDL, Schema_Light) [8]

    // 3. SQL Selection Scaling (Tournament Selection)
    Final_SQL = SQL_Selection_Scaling(SQL_Candidates, Q_u, Schema_Light) [8]

    OUTPUT Final_SQL
```