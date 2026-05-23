from __future__ import annotations

from interview_domain import (
    DifficultyLevel,
    FollowUp,
    InMemoryInterviewStore,
    InterviewMode,
    InterviewTemplate,
    Question,
    Rubric,
    RubricDimension,
)

DEFAULT_DSA_HIRING_TEMPLATE_ID = "dsa_hiring_core_v1"
DEFAULT_DSA_PRACTICE_TEMPLATE_ID = "dsa_practice_core_v1"


def base_dsa_rubric() -> Rubric:
    return Rubric(
        dimensions=[
            RubricDimension(
                key="problem_understanding",
                label="Problem understanding",
                description="Understands the problem statement, constraints, and target output.",
            ),
            RubricDimension(
                key="algorithm_choice",
                label="Algorithm choice",
                description="Chooses a reasonable approach and justifies trade-offs.",
            ),
            RubricDimension(
                key="correctness",
                label="Correctness",
                description="Reasons accurately about why the approach works.",
            ),
            RubricDimension(
                key="complexity",
                label="Complexity",
                description="Explains time and space complexity clearly.",
            ),
            RubricDimension(
                key="edge_cases",
                label="Edge cases",
                description="Identifies edge cases, failure modes, and input boundaries.",
            ),
            RubricDimension(
                key="communication",
                label="Communication",
                description="Communicates clearly, in sequence, and with useful precision.",
            ),
        ],
        fail_conditions=[
            "Provides a memorized answer without reasoning.",
            "Cannot justify correctness or complexity.",
            "Ignores key constraints or major edge cases.",
        ],
    )


def build_dsa_questions() -> list[Question]:
    rubric = base_dsa_rubric()
    return [
        Question(
            id="arrays_strings_longest_unique_substring",
            topic="arrays_strings",
            title="Longest substring without repeats",
            prompt=(
                "You are given a string and need the length of the longest substring with no repeated characters. "
                "Walk me through your approach, how you would track the window, and the complexity."
            ),
            difficulty=DifficultyLevel.MEDIUM,
            time_cap_seconds=360,
            expected_concepts=["sliding window", "hash map", "window invariants", "O(n) scan"],
            follow_ups=[
                FollowUp(
                    id="arrays_strings_longest_unique_substring_followup_unicode",
                    prompt="How would your approach change if the input were a very large Unicode stream?",
                    intent="Probe scalability and assumptions about character space.",
                    trigger="candidate_gives_standard_hashmap_solution",
                ),
                FollowUp(
                    id="arrays_strings_longest_unique_substring_followup_indices",
                    prompt="What exact invariant are you maintaining for the left pointer when you see a duplicate?",
                    intent="Check correctness of window updates.",
                    trigger="candidate_solution_is_hand_wavy",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Resets the window incorrectly on duplicates.",
                "Uses an O(n^2) approach without acknowledging the trade-off.",
            ],
        ),
        Question(
            id="hashing_top_k_frequent",
            topic="hashing",
            title="Top K frequent elements",
            prompt=(
                "Given an integer array and an integer k, explain how you would return the k most frequent elements. "
                "Compare at least two possible approaches and justify which one you would choose."
            ),
            difficulty=DifficultyLevel.MEDIUM,
            time_cap_seconds=360,
            expected_concepts=["frequency map", "heap", "bucket sort trade-off", "selection strategy"],
            follow_ups=[
                FollowUp(
                    id="hashing_top_k_frequent_followup_memory",
                    prompt="If memory were tight but n were large, what trade-off would you make?",
                    intent="Test trade-off reasoning.",
                    trigger="candidate_picks_heap_or_bucket_solution",
                ),
                FollowUp(
                    id="hashing_top_k_frequent_followup_ties",
                    prompt="How would you handle ties or unstable ordering requirements?",
                    intent="Check handling of output ambiguity.",
                    trigger="candidate_ignores_output_contract_details",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Cannot explain why the chosen data structure is appropriate.",
                "Confuses element frequency with element value ordering.",
            ],
        ),
        Question(
            id="two_pointers_sorted_two_sum",
            topic="two_pointers",
            title="Two Sum on a sorted array",
            prompt=(
                "Suppose the input array is already sorted and you need to find whether two numbers sum to a target. "
                "Explain the two-pointer approach, why it works, and when you would still prefer hashing."
            ),
            difficulty=DifficultyLevel.EASY,
            time_cap_seconds=300,
            expected_concepts=["two pointers", "sorted invariant", "trade-off vs hashing", "O(n) time"],
            follow_ups=[
                FollowUp(
                    id="two_pointers_sorted_two_sum_followup_duplicates",
                    prompt="What changes if the array contains duplicates and you need all valid pairs?",
                    intent="Extend a standard pattern into a richer variant.",
                    trigger="candidate_solves_base_problem_cleanly",
                ),
                FollowUp(
                    id="two_pointers_sorted_two_sum_followup_unsorted",
                    prompt="How would your approach differ if the input were not sorted and could not be modified?",
                    intent="Check adaptability across constraints.",
                    trigger="candidate_needs_comparison_with_hashing",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Cannot justify pointer movement decisions.",
                "Breaks correctness when duplicate values appear.",
            ],
        ),
        Question(
            id="stack_queue_valid_parentheses",
            topic="stack_queue",
            title="Valid parentheses checker",
            prompt=(
                "Describe how you would determine whether a string of brackets is valid. "
                "Focus on the stack operations, invalid states, and how you would explain correctness."
            ),
            difficulty=DifficultyLevel.EASY,
            time_cap_seconds=300,
            expected_concepts=["stack", "matching pairs", "early invalid detection", "empty stack condition"],
            follow_ups=[
                FollowUp(
                    id="stack_queue_valid_parentheses_followup_stream",
                    prompt="How would you adapt this if characters arrived as a stream?",
                    intent="Probe state management under streaming input.",
                    trigger="candidate_solves_base_problem_cleanly",
                ),
                FollowUp(
                    id="stack_queue_valid_parentheses_followup_custom_pairs",
                    prompt="How would you support arbitrary bracket pairs supplied at runtime?",
                    intent="Check abstraction and extensibility.",
                    trigger="candidate_uses_hardcoded_logic",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Fails to reject mismatched closing brackets.",
                "Does not account for leftover open brackets at the end.",
            ],
        ),
        Question(
            id="trees_graphs_bfs_shortest_path_grid",
            topic="trees_graphs",
            title="Shortest path in a grid",
            prompt=(
                "Imagine a grid with blocked and open cells, and you need the shortest path from the top-left to the "
                "bottom-right. Explain how you would model the problem, why BFS is appropriate, and what state you track."
            ),
            difficulty=DifficultyLevel.MEDIUM,
            time_cap_seconds=420,
            expected_concepts=["BFS", "graph modeling", "visited set", "level order distance"],
            follow_ups=[
                FollowUp(
                    id="trees_graphs_bfs_shortest_path_grid_followup_weighted",
                    prompt="What changes if moving into cells has different costs?",
                    intent="Test whether the candidate knows when BFS stops being valid.",
                    trigger="candidate_solves_base_problem_cleanly",
                ),
                FollowUp(
                    id="trees_graphs_bfs_shortest_path_grid_followup_space",
                    prompt="How would you think about the memory cost of the queue and visited structure?",
                    intent="Probe complexity depth.",
                    trigger="candidate_omits_space_analysis",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Uses DFS while still claiming shortest path guarantees.",
                "Does not track visited state correctly.",
            ],
        ),
        Question(
            id="dynamic_programming_coin_change_min_coins",
            topic="dynamic_programming",
            title="Minimum coins for target amount",
            prompt=(
                "Given coin denominations and a target amount, explain how you would find the minimum number of coins "
                "needed to make that amount. Talk through the recurrence, base cases, and iteration order."
            ),
            difficulty=DifficultyLevel.HARD,
            time_cap_seconds=480,
            expected_concepts=["dynamic programming", "state definition", "recurrence", "unreachable states"],
            follow_ups=[
                FollowUp(
                    id="dynamic_programming_coin_change_min_coins_followup_greedy",
                    prompt="Why does a greedy choice fail for some coin systems?",
                    intent="Check conceptual understanding beyond formula recall.",
                    trigger="candidate_jumps_directly_to_dp",
                ),
                FollowUp(
                    id="dynamic_programming_coin_change_min_coins_followup_reconstruction",
                    prompt="How would you reconstruct which coins were chosen, not just the count?",
                    intent="Extend the DP to solution reconstruction.",
                    trigger="candidate_solves_base_problem_cleanly",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "Cannot define a stable DP state or recurrence.",
                "Overlooks unreachable amounts or invalid base cases.",
            ],
        ),
        Question(
            id="complexity_analysis_nested_loops_membership",
            topic="complexity_analysis",
            title="Reasoning about complexity trade-offs",
            prompt=(
                "Suppose one solution uses nested loops to compare each element against all others, and another first "
                "builds a hash-based lookup structure. Explain how you would analyze the time-space trade-off and decide "
                "which approach is more appropriate."
            ),
            difficulty=DifficultyLevel.MEDIUM,
            time_cap_seconds=300,
            expected_concepts=["Big-O analysis", "trade-offs", "memory vs runtime", "constraint-driven choice"],
            follow_ups=[
                FollowUp(
                    id="complexity_analysis_nested_loops_membership_followup_constants",
                    prompt="When might the asymptotically better solution still lose in practice?",
                    intent="Probe practical complexity reasoning.",
                    trigger="candidate_gives_only_big_o_labels",
                ),
                FollowUp(
                    id="complexity_analysis_nested_loops_membership_followup_sorted_input",
                    prompt="How would your answer change if the data were already sorted?",
                    intent="Check whether the candidate re-evaluates assumptions.",
                    trigger="candidate_ignores alternative structure in input",
                ),
            ],
            rubric=rubric,
            fail_conditions=[
                "States complexity labels without explaining why.",
                "Ignores the extra memory cost of the lookup structure.",
            ],
        ),
    ]


def build_default_templates() -> list[InterviewTemplate]:
    question_ids = [question.id for question in build_dsa_questions()]
    return [
        InterviewTemplate(
            id=DEFAULT_DSA_HIRING_TEMPLATE_ID,
            name="DSA Hiring Core",
            mode=InterviewMode.HIRING,
            topic_family="dsa",
            question_ids=question_ids,
            default_question_count=6,
            default_time_cap_seconds=360,
        ),
        InterviewTemplate(
            id=DEFAULT_DSA_PRACTICE_TEMPLATE_ID,
            name="DSA Practice Core",
            mode=InterviewMode.PRACTICE,
            topic_family="dsa",
            question_ids=question_ids,
            default_question_count=6,
            default_time_cap_seconds=360,
        ),
    ]


def seed_question_bank(store: InMemoryInterviewStore) -> None:
    questions = build_dsa_questions()
    for question in questions:
        store.questions[question.id] = question

    for template in build_default_templates():
        store.templates[template.id] = template


def default_template_id_for_mode(mode: InterviewMode) -> str:
    if mode == InterviewMode.PRACTICE:
        return DEFAULT_DSA_PRACTICE_TEMPLATE_ID
    return DEFAULT_DSA_HIRING_TEMPLATE_ID
