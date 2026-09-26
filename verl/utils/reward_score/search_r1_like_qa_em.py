# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import os
import random
import re
import string


def is_valid_sequence(text):
    """
    Check that a trajectory follows the expected ReAct/tool-use format:
    1. Start with <thought>.
    2. Follow a thought with <tool_call> or <answer>.
    3. Follow a tool call with <tool_response>.
    4. Resume thinking after each tool response.
    5. Finish with <answer>...</answer>.
    """

    # 1. Check balanced tags.
    tags_to_check = ["thought", "tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", text))
        closing_count = len(re.findall(f"</{tag}>", text))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # 2. Split the text into content and opening/closing tags.
    split_pattern = r"(</?(?:thought|tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, text)

    # 3. Validate the sequence with a state machine.
    state = "expect_thought"

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check whether this part is a tag.
        if re.match(r"^</?(?:thought|tool_call|tool_response|answer)>$", part):
            tag = part

            # --- State transitions ---

            # Expect a thought at the start or after a tool response.
            if state == "expect_thought":
                if tag == "<thought>":
                    state = "in_thought"
                else:
                    return False, f"Expected <thought> at start or after tool response, but found {tag}"

            # Inside a thought.
            elif state == "in_thought":
                if tag == "</thought>":
                    state = "after_thought"
                else:
                    return False, f"Inside <thought>, expected </thought> but found {tag}"

            # After a thought, either call a tool or answer.
            elif state == "after_thought":
                if tag == "<tool_call>":
                    state = "in_tool_call"
                elif tag == "<answer>":
                    state = "in_answer"
                else:
                    return False, f"After </thought>, expected <tool_call> or <answer>, but found {tag}"

            # Inside a tool call.
            elif state == "in_tool_call":
                if tag == "</tool_call>":
                    state = "after_tool_call"
                else:
                    return False, f"Inside <tool_call>, expected </tool_call> but found {tag}"

            # Wait for the tool response.
            elif state == "after_tool_call":
                if tag == "<tool_response>":
                    state = "in_tool_response"
                else:
                    return False, f"After </tool_call>, expected <tool_response>, but found {tag}"

            # Inside a tool response.
            elif state == "in_tool_response":
                if tag == "</tool_response>":
                    # A tool response must be followed by another thought.
                    state = "expect_thought"
                else:
                    return False, f"Inside <tool_response>, expected </tool_response> but found {tag}"

            # Inside the final answer.
            elif state == "in_answer":
                if tag == "</answer>":
                    state = "end"
                else:
                    return False, f"Inside <answer>, expected </answer> but found {tag}"

            # The sequence has ended.
            elif state == "end":
                return False, f"Found extra tag {tag} after sequence ended"

        else:
            # Free text outside tags has no additional constraints here.
            pass

    # 4. Require a complete final answer.
    if state != "end":
        return False, f"Incomplete sequence, ended in state: {state}"

    return True, "Valid sequence format"

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth,extra_info, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer_lst = extract_solution(solution_str=solution_str)

    efficiency_score = 0
    answer = None
    if answer_lst and len(answer_lst) >=2:
        efficiency_score = 0
        answer = answer_lst[-1].group(1).strip()
    elif answer_lst and len(answer_lst) ==1:
        efficiency_score = 0.5
        answer = answer_lst[-1].group(1).strip()
    elif answer_lst is None:
        efficiency_score = 0
        answer = None

    open_count, close_count = count_answer_tags(solution_str)

    is_format_correct = is_valid_sequence(solution_str)
    try :
        if is_format_correct[0]:
            format_score = 0.1
        else:
            format_score = 0
    except:
        print(f"error{is_format_correct}")

    debug_reward = os.getenv("VERL_DEBUG_REWARD", "0") == "1"
    do_print = debug_reward and random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")
        print(f"format_score:{format_score}")
        print(f"extra_info:{extra_info}")

    res_dic = {}
    res_dic['format_score'] = format_score
    res_dic['efficiency_score'] = efficiency_score
    res_dic['origin_score'] = 0

    if answer is None:
        res_dic['score'] = 0
        return res_dic
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                res_dic['score'] = score+format_score
                res_dic['origin_score'] = score

                return res_dic
            res_dic['origin_score'] = score
            res_dic['score'] = score+format_score
            return res_dic
        else:
            res_dic['origin_score'] = 0
            res_dic['score'] = format_score
            return res_dic


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
