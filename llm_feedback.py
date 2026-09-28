import math

def generate_feedback(final_score, label, traits, segment_scores=None):
    cue_names = {
        "openness_signal": "Openness cue",
        "conscientiousness_signal": "Conscientiousness cue",
        "extraversion_signal": "Extraversion cue",
        "agreeableness_signal": "Agreeableness cue",
        "neuroticism_signal": "Neuroticism cue",
    }
    valid_cues = {}
    for key, name in cue_names.items():
        try:
            value = float(traits[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid_cues[name] = value

    if final_score is None:
        signal_note = "No reliable video signal was available for this session. Use interview answers and a role-specific rubric for review."
    else:
        try:
            score_text = f"{float(final_score):.2f}/7"
        except (TypeError, ValueError):
            score_text = "not available"
        finite_segments = []
        for score in segment_scores or []:
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                finite_segments.append(score)
        if len(finite_segments) > 1:
            spread = max(finite_segments) - min(finite_segments)
            consistency = "similar across sampled segments" if spread < 0.15 else "varied across sampled segments"
        elif finite_segments:
            consistency = "based on one sampled segment"
        else:
            consistency = "based on available sampled frames"
        signal_note = (
            f"Experimental video signal: {score_text}, {consistency}. "
            "This summarizes selected-frame facial-action and head-pose measurements only; it is not a performance rating."
        )

    if valid_cues:
        highest_name, highest_value = max(valid_cues.items(), key=lambda item: item[1])
        lowest_name, lowest_value = min(valid_cues.items(), key=lambda item: item[1])
        takeaways = [
            f"The model's highest numeric cue was {highest_name} ({highest_value:.2f}/3); its lowest was {lowest_name} ({lowest_value:.2f}/3). These labels are model outputs, not personality findings.",
            "The video model does not assess the candidate's spoken answers, experience, communication skill, or job readiness.",
        ]
    else:
        takeaways = ["No interpretable modeled cues were produced. Base review on the candidate's role-related answers and evidence."]

    return {
        "signal_note": signal_note,
        "candidate_takeaways": takeaways,
        "practice_steps": [
            "Prepare one role-specific STAR example: situation, task, personal action, and measurable result.",
            "Rehearse a concise 60–90 second answer that explains the decision, your contribution, and what changed.",
            "Use a stable camera at eye level and even lighting so the recording is clear; this improves capture quality, not the score's validity.",
        ],
        "followup_note": "Ask every candidate the same role-related behavioral question and score the answer against the same written criteria for evidence, relevance, and outcome.",
        "limits_note": "This experimental video-only prototype does not analyze speech, answers, skills, personality, confidence, or job suitability. Do not use its score to rank, screen out, or make a hiring decision.",
    }