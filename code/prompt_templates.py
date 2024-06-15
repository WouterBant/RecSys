def create_prompt_subtitles(titles, subtitles, title, subtitle):
    prompt = f"En bruger har for nylig læst artikler: "
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        s = s[:min(len(s), 60)]  # clip long subtitles
        prompt += f"{s}, "
    prompt += f"vil brugeren læse artiklen {subtitle}? (ja/nej)\n"
    return prompt

def create_prompt_titles(titles, subtitles, title, subtitle):
    prompt = f"En bruger har for nylig læst artikler: "
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"{t}, "
    prompt += f"vil brugeren læse artiklen {title}? (ja/nej)\n"
    return prompt

def create_prompt_qa_fast(titles):
    prompt = f"En bruger har for nylig læst artikler: "
    for t in titles:
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"{t}, "
    prompt += f"hvilken af følgende artikler er brugeren mest tilbøjelig til at klikke på?\n"
    return prompt