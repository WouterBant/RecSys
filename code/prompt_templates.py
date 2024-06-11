def create_prompt(titles, subtitles, title, subtitle):
    prompt = f"En bruger har for nylig læst artikler: "
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        s = s[:min(len(s), 60)]  # clip long prompts
        prompt += f"{s}, "
    prompt += f"vil brugeren læse artiklen {subtitle}? (ja/nej)\n"
    return prompt