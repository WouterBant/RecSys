# def create_prompt(titles, subtitles, title, subtitle):
#     prompt = f"En bruger har for nylig læst artikler:\n"
#     for i, (t, s) in enumerate(zip(titles, subtitles)):
#         prompt += f"Artikel {i+1}:\nTitel: {t}\nUndertekst: {s}\n\n"
#     prompt += f"vil brugeren læse artiklen:\n Titel: {title}\n Undertekst: {subtitle}? (ja/nej)\n"
#     return prompt

def create_prompt(titles, subtitles, title, subtitle):
    prompt = f"En bruger har for nylig læst artikler: "
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        prompt += f"{s}, "
    prompt += f"vil brugeren læse artiklen {subtitle}? (ja/nej)\n"
    return prompt